#!/usr/bin/env python3
import copy
import json
from pathlib import Path
from typing import Annotated, cast

import h5py
import numpy as np
import typer

from dyn_net.integrator.jit import integrate_chunked_jit_timed
from dyn_net.utils.simulation_steps import (
    prepare_initial_condition,
    prepare_integrator,
    prepare_network,
    prepare_noise,
    prepare_rng,
    prepare_system,
)
from dyn_net.utils.stats import open_stats_buffer, close_stats_writer
from dyn_net.utils.table_overrides import apply_overrides, load_table_row
from dyn_net.utils.aggregate import (
    AggregateState,
    WorkerFile,
    assign_worker,
    count_stats_rows,
    flush_worker,
    init_worker_file,
    merge_aggregate,
    update_aggregate,
)

app = typer.Typer(add_completion=False)


def _load_base_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def _apply_table_overrides(
    base_config: dict,
    params_table: str | None,
    row_index: int | None,
) -> tuple[dict, str | None, bool]:
    if not params_table:
        return base_config, None, False
    if row_index is None:
        raise ValueError("--row-index is required with --params-table")
    row = load_table_row(Path(params_table), row_index)
    row_run_id = row.get("run_id")
    apply_overrides(base_config, row)
    return base_config, row_run_id, True


def _prepare_graph_bundle(graph_config: dict):
    A = prepare_network(graph_config)
    n = A.shape[0]
    _, pF, stats_fn, stats_fields, kernel, kernel_params_builder = prepare_system(
        graph_config, A
    )
    pG = prepare_noise(graph_config)
    p_int = prepare_integrator(graph_config)
    kernel_params = kernel_params_builder(pF, pG)
    n_rows = count_stats_rows(p_int)
    return {
        "A": A,
        "n": n,
        "pF": pF,
        "stats_fn": stats_fn,
        "stats_fields": stats_fields,
        "kernel": kernel,
        "kernel_params": kernel_params,
        "p_int": p_int,
        "n_rows": n_rows,
    }


def _run_single_stats(bundle: dict, run_config: dict) -> np.ndarray:
    rng = prepare_rng(run_config)
    x0 = prepare_initial_condition(run_config, bundle["n"], rng)
    stats_writer = open_stats_buffer(bundle["stats_fields"], bundle["n_rows"])
    try:
        integrate_chunked_jit_timed(
            bundle["kernel"],
            x0,
            params_int=bundle["p_int"],
            kernel_params=bundle["kernel_params"],
            stats_fn=bundle["stats_fn"],
            stats_writer=stats_writer,
            stats_params=bundle["pF"],
            state_writer=None,
            state_transform=None,
        )
    finally:
        close_stats_writer(stats_writer)
    stats_arr = stats_writer[1]
    written = stats_writer[3]
    if written != bundle["n_rows"]:
        raise ValueError(
            f"Stats rows mismatch (expected {bundle['n_rows']}, got {written})."
        )
    return stats_arr


@app.command()
def worker(
    config: Annotated[str, typer.Option(help="Path to base JSON config.")],
    output_dir: Annotated[str, typer.Option(help="Output directory.")] = "results",
    run_id: Annotated[str, typer.Option(help="Run identifier used for output folder.")] = "bulk",
    graph_realizations: Annotated[int, typer.Option(help="Number of graph realizations.")] = 1,
    noise_realizations: Annotated[int, typer.Option(help="Number of noise realizations per graph.")] = 1,
    worker_id: Annotated[int, typer.Option(help="Worker index (0-based).")] = 0,
    num_workers: Annotated[int, typer.Option(help="Total number of workers.")] = 1,
    flush_every: Annotated[int, typer.Option(help="Flush aggregate every N runs.")] = 10,
    base_seed: Annotated[int | None, typer.Option(help="Optional base seed for reproducibility.")] = None,
    params_table: Annotated[str | None, typer.Option(help="CSV/TSV file with one row per parameter set.")] = None,
    row_index: Annotated[int | None, typer.Option(help="1-based row index in params table.")] = None,
) -> None:
    if graph_realizations <= 0 or noise_realizations <= 0:
        raise ValueError("--graph-realizations and --noise-realizations must be >= 1")
    if num_workers <= 0:
        raise ValueError("--num-workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("--worker-id must be in [0, num-workers)")
    if flush_every <= 0:
        raise ValueError("--flush-every must be >= 1")

    config_path = Path(config)
    base_config = _load_base_config(config_path)
    base_config, row_run_id, table_applied = _apply_table_overrides(
        base_config, params_table, row_index
    )
    if row_run_id:
        run_id = row_run_id

    run_dir = Path(output_dir) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    worker_path = run_dir / f"worker_{worker_id}.h5"

    seed_seq = np.random.SeedSequence(base_seed)
    graph_seqs = seed_seq.spawn(graph_realizations)

    state = AggregateState()
    runs_since_flush = 0

    n_rows = None
    stats_fields = None
    worker_file = None

    for g_idx, g_seq in enumerate(graph_seqs):
        graph_seed = int(g_seq.generate_state(1, dtype=np.uint32)[0])
        graph_config = copy.deepcopy(base_config)
        graph_config.setdefault("network", {}).setdefault("params", {})["seed"] = graph_seed

        bundle = _prepare_graph_bundle(graph_config)
        if n_rows is None:
            n_rows = bundle["n_rows"]
        elif n_rows != bundle["n_rows"]:
            raise ValueError("Integrator stats shape changed across runs.")
        stats_fields = bundle["stats_fields"]

        noise_seqs = g_seq.spawn(noise_realizations)
        for n_idx, n_seq in enumerate(noise_seqs):
            run_index = g_idx * noise_realizations + n_idx
            if not assign_worker(run_index, worker_id, num_workers):
                continue

            run_config = copy.deepcopy(graph_config)
            run_seed = int(n_seq.generate_state(1, dtype=np.uint32)[0])
            run_config.setdefault("run", {})["seed"] = run_seed

            if worker_file is None:
                worker_file = init_worker_file(worker_path, list(stats_fields), n_rows)
                worker_file.fh.attrs["worker_id"] = int(worker_id)
                worker_file.fh.attrs["num_workers"] = int(num_workers)
                worker_file.fh.attrs["graph_realizations"] = int(graph_realizations)
                worker_file.fh.attrs["noise_realizations"] = int(noise_realizations)
                worker_file.fh.attrs["flush_every"] = int(flush_every)
                if base_seed is not None:
                    worker_file.fh.attrs["base_seed"] = int(base_seed)
                if table_applied:
                    assert row_index is not None
                    worker_file.fh.attrs["params_table"] = str(params_table)
                    worker_file.fh.attrs["row_index"] = int(row_index)
                worker_file.fh.flush()

            stats_arr = _run_single_stats(bundle, run_config)
            update_aggregate(state, stats_arr)
            runs_since_flush += 1
            if runs_since_flush >= flush_every:
                flush_worker(worker_file, state)
                runs_since_flush = 0

    if worker_file is not None:
        flush_worker(worker_file, state)
        worker_file.fh.close()


@app.command()
def aggregate(
    output_dir: Annotated[str, typer.Option(help="Output directory.")] = "results",
    run_id: Annotated[str, typer.Option(help="Run identifier used for output folder.")] = "bulk",
    pattern: Annotated[str, typer.Option(help="Worker file pattern.")] = "worker_*.h5",
) -> None:
    run_dir = Path(output_dir) / str(run_id)
    worker_paths = sorted(run_dir.glob(pattern))
    if not worker_paths:
        raise ValueError(f"No worker files found in {run_dir} matching {pattern}")

    state = AggregateState()
    fieldnames = None

    for path in worker_paths:
        with h5py.File(path, "r", libver="latest", swmr=True) as fh:
            if fieldnames is None:
                field_attr = fh.attrs.get("fieldnames")
                if field_attr is None:
                    raise ValueError(f"Missing fieldnames attribute in {path}")
                field_arr = np.asarray(field_attr)
                fieldnames = [s.decode("utf-8") for s in field_arr.tolist()]
            worker_mean = np.asarray(cast(h5py.Dataset, fh["mean"])[...])
            worker_m2 = np.asarray(cast(h5py.Dataset, fh["m2"])[...])
            worker_count = int(cast(h5py.Dataset, fh["count"])[()])
            state = merge_aggregate(state, worker_mean, worker_m2, worker_count)

    if state.mean is None:
        raise ValueError("No completed worker stats found to aggregate.")
    assert state.m2 is not None

    if state.count > 1:
        std = np.sqrt(state.m2 / (state.count - 1))
    else:
        std = np.zeros_like(state.mean)

    agg_path = run_dir / "aggregate.h5"
    with h5py.File(agg_path, "w", libver="latest") as fh:
        fh.create_dataset(
            "mean",
            data=state.mean,
            compression="gzip",
            compression_opts=4,
        )
        fh.create_dataset(
            "std",
            data=std,
            compression="gzip",
            compression_opts=4,
        )
        fh.create_dataset("count", data=int(state.count))
        fh.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")
        fh.flush()


if __name__ == "__main__":
    app()
