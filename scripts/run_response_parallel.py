#!/usr/bin/env python3
import copy
import csv
import json
from pathlib import Path
from typing import Annotated, cast

import h5py
import numpy as np
import typer

from dataclasses import dataclass

from dyn_net.integrator.jit import integrate_chunked_jit_timed
from dyn_net.perturbations import get_perturbation
from dyn_net.utils.aggregate import (
    AggregateState,
    count_stats_rows,
    merge_aggregate,
    update_aggregate,
)
from dyn_net.utils.simulation_steps import (
    prepare_integrator,
    prepare_network,
    prepare_noise,
    prepare_rng,
    prepare_system,
)
from dyn_net.utils.stats import open_stats_buffer, close_stats_writer
from dyn_net.utils.table_overrides import apply_overrides

app = typer.Typer(add_completion=False)


def _parse_table(path: Path):
    delim = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", newline="") as fh:
        lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
        reader = csv.DictReader(lines, delimiter=delim)
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def _row_overrides(row: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in row.items() if "." in k}


def _load_state(path: Path, time_index: int) -> np.ndarray:
    with h5py.File(path, "r") as h5f:
        dset = cast(h5py.Dataset, h5f["state"])
        x = np.asarray(dset[time_index, :], dtype=float)
    return x.reshape(-1)


def _apply_perturbation(x: np.ndarray, perturbation_name: str) -> np.ndarray:
    perturb = get_perturbation(perturbation_name)
    return perturb(x)


def _run_single_stats(bundle: dict, x0: np.ndarray, run_config: dict) -> np.ndarray:
    _ = prepare_rng(run_config)
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


def _prepare_bundle(config: dict):
    A = prepare_network(config)
    n = A.shape[0]
    _, pF, stats_fn, stats_fields, kernel, kernel_params_builder = prepare_system(
        config, A
    )
    pG = prepare_noise(config)
    p_int = prepare_integrator(config)
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


@dataclass
class ResponseWorkerFile:
    fh: h5py.File
    mean_plus: h5py.Dataset
    m2_plus: h5py.Dataset
    count_plus: h5py.Dataset
    mean_minus: h5py.Dataset
    m2_minus: h5py.Dataset
    count_minus: h5py.Dataset


def _init_response_worker(path: Path, fieldnames: list[str], n_rows: int) -> ResponseWorkerFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = h5py.File(path, "w", libver="latest")
    n_fields = len(fieldnames)
    chunk_rows = max(1, min(1024, n_rows))
    mean_plus = fh.create_dataset(
        "mean_plus",
        shape=(n_rows, n_fields),
        maxshape=(n_rows, n_fields),
        chunks=(chunk_rows, n_fields),
        dtype=np.float64,
        compression="gzip",
        compression_opts=4,
    )
    m2_plus = fh.create_dataset(
        "m2_plus",
        shape=(n_rows, n_fields),
        maxshape=(n_rows, n_fields),
        chunks=(chunk_rows, n_fields),
        dtype=np.float64,
        compression="gzip",
        compression_opts=4,
    )
    count_plus = fh.create_dataset("count_plus", shape=(), dtype=np.int64)
    mean_minus = fh.create_dataset(
        "mean_minus",
        shape=(n_rows, n_fields),
        maxshape=(n_rows, n_fields),
        chunks=(chunk_rows, n_fields),
        dtype=np.float64,
        compression="gzip",
        compression_opts=4,
    )
    m2_minus = fh.create_dataset(
        "m2_minus",
        shape=(n_rows, n_fields),
        maxshape=(n_rows, n_fields),
        chunks=(chunk_rows, n_fields),
        dtype=np.float64,
        compression="gzip",
        compression_opts=4,
    )
    count_minus = fh.create_dataset("count_minus", shape=(), dtype=np.int64)
    fh.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")
    fh.swmr_mode = True
    fh.flush()
    return ResponseWorkerFile(
        fh=fh,
        mean_plus=mean_plus,
        m2_plus=m2_plus,
        count_plus=count_plus,
        mean_minus=mean_minus,
        m2_minus=m2_minus,
        count_minus=count_minus,
    )


def _flush_response_worker(worker: ResponseWorkerFile, plus: AggregateState, minus: AggregateState) -> None:
    if plus.mean is None or plus.m2 is None:
        worker.mean_plus[...] = 0.0
        worker.m2_plus[...] = 0.0
        worker.count_plus[...] = 0
    else:
        worker.mean_plus[...] = plus.mean
        worker.m2_plus[...] = plus.m2
        worker.count_plus[...] = int(plus.count)
    if minus.mean is None or minus.m2 is None:
        worker.mean_minus[...] = 0.0
        worker.m2_minus[...] = 0.0
        worker.count_minus[...] = 0
    else:
        worker.mean_minus[...] = minus.mean
        worker.m2_minus[...] = minus.m2
        worker.count_minus[...] = int(minus.count)
    worker.fh.flush()


@app.command()
def worker(
    config: Annotated[str, typer.Option(help="Path to base JSON config.")],
    table: Annotated[str, typer.Option(help="Response samples TSV/CSV.")],
    output_dir: Annotated[str, typer.Option(help="Output directory.")] = "results",
    worker_id: Annotated[int, typer.Option(help="Worker index (0-based).") ] = 0,
    num_workers: Annotated[int, typer.Option(help="Total number of workers.") ] = 1,
    flush_every: Annotated[int, typer.Option(help="Flush aggregate every N runs.") ] = 10,
    log_every: Annotated[int, typer.Option(help="Log progress every N runs.") ] = 10,
    base_seed: Annotated[int | None, typer.Option(help="Optional base seed for reproducibility.") ] = None,
) -> None:
    if num_workers <= 0:
        raise ValueError("--num-workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("--worker-id must be in [0, num-workers)")
    if flush_every <= 0:
        raise ValueError("--flush-every must be >= 1")
    if log_every <= 0:
        raise ValueError("--log-every must be >= 1")

    config_path = Path(config)
    base_config = json.loads(config_path.read_text())
    table_path = Path(table)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_path = out_dir / f"worker_{worker_id}.h5"

    seed_rng = np.random.default_rng(base_seed)

    state_plus = AggregateState()
    state_minus = AggregateState()
    runs_since_flush = 0
    runs_done = 0
    n_rows = None
    stats_fields = None
    worker_file: ResponseWorkerFile | None = None

    for row_index, row in _parse_table(table_path):
        if (row_index - 1) % num_workers != worker_id:
            continue

        overrides = _row_overrides(row)
        run_config = copy.deepcopy(base_config)
        if overrides:
            apply_overrides(run_config, overrides)

        bundle = _prepare_bundle(run_config)
        if n_rows is None:
            n_rows = bundle["n_rows"]
            stats_fields = bundle["stats_fields"]
        elif n_rows != bundle["n_rows"]:
            raise ValueError("Integrator stats shape changed across runs.")

        state_path = Path(row["state_path"])
        time_index = int(row["time_index"])
        x0 = _load_state(state_path, time_index)

        eps = float(run_config.get("perturbation", {}).get("epsilon", 0.0))
        if eps == 0.0:
            raise ValueError("perturbation.epsilon must be non-zero in base config")
        perturb_name = run_config.get("perturbation", {}).get("type", "alpha_rot")
        delta = _apply_perturbation(x0, perturb_name)

        x_plus = x0 + eps * delta
        x_minus = x0 - eps * delta

        for sign, x_init in (("plus", x_plus), ("minus", x_minus)):
            run_seed = int(seed_rng.integers(0, 2**32 - 1))
            run_config.setdefault("run", {})["seed"] = run_seed
            stats_arr = _run_single_stats(bundle, x_init, run_config)
            if sign == "plus":
                update_aggregate(state_plus, stats_arr)
            else:
                update_aggregate(state_minus, stats_arr)

        if worker_file is None:
            if stats_fields is None or n_rows is None:
                raise ValueError("Missing stats metadata before flush.")
            worker_file = _init_response_worker(worker_path, list(stats_fields), n_rows)
            worker_file.fh.attrs["worker_id"] = int(worker_id)
            worker_file.fh.attrs["num_workers"] = int(num_workers)
            worker_file.fh.attrs["flush_every"] = int(flush_every)
            if base_seed is not None:
                worker_file.fh.attrs["base_seed"] = int(base_seed)
            worker_file.fh.flush()

        runs_since_flush += 1
        runs_done += 1
        if runs_done % log_every == 0:
            print(
                f"worker {worker_id}: processed {runs_done} runs (table row {row_index})",
                flush=True,
            )
        if runs_since_flush >= flush_every:
            _flush_response_worker(worker_file, state_plus, state_minus)
            runs_since_flush = 0

    if worker_file is None:
        if n_rows is None or stats_fields is None:
            return
        worker_file = _init_response_worker(worker_path, list(stats_fields), n_rows)
        worker_file.fh.attrs["worker_id"] = int(worker_id)
        worker_file.fh.attrs["num_workers"] = int(num_workers)
        worker_file.fh.attrs["flush_every"] = int(flush_every)
        if base_seed is not None:
            worker_file.fh.attrs["base_seed"] = int(base_seed)

    _flush_response_worker(worker_file, state_plus, state_minus)
    worker_file.fh.close()


@app.command()
def aggregate(
    output_dir: Annotated[str, typer.Option(help="Output directory.")] = "results",
    pattern: Annotated[str, typer.Option(help="Worker file pattern.")] = "worker_*.h5",
) -> None:
    out_dir = Path(output_dir)
    worker_paths = sorted(out_dir.glob(pattern))
    if not worker_paths:
        raise ValueError(f"No worker files found in {out_dir} matching {pattern}")

    state_plus = AggregateState()
    state_minus = AggregateState()
    fieldnames = None

    for path in worker_paths:
        with h5py.File(path, "r", libver="latest", swmr=True) as fh:
            if fieldnames is None:
                field_attr = fh.attrs.get("fieldnames")
                if field_attr is None:
                    raise ValueError(f"Missing fieldnames attribute in {path}")
                field_arr = np.asarray(field_attr)
                fieldnames = [s.decode("utf-8") for s in field_arr.tolist()]

            mean_p = np.asarray(cast(h5py.Dataset, fh["mean_plus"])[...])
            m2_p = np.asarray(cast(h5py.Dataset, fh["m2_plus"])[...])
            count_p = int(cast(h5py.Dataset, fh["count_plus"])[()])
            state_plus = merge_aggregate(state_plus, mean_p, m2_p, count_p)

            mean_m = np.asarray(cast(h5py.Dataset, fh["mean_minus"])[...])
            m2_m = np.asarray(cast(h5py.Dataset, fh["m2_minus"])[...])
            count_m = int(cast(h5py.Dataset, fh["count_minus"])[()])
            state_minus = merge_aggregate(state_minus, mean_m, m2_m, count_m)

    if state_plus.mean is None or state_minus.mean is None:
        raise ValueError("No completed worker stats found to aggregate.")
    if state_plus.m2 is None or state_minus.m2 is None:
        raise ValueError("Missing variance accumulators in worker stats.")
    if fieldnames is None:
        raise ValueError("Missing fieldnames in worker files.")

    std_plus = np.sqrt(state_plus.m2 / max(1, state_plus.count - 1)) 
    std_minus = np.sqrt(state_minus.m2 / max(1, state_minus.count - 1))

    agg_path = out_dir / "aggregate.h5"
    with h5py.File(agg_path, "w", libver="latest") as fh:
        fh.create_dataset("mean_plus", data=state_plus.mean, compression="gzip", compression_opts=4)
        fh.create_dataset("std_plus", data=std_plus, compression="gzip", compression_opts=4)
        fh.create_dataset("count_plus", data=int(state_plus.count))
        fh.create_dataset("mean_minus", data=state_minus.mean, compression="gzip", compression_opts=4)
        fh.create_dataset("std_minus", data=std_minus, compression="gzip", compression_opts=4)
        fh.create_dataset("count_minus", data=int(state_minus.count))
        fh.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")
        fh.flush()


if __name__ == "__main__":
    app()
