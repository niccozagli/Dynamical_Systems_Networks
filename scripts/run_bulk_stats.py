#!/usr/bin/env python3
import copy
import csv
import json
from pathlib import Path
from typing import Annotated, Iterable

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
from dyn_net.utils.table_overrides import apply_overrides

app = typer.Typer(add_completion=False)


def _iter_table_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    delim = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", newline="") as fh:
        lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
        reader = csv.DictReader(lines, delimiter=delim)
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def _count_stats_rows(p_int) -> int:
    n = int((p_int.tmax - p_int.tmin) / p_int.dt)
    count = n // int(p_int.stats_every)
    if p_int.write_stats_at_start:
        count += 1
    return count


def _write_stats_aggregate(path: Path, fieldnames, mean, std, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver="latest") as fh:
        fh.create_dataset(
            "mean",
            data=mean,
            compression="gzip",
            compression_opts=4,
        )
        fh.create_dataset(
            "std",
            data=std,
            compression="gzip",
            compression_opts=4,
        )
        fh.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")
        fh.attrs["count"] = int(count)


def _run_bulk_for_config(
    base_config: dict,
    output_dir: Path,
    run_id: str,
    *,
    graph_realizations: int,
    noise_realizations: int,
    seed_seq: np.random.SeedSequence,
) -> None:
    if graph_realizations <= 0 or noise_realizations <= 0:
        raise ValueError("graph_realizations and noise_realizations must be >= 1.")

    run_dir = output_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    total_runs = int(graph_realizations * noise_realizations)
    graph_seqs = seed_seq.spawn(graph_realizations)

    mean = None
    m2 = None
    count = 0
    ref_steps = None
    ref_time = None

    for g_idx, g_seq in enumerate(graph_seqs, start=1):
        graph_seed = int(g_seq.generate_state(1, dtype=np.uint32)[0])
        graph_config = copy.deepcopy(base_config)
        graph_config.setdefault("network", {}).setdefault("params", {})["seed"] = graph_seed

        A = prepare_network(graph_config)
        n = A.shape[0]

        _, pF, stats_fn, stats_fields, kernel, kernel_params_builder = prepare_system(
            graph_config, A
        )
        pG = prepare_noise(graph_config)
        p_int = prepare_integrator(graph_config)
        kernel_params = kernel_params_builder(pF, pG)

        n_rows = _count_stats_rows(p_int)
        noise_seqs = g_seq.spawn(noise_realizations)

        step_idx = stats_fields.index("step") if "step" in stats_fields else None
        time_idx = stats_fields.index("t") if "t" in stats_fields else None

        for n_idx, n_seq in enumerate(noise_seqs, start=1):
            run_config = copy.deepcopy(graph_config)
            run_seed = int(n_seq.generate_state(1, dtype=np.uint32)[0])
            run_config.setdefault("run", {})["seed"] = run_seed

            rng = prepare_rng(run_config)
            x0 = prepare_initial_condition(run_config, n, rng)

            stats_writer = open_stats_buffer(stats_fields, n_rows)
            try:
                integrate_chunked_jit_timed(
                    kernel,
                    x0,
                    params_int=p_int,
                    kernel_params=kernel_params,
                    stats_fn=stats_fn,
                    stats_writer=stats_writer,
                    stats_params=pF,
                    state_writer=None,
                    state_transform=None,
                )
            finally:
                close_stats_writer(stats_writer)

            stats_arr = stats_writer[1]
            written = stats_writer[3]
            if written != n_rows:
                raise ValueError(
                    f"Stats rows mismatch (expected {n_rows}, got {written})."
                )

            if step_idx is not None:
                steps = stats_arr[:, step_idx]
                if ref_steps is None:
                    ref_steps = steps.copy()
                elif not np.allclose(steps, ref_steps):
                    raise ValueError("Stats 'step' column differs across runs.")
            if time_idx is not None:
                times = stats_arr[:, time_idx]
                if ref_time is None:
                    ref_time = times.copy()
                elif not np.allclose(times, ref_time):
                    raise ValueError("Stats 't' column differs across runs.")

            count += 1
            if mean is None:
                mean = stats_arr.astype(np.float64, copy=True)
                m2 = np.zeros_like(mean)
            else:
                delta = stats_arr - mean
                mean += delta / count
                m2 += delta * (stats_arr - mean)

    if mean is None:
        mean = np.zeros((0, 0), dtype=np.float64)
        std = mean.copy()
    else:
        if count > 1:
            std = np.sqrt(m2 / (count - 1))
        else:
            std = np.zeros_like(mean)

    _write_stats_aggregate(run_dir / "stats_agg.h5", stats_fields, mean, std, count)
    (run_dir / "config_used.json").write_text(json.dumps(base_config, indent=2))
    meta = {
        "graph_realizations": int(graph_realizations),
        "noise_realizations": int(noise_realizations),
        "total_runs": int(total_runs),
        "seed_strategy": "SeedSequence.spawn",
    }
    (run_dir / "bulk_meta.json").write_text(json.dumps(meta, indent=2))


@app.command()
def main(
    config: Annotated[str, typer.Option(help="Path to base JSON config.")],
    output_dir: Annotated[str, typer.Option(help="Output directory.")] = "results",
    run_prefix: Annotated[str, typer.Option(help="Prefix for output folders.")] = "bulk",
    graph_realizations: Annotated[int, typer.Option(help="Number of graph realizations.")] = 1,
    noise_realizations: Annotated[int, typer.Option(help="Number of noise realizations per graph.")] = 1,
    base_seed: Annotated[int | None, typer.Option(help="Optional base seed for reproducibility.")] = None,
    params_table: Annotated[str | None, typer.Option(help="CSV/TSV file with one row per parameter set.")] = None,
    row_index: Annotated[int | None, typer.Option(help="1-based row index in params table.")] = None,
    all_rows: Annotated[bool, typer.Option(help="Process all rows in params table.")] = False,
) -> None:
    """Run many realizations and aggregate stats without writing state files."""
    config_path = Path(config)
    base_config = json.loads(config_path.read_text())
    out_dir = Path(output_dir)

    if params_table:
        table_path = Path(params_table)
        rows = list(_iter_table_rows(table_path))
        if all_rows:
            if not rows:
                raise ValueError(f"No rows found in {table_path}.")
            seed_seq = np.random.SeedSequence(base_seed)
            row_seqs = seed_seq.spawn(len(rows))
            for (idx, row), row_seq in zip(rows, row_seqs, strict=True):
                row_config = copy.deepcopy(base_config)
                apply_overrides(row_config, row)
                row_run_id = row.get("run_id") or f"{run_prefix}_row_{idx}"
                _run_bulk_for_config(
                    row_config,
                    out_dir,
                    row_run_id,
                    graph_realizations=graph_realizations,
                    noise_realizations=noise_realizations,
                    seed_seq=row_seq,
                )
            return
        if row_index is None:
            raise ValueError("--row-index is required unless --all-rows is set.")
        row_matches = [r for r in rows if r[0] == row_index]
        if not row_matches:
            raise ValueError(f"Row {row_index} not found in {table_path}.")
        _, row = row_matches[0]
        row_config = copy.deepcopy(base_config)
        apply_overrides(row_config, row)
        row_run_id = row.get("run_id") or f"{run_prefix}_row_{row_index}"
        seed_seq = np.random.SeedSequence(base_seed)
        _run_bulk_for_config(
            row_config,
            out_dir,
            row_run_id,
            graph_realizations=graph_realizations,
            noise_realizations=noise_realizations,
            seed_seq=seed_seq,
        )
        return

    if all_rows or row_index is not None:
        raise ValueError("--params-table is required for --all-rows/--row-index.")
    seed_seq = np.random.SeedSequence(base_seed)
    _run_bulk_for_config(
        base_config,
        out_dir,
        run_prefix,
        graph_realizations=graph_realizations,
        noise_realizations=noise_realizations,
        seed_seq=seed_seq,
    )


if __name__ == "__main__":
    app()
