#!/usr/bin/env python3

import copy
import json
import secrets
from pathlib import Path
from typing import Annotated

import typer
from dyn_net.integrator.jit import integrate_chunked_jit_timed
from dyn_net.utils.simulation_steps import (
    prepare_initial_condition,
    prepare_integrator,
    prepare_network,
    prepare_noise,
    prepare_rng,
    prepare_state_transform,
    prepare_system,
)
from dyn_net.utils.state import open_state_writer, close_state_writer
from dyn_net.utils.stats import open_stats_writer, close_stats_writer
from dyn_net.utils.table_overrides import apply_overrides, load_table_row


app = typer.Typer(add_completion=False)


def _run_single(
    config_data: dict,
    output_dir: str,
    run_id: str,
    *,
    params_table: str | None = None,
    row_index: int | None = None,
) -> None:
    config_data = copy.deepcopy(config_data)
    # Optionally apply a row-based override table for sweeps/array jobs.
    row_run_id = None
    if params_table:
        if row_index is None:
            raise ValueError("--row-index is required with --params-table")
        row = load_table_row(Path(params_table), row_index)
        row_run_id = row.get("run_id")
        apply_overrides(config_data, row)
        table_applied = True
    else:
        table_applied = False

    # Ensure reproducible seeds are recorded. If not provided, draw from OS entropy.
    net_params = config_data.setdefault("network", {}).setdefault("params", {})
    if net_params.get("seed") is None:
        net_params["seed"] = int(secrets.randbits(32))
    run_cfg = config_data.setdefault("run", {})
    if run_cfg.get("seed") is None:
        run_cfg["seed"] = int(secrets.randbits(32))

    A = prepare_network(config_data)
    n = A.shape[0]

    _, pF, stats_fn, stats_fields, kernel, kernel_params_builder = prepare_system(
        config_data, A
    )
    pG = prepare_noise(config_data)
    p_int = prepare_integrator(config_data)
    rng = prepare_rng(config_data)
    x0 = prepare_initial_condition(config_data, n, rng)
    kernel_params = kernel_params_builder(pF, pG)
    state_transform = prepare_state_transform(config_data["system"]["name"])

    # Prepare output writers.
    # In sweeps, prefer the CLI run_id when explicitly provided (e.g., row/rep suffixes).
    # Fall back to table run_id only when the CLI default is used.
    if table_applied and row_run_id and run_id == "run_local":
        run_id = row_run_id
    elif (not table_applied) and ("run_id" in config_data):
        run_id = config_data["run_id"]
    run_dir = Path(output_dir) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "stats.h5"
    state_path = run_dir / "state.h5"

    # Persist config immediately so it exists even if the run is interrupted.
    (run_dir / "config_used.json").write_text(json.dumps(config_data, indent=2))

    stats_writer = open_stats_writer(stats_path, fieldnames=stats_fields)
    state_writer = open_state_writer(state_path, dim=len(x0))

    # Run the JIT integrator and persist timings + config used.
    timings = None
    try:
        _, timings = integrate_chunked_jit_timed(
            kernel,
            x0,
            params_int=p_int,
            kernel_params=kernel_params,
            stats_fn=stats_fn,
            stats_writer=stats_writer,
            stats_params=pF,
            state_writer=state_writer,
            state_transform=state_transform,
        )
    finally:
        close_stats_writer(stats_writer)
        close_state_writer(state_writer)

    (run_dir / "timings.json").write_text(json.dumps(timings, indent=2))


@app.command()
def main(
    config: Annotated[str, typer.Option(help="Path to base JSON config.")],
    params_table: Annotated[str | None, typer.Option(help="CSV/TSV file with one run per row.")] = None,
    row_index: Annotated[int | None, typer.Option(help="1-based row index in params table.")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory.")] = "results",
    run_id: Annotated[str, typer.Option(help="Run identifier used for output folder.")] = "run_local",
) -> None:
    """Run a single simulation from a JSON config."""
    config_path = Path(config)
    config_data = json.loads(config_path.read_text())
    _run_single(
        config_data,
        output_dir,
        run_id,
        params_table=params_table,
        row_index=row_index,
    )


if __name__ == "__main__":
    app()
