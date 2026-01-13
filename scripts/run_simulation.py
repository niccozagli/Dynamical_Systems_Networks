#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Annotated

import typer

import numpy as np

from dyn_net.dynamical_systems.system_bundle import get_system_bundle
from dyn_net.dynamical_systems.state_transform import get_state_transform
from dyn_net.integrator.jit import integrate_chunked_jit_timed
from dyn_net.integrator.params import EulerMaruyamaParams
from dyn_net.noise import get_noise
from dyn_net.utils.initial_condition import build_initial_condition
from dyn_net.utils.network import build_network_from_config
from dyn_net.utils.state import open_state_writer, close_state_writer
from dyn_net.utils.stats import open_stats_writer, close_stats_writer
from dyn_net.utils.validation import validate_config


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[str, typer.Option(help="Path to base JSON config.")],
    output_dir: Annotated[str, typer.Option(help="Output directory.")] = "results",
    run_id: Annotated[str, typer.Option(help="Run identifier used for output folder.")] = "run_local",
) -> None:
    """Validate choices and parameters in a JSON config."""
    config_path = Path(config)
    config_data = json.loads(config_path.read_text())
    validate_config(config_data)
    # Build adjacency from config before assembling system params.
    A = build_network_from_config(config_data)

    # Resolve system bundle and noise/integrator params.
    system_cfg = config_data["system"]
    noise_cfg = config_data["noise"]
    integrator_cfg = config_data["integrator"]
    system_params = dict(system_cfg.get("params", {}))
    system_params["A"] = A
    _, pF, stats_fn, stats_fields, kernel, kernel_params_builder = get_system_bundle(
        system_cfg["name"], system_params
    )
    _, pG = get_noise(noise_cfg["name"], noise_cfg.get("params", {}))
    p_int = EulerMaruyamaParams.model_validate(integrator_cfg)

    # Seed RNGs for initial conditions and any stochastic pieces.
    run_seed = config_data.get("run", {}).get("seed")
    rng = np.random.default_rng(run_seed)
    if run_seed is not None:
        np.random.seed(int(run_seed))

    n = A.shape[0]
    x0 = build_initial_condition(config_data, n, rng)
    kernel_params = kernel_params_builder(pF, pG)

    # Prepare output writers.
    run_dir = Path(output_dir) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "stats.h5"
    state_path = run_dir / "state.h5"

    stats_writer = open_stats_writer(stats_path, fieldnames=stats_fields)
    state_writer = open_state_writer(state_path, dim=n)
    state_transform = get_state_transform(system_cfg["name"])

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
    (run_dir / "config_used.json").write_text(json.dumps(config_data, indent=2))

if __name__ == "__main__":
    app()
