#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output: Annotated[str, typer.Option(help="Path to write the JSON config.")] = "config.json",
    system: Annotated[str, typer.Option(help="System name.")] = "kuramoto",
    n: Annotated[int, typer.Option(help="Number of nodes/agents.")] = 1000,
    p: Annotated[float, typer.Option(help="Edge probability for Erdos-Renyi.")] = 0.4,
    theta: Annotated[float, typer.Option(help="Kuramoto coupling parameter.")] = 1.0,
    sigma: Annotated[float, typer.Option(help="Noise amplitude.")] = 0.3,
    tmax: Annotated[float, typer.Option(help="Final time.")] = 100.0,
    dt: Annotated[float, typer.Option(help="Time step.")] = 0.01,
    stats_every: Annotated[int, typer.Option(help="Stats sampling interval.")] = 10,
    state_every: Annotated[int, typer.Option(help="State sampling interval.")] = 200,
    ic_type: Annotated[str, typer.Option(help="Initial condition type.")] = "uniform",
) -> None:
    config = {
        "system": {
            "name": system,
            "params": {
                "theta": theta,
            },
        },
        "network": {
            "name": "erdos_renyi",
            "params": {
                "n": n,
                "p": p,
                "directed": False,
            },
        },
        "noise": {
            "name": "additive_gaussian",
            "params": {
                "sigma": sigma,
            },
        },
        "integrator": {
            "tmin": 0.0,
            "tmax": tmax,
            "dt": dt,
            "stats_every": stats_every,
            "state_every": state_every,
            "write_stats_at_start": True,
            "write_state_at_start": True,
        },
        "initial_condition": {
            "type": ic_type,
        },
    }

    path = Path(output)
    path.write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    app()
