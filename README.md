# dynamics-on-network

Python package for studying dynamical systems on networks.

## Installation

This project uses `uv` to manage the Python environment and dependencies. If you don't have uv, you can easily download it at (https://docs.astral.sh/uv/getting-started/installation/).

This project uses Python 3.10. The version is pinned in `.python-version`. If you don't have this Python version, `uv` can install it for you (macOS/Linux):

```bash
uv python install
uv venv
```

Then you can install all the dependencies needed for this project as:

```bash
uv sync
```

## How to run a simulation

A template notebook on how to run a simulation of Kuramoto on graph is available at 

- `notebooks/template_kuramoto_simulation_run.ipynb`

For the data analysis part of the results you can use the following notebook

- `notebooks/template_data_analysis.ipynb`

Please note that the output of the simulation is continuously update as the simulation runs and can be synchronously read with the data analysis notebook as the simulation is running. 

<details>
<summary>How to add a new dynamical system</summary>

Follow these steps to make a new system available by string name (e.g. `"kuramoto"`):

1) Add the dynamics implementation in `src/dyn_net/dynamical_systems/`.
   - Create a file like `my_system.py` with:
     - a params `BaseModel`
     - a drift function `F(x, t, p)`
     - optional `compute_stats` and `STATS_FIELDS`

2) Register the drift in `src/dyn_net/dynamical_systems/registry.py`.
   - Add your `(F, Params)` pair to `_REGISTRY`.

3) Register stats in `src/dyn_net/dynamical_systems/stats_registry.py`.
   - Add your `(compute_stats, STATS_FIELDS)` pair to `_STATS_REGISTRY`.

4) Provide a JIT kernel if you plan to run JIT-only simulations.
   - Add a `jit_my_system.py` with a kernel + params builder.
   - Register it in `src/dyn_net/dynamical_systems/jit_registry.py`.

5) (Optional) Add an initial condition builder.
   - Implement it in your system file (e.g., `build_initial_condition`).
   - Register it in `src/dyn_net/dynamical_systems/initial_conditions.py`.

6) (Optional) Add a state transform if needed for saved state.
   - Register in `src/dyn_net/dynamical_systems/state_transform.py`.

Once those are in place, `scripts/run_simulation.py` will accept the new system name in the config.
</details>


<!--
## Cluster notes

For this cluster, a shared venv on the scratch disk has been the most reliable option.
Create it on macomp02 so it is local to `/scratchcomp02`:

```bash
SCRATCH=/scratchcomp02/$USER
VENV="$SCRATCH/venvs/dyn-net-py3.10"
CACHE="$SCRATCH/uv-cache"

mkdir -p "$SCRATCH/venvs" "$CACHE"
export UV_CACHE_DIR="$CACHE"
export PIP_CACHE_DIR="$CACHE"

/usr/bin/python3.10 -m venv "$VENV"
source "$VENV/bin/activate"
uv sync -v
```
-->
