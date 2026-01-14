# dynamics-on-network

Python package for studying dynamical systems on networks.

## Installation

This project uses Poetry.

```bash
poetry install
```

## How to run a simulation

Create a config file (e.g., `configs/config.json`) and run:

```bash
poetry run python scripts/run_simulation.py --config configs/config.json --output-dir results --run-id run_0001
```

## Cluster notes

On the cluster, install Poetry and ensure `~/.local/bin` is on your `PATH`:

```bash
python3 -m pip install --user poetry
export PATH="$HOME/.local/bin:$PATH"
```

For this cluster, a shared venv on the scratch disk has been the most reliable option.
Create it on macomp02 so it is local to `/scratchcomp02`:

```bash
SCRATCH=/scratchcomp02/$USER
VENV="$SCRATCH/venvs/dyn-net-py3.10"
CACHE="$SCRATCH/pypoetry-cache"

mkdir -p "$SCRATCH/venvs" "$CACHE"
export POETRY_CACHE_DIR="$CACHE"
export PIP_CACHE_DIR="$CACHE"

/usr/bin/python3.10 -m venv "$VENV"
source "$VENV/bin/activate"
poetry config virtualenvs.create false
poetry install -vv
```

## How to add a new dynamical system

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
