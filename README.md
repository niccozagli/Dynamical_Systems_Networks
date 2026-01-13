# dynamics-on-network

Python package for studying dynamical systems on networks.

## Installation

This project uses `uv`.

```bash
uv venv --python 3.10
uv sync
```

## How to run a simulation

Create a config file (e.g., `configs/config.json`) and run:

```bash
uv run python scripts/run_simulation.py --config configs/config.json --output-dir results --run-id run_0001
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
