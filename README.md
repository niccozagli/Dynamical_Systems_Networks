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

## Linear Response Experiments

This pipeline separates (1) long unperturbed runs that sample the invariant measure
from (2) response runs that start from sampled states, apply a perturbation, and
aggregate statistics across many realizations.

### 0) Create configs

Use `notebooks/create_config.ipynb` to create:

- An **unperturbed config** (long integration, state saved).
- A **response config** (response integrator settings + `perturbation` block).

### 1) Run unperturbed simulations (one folder per graph)

Logic: each run is an independent graph realization; the runner creates a new
graph seed when one is not provided and writes `state.h5` plus
`config_used.json` under `results/.../<run_id>/`.

Local:

```bash
./scripts/single/run_simulations.sh \
  --config configs/linear_response/unperturbed_runs/poisson/config_...json \
  --output-dir results/linear_response/poisson/critical/n1000/unperturbed_runs \
  --num-graphs 10 \
  --workers 8
```

PBS:

```bash
qsub -e trash -o trash scripts/cluster/single/run_simulations_pbs.sh \
  --config configs/linear_response/unperturbed_runs/poisson/config_...json \
  --output-dir results/linear_response/poisson/critical/n1000/unperturbed_runs \
  --num-graphs 10 \
  --workers 8
```

Each PBS job writes into a job-specific directory:

- `results/.../unperturbed_runs/job_<PBS_JOBID>/...`

### 2) Build the master response table (post-transient only)

Logic: scan all available unperturbed `state.h5` files, keep only rows with
`t >= transient`, and write a master table of candidate initial conditions.
Each row includes a stable `sample_id = "{run_id}:{time_index}"`, so you can
rebuild the table later without losing the ability to track what was used.

Local (typically from the login node on cluster):

```bash
./scripts/run_build_response_table.sh \
  --unperturbed-root results/.../unperturbed_runs/job_<PBS_JOBID> \
  --output-dir results/.../unperturbed_runs \
  --transient 5000
```

This writes:

- `results/.../unperturbed_runs/response_samples.tsv`

### 3) Claim a chunk of unused samples (optionally random)

Logic: you usually do not want to run on the entire master table at once.
`claim_response_chunk.py` creates a smaller chunk table and records which
`sample_id`s have been claimed in `used_sample_ids.txt`.

Sequential (fast, takes the next unused rows):

```bash
./scripts/run_claim_response_chunk.sh \
  --table results/.../unperturbed_runs/response_samples.tsv \
  --output-dir results/.../unperturbed_runs/chunks \
  --chunk-size 50000
```

Randomized (uniform over unused rows via reservoir sampling; scans the table):

```bash
./scripts/run_claim_response_chunk.sh \
  --table results/.../unperturbed_runs/response_samples.tsv \
  --output-dir results/.../unperturbed_runs/chunks \
  --chunk-size 50000 \
  --randomize
```

Outputs:

- Chunk table: `.../chunks/response_samples_chunk_0001.tsv`
- Claimed IDs: `.../chunks/used_sample_ids.txt`

Important:

- Re-running the claim step with the same `--output-dir` will skip already used
  samples.
- You can rebuild the master table later; the `sample_id` scheme keeps it stable.

### 4) Run response experiments on a chunk

Logic per row:

1) Read `state_path` at `time_index`.
2) Rebuild the same graph from `network.params.seed`.
3) Apply the configured perturbation and run both `+epsilon` and `-epsilon`.
4) Aggregate stats in memory and flush worker snapshots periodically.

Local:

```bash
./scripts/run_response_experiments.sh \
  --config configs/linear_response/perturbed_runs/poisson/config_...json \
  --table results/.../unperturbed_runs/chunks/response_samples_chunk_0001.tsv \
  --output-dir results/.../perturbed_runs \
  --workers 8 \
  --flush-every 10
```

PBS:

```bash
qsub -e trash -o trash scripts/cluster/run_response_experiments_pbs.sh \
  --config configs/linear_response/perturbed_runs/poisson/config_...json \
  --table results/.../unperturbed_runs/chunks/response_samples_chunk_0001.tsv \
  --output-dir results/.../perturbed_runs \
  --workers 8 \
  --flush-every 10
```

Each PBS job writes into:

- `results/.../perturbed_runs/job_<PBS_JOBID>/worker_*.h5`

### 5) Aggregate partial results as you go

Logic: aggregation reads worker snapshots and produces a single `aggregate.h5`.
This is safe to run while workers are still running (you will just aggregate the
data flushed so far).

```bash
./scripts/run_response_aggregate.sh \
  --output-dir results/.../perturbed_runs/job_<PBS_JOBID>
```

The aggregate includes:

- `mean_plus`, `std_plus`, `count_plus`
- `mean_minus`, `std_minus`, `count_minus`
- Attributes such as:
  - `perturbation_type`, `perturbation_epsilon`
  - `graph_count`, `sample_count`
  - `runs_done`, `worker_time_s`, `runs_per_s`

### 6) Analyse the response

See:

- `notebooks/prototyping/prototyping_data_analysis.ipynb`

The notebook contains a "Response Experiments Analysis" section that reads
`aggregate.h5` and plots summary statistics.

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
