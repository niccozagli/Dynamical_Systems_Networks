#!/usr/bin/env python3
import copy
import json
import time
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer

from dyn_net.integrator.jit import integrate_chunked_jit_timed
from dyn_net.perturbations import get_perturbation
from dyn_net.utils.aggregate import AggregateState, count_stats_rows, merge_aggregate, update_aggregate
from dyn_net.utils.simulation_steps import (
    prepare_integrator,
    prepare_network,
    prepare_noise,
    prepare_rng,
    prepare_system,
)
from dyn_net.utils.stats import open_stats_buffer, close_stats_writer


app = typer.Typer(add_completion=False)


def _deep_update(target: dict, override: dict) -> dict:
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], val)
        else:
            target[key] = val
    return target


def _load_run_config(unperturbed_dir: Path, response_config_path: Path) -> dict:
    base_config = json.loads((unperturbed_dir / "config_used.json").read_text())
    response_cfg = json.loads(response_config_path.read_text())
    run_config = copy.deepcopy(base_config)
    _deep_update(run_config, response_cfg)

    # Always preserve the graph seed from the unperturbed run.
    base_seed_val = base_config.get("network", {}).get("params", {}).get("seed")
    if base_seed_val is not None:
        run_config.setdefault("network", {}).setdefault("params", {})["seed"] = base_seed_val

    eps = float(run_config.get("perturbation", {}).get("epsilon", 0.0))
    if eps == 0.0:
        raise ValueError("perturbation.epsilon must be non-zero in response config")
    perturb_name = run_config.get("perturbation", {}).get("type")
    if not perturb_name:
        raise ValueError("perturbation.type must be provided in response config")

    return run_config


def _prepare_bundle(config: dict) -> dict:
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


def _init_worker_file(path: Path, fieldnames: list[str], n_rows: int) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = h5py.File(path, "a", libver="latest")
    shape = (int(n_rows), int(len(fieldnames)))
    chunk_rows = max(1, min(1024, shape[0]))

    if "mean_plus" not in fh:
        fh.create_dataset(
            "mean_plus",
            shape=shape,
            maxshape=shape,
            chunks=(chunk_rows, shape[1]),
            dtype=np.float64,
            compression="gzip",
            compression_opts=4,
        )
        fh.create_dataset(
            "m2_plus",
            shape=shape,
            maxshape=shape,
            chunks=(chunk_rows, shape[1]),
            dtype=np.float64,
            compression="gzip",
            compression_opts=4,
        )
        fh.create_dataset("count_plus", shape=(), dtype=np.int64)
        fh.create_dataset(
            "mean_minus",
            shape=shape,
            maxshape=shape,
            chunks=(chunk_rows, shape[1]),
            dtype=np.float64,
            compression="gzip",
            compression_opts=4,
        )
        fh.create_dataset(
            "m2_minus",
            shape=shape,
            maxshape=shape,
            chunks=(chunk_rows, shape[1]),
            dtype=np.float64,
            compression="gzip",
            compression_opts=4,
        )
        fh.create_dataset("count_minus", shape=(), dtype=np.int64)
        fh.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")
        fh.flush()

    if not fh.swmr_mode:
        fh.flush()
        fh.swmr_mode = True
        fh.flush()

    return fh


def _load_aggregate_state(fh: h5py.File) -> tuple[AggregateState, AggregateState, int]:
    runs_done = int(fh.attrs.get("runs_done", 0))
    state_plus = AggregateState()
    state_minus = AggregateState()
    if "mean_plus" in fh:
        count_plus = int(fh["count_plus"][()])
        if count_plus > 0:
            state_plus.mean = np.asarray(fh["mean_plus"][...])
            state_plus.m2 = np.asarray(fh["m2_plus"][...])
            state_plus.count = count_plus
    if "mean_minus" in fh:
        count_minus = int(fh["count_minus"][()])
        if count_minus > 0:
            state_minus.mean = np.asarray(fh["mean_minus"][...])
            state_minus.m2 = np.asarray(fh["m2_minus"][...])
            state_minus.count = count_minus
    return state_plus, state_minus, runs_done


def _flush_worker(
    fh: h5py.File,
    plus: AggregateState,
    minus: AggregateState,
    *,
    runs_done: int,
    worker_start: float,
    sample_count: int,
) -> None:
    if plus.mean is None or plus.m2 is None:
        fh["mean_plus"][...] = 0.0
        fh["m2_plus"][...] = 0.0
        fh["count_plus"][...] = 0
    else:
        fh["mean_plus"][...] = plus.mean
        fh["m2_plus"][...] = plus.m2
        fh["count_plus"][...] = int(plus.count)

    if minus.mean is None or minus.m2 is None:
        fh["mean_minus"][...] = 0.0
        fh["m2_minus"][...] = 0.0
        fh["count_minus"][...] = 0
    else:
        fh["mean_minus"][...] = minus.mean
        fh["m2_minus"][...] = minus.m2
        fh["count_minus"][...] = int(minus.count)

    fh.attrs["runs_done"] = int(runs_done)
    fh.attrs["sample_count"] = int(sample_count)
    elapsed = time.perf_counter() - worker_start
    fh.attrs["worker_time_s"] = float(elapsed)
    fh.attrs["runs_per_s"] = float(runs_done / elapsed) if elapsed > 0 else 0.0

    if not fh.swmr_mode:
        fh.flush()
        fh.swmr_mode = True
    fh.flush()


def _worker_rng(base_seed: int | None, total_workers: int, global_worker_id: int) -> np.random.Generator:
    if base_seed is None:
        return np.random.default_rng()
    seed_seq = np.random.SeedSequence(base_seed)
    child = seed_seq.spawn(total_workers)[global_worker_id]
    return np.random.default_rng(child)


def _assert_output_compatible(out_dir: Path, perturb_name: str, eps: float) -> None:
    cfg_path = out_dir / "config_used.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to parse existing config_used.json in {out_dir}: {exc}") from exc
        pert_cfg = cfg.get("perturbation", {})
        cfg_type = pert_cfg.get("type")
        if cfg_type is not None and str(cfg_type) != str(perturb_name):
            raise ValueError(
                f"Output dir already contains perturbation_type={cfg_type}; "
                f"expected {perturb_name}. Choose a different output dir."
            )
        cfg_eps = pert_cfg.get("epsilon")
        if cfg_eps is not None and float(cfg_eps) != float(eps):
            raise ValueError(
                f"Output dir already contains perturbation_epsilon={cfg_eps}; "
                f"expected {eps}. Choose a different output dir."
            )
        return

    worker_paths = sorted(out_dir.glob("worker_*.h5"))
    if not worker_paths:
        return
    with h5py.File(worker_paths[0], "r", libver="latest") as fh:
        ptype = fh.attrs.get("perturbation_type")
        if ptype is not None and str(ptype) != str(perturb_name):
            raise ValueError(
                f"Output dir already contains perturbation_type={ptype}; "
                f"expected {perturb_name}. Choose a different output dir."
            )
        peps = fh.attrs.get("perturbation_epsilon")
        if peps is not None and float(peps) != float(eps):
            raise ValueError(
                f"Output dir already contains perturbation_epsilon={peps}; "
                f"expected {eps}. Choose a different output dir."
            )


@app.command()
def worker(
    unperturbed_dir: Annotated[str, typer.Option(help="Folder with state.h5 and config_used.json.")],
    response_config: Annotated[str, typer.Option(help="JSON config with perturbation + integrator overrides.")],
    output_dir: Annotated[str, typer.Option(help="Output directory; writes to <output_dir>/response/.")],
    transient: Annotated[float, typer.Option(help="Only use samples with t >= transient.")] = 0.0,
    worker_id: Annotated[int, typer.Option(help="Worker index (0-based).")] = 0,
    num_workers: Annotated[int, typer.Option(help="Total workers per job.")] = 1,
    job_id: Annotated[int, typer.Option(help="Job index (0-based).")] = 0,
    num_jobs: Annotated[int, typer.Option(help="Total number of jobs.")] = 1,
    flush_every: Annotated[int, typer.Option(help="Flush worker every N samples.")] = 50,
    base_seed: Annotated[int | None, typer.Option(help="Optional RNG seed for response runs.")] = None,
    sample_dt: Annotated[float | None, typer.Option(help="Optional sampling interval for state.h5 (in time units).")] = None,
) -> None:
    if num_workers <= 0 or num_jobs <= 0:
        raise ValueError("--num-workers and --num-jobs must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("--worker-id must be in [0, num-workers)")
    if job_id < 0 or job_id >= num_jobs:
        raise ValueError("--job-id must be in [0, num-jobs)")
    if flush_every <= 0:
        raise ValueError("--flush-every must be >= 1")
    if sample_dt is not None and sample_dt <= 0:
        raise ValueError("--sample-dt must be > 0")

    unperturbed_dir = Path(unperturbed_dir)
    state_path = unperturbed_dir / "state.h5"
    cfg_path = unperturbed_dir / "config_used.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.h5 in {unperturbed_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_used.json in {unperturbed_dir}")

    run_config = _load_run_config(unperturbed_dir, Path(response_config))
    perturb_name = str(run_config["perturbation"]["type"])
    eps = float(run_config["perturbation"]["epsilon"])
    perturb = get_perturbation(perturb_name)

    bundle = _prepare_bundle(run_config)
    fieldnames = list(bundle["stats_fields"])

    total_workers = num_workers * num_jobs
    global_worker_id = job_id * num_workers + worker_id

    out_dir = Path(output_dir) / "response"
    out_dir.mkdir(parents=True, exist_ok=True)
    _assert_output_compatible(out_dir, perturb_name, eps)
    (out_dir / "config_used.json").write_text(
        json.dumps({k: v for k, v in run_config.items() if k != "initial_condition"}, indent=2)
    )

    worker_path = out_dir / f"worker_{global_worker_id:04d}.h5"
    fh = _init_worker_file(worker_path, fieldnames, bundle["n_rows"])

    fh.attrs["worker_id"] = int(worker_id)
    fh.attrs["num_workers"] = int(num_workers)
    fh.attrs["job_id"] = int(job_id)
    fh.attrs["num_jobs"] = int(num_jobs)
    fh.attrs["global_worker_id"] = int(global_worker_id)
    fh.attrs["total_workers"] = int(total_workers)
    fh.attrs["perturbation_type"] = str(perturb_name)
    fh.attrs["perturbation_epsilon"] = float(eps)
    if base_seed is not None:
        fh.attrs["base_seed"] = int(base_seed)
    fh.flush()

    state_plus, state_minus, runs_done = _load_aggregate_state(fh)
    rng = _worker_rng(base_seed, total_workers, global_worker_id)

    worker_start = time.perf_counter()
    sample_count = int(runs_done)
    assigned_seen = 0

    with h5py.File(state_path, "r") as h5f:
        state_dset = h5f["state"]
        time_dset = h5f["time"]
        times = np.asarray(time_dset[...], dtype=float)
        indices = np.where(times >= float(transient))[0]
        if indices.size == 0:
            raise ValueError("No samples found with t >= transient.")
        if sample_dt is not None:
            sampled = []
            next_t = times[int(indices[0])]
            tol = 1e-12
            for idx in indices:
                t = times[int(idx)]
                if t + tol >= next_t:
                    sampled.append(int(idx))
                    next_t = t + float(sample_dt)
            if not sampled:
                raise ValueError("No samples matched --sample-dt; check dt and sample-dt.")
            indices = np.asarray(sampled, dtype=int)

        for pos, idx in enumerate(indices):
            if pos % total_workers != global_worker_id:
                continue
            if assigned_seen < runs_done:
                assigned_seen += 1
                continue

            x0 = np.asarray(state_dset[int(idx), :], dtype=float).reshape(-1)
            delta = perturb(x0)
            x_plus = x0 + eps * delta
            x_minus = x0 - eps * delta

            run_seed = int(rng.integers(0, 2**32 - 1))
            run_config.setdefault("run", {})["seed"] = run_seed
            for x_init, target in ((x_plus, state_plus), (x_minus, state_minus)):
                stats_arr = _run_single_stats(bundle, x_init, run_config)
                update_aggregate(target, stats_arr)

            runs_done += 1
            sample_count += 1
            assigned_seen += 1

            if runs_done % flush_every == 0:
                _flush_worker(
                    fh,
                    state_plus,
                    state_minus,
                    runs_done=runs_done,
                    worker_start=worker_start,
                    sample_count=sample_count,
                )

    _flush_worker(
        fh,
        state_plus,
        state_minus,
        runs_done=runs_done,
        worker_start=worker_start,
        sample_count=sample_count,
    )
    fh.close()


@app.command()
def aggregate(
    output_dir: Annotated[str, typer.Option(help="Output directory (contains response/).")],
    pattern: Annotated[str, typer.Option(help="Worker file pattern.")] = "worker_*.h5",
) -> None:
    out_dir = Path(output_dir) / "response"
    worker_paths = sorted(out_dir.glob(pattern))
    if not worker_paths:
        raise ValueError(f"No worker files found in {out_dir} matching {pattern}")

    state_plus = AggregateState()
    state_minus = AggregateState()
    fieldnames = None
    sample_count = 0
    epsilon_ref: float | None = None
    perturbation_type_ref: str | None = None

    for path in worker_paths:
        with h5py.File(path, "r", libver="latest", swmr=True) as fh:
            if fieldnames is None:
                field_attr = fh.attrs.get("fieldnames")
                if field_attr is None:
                    raise ValueError(f"Missing fieldnames attribute in {path}")
                fieldnames = [s.decode("utf-8") for s in np.asarray(field_attr).tolist()]

            mean_p = np.asarray(fh["mean_plus"][...])
            m2_p = np.asarray(fh["m2_plus"][...])
            count_p = int(fh["count_plus"][()])
            state_plus = merge_aggregate(state_plus, mean_p, m2_p, count_p)

            mean_m = np.asarray(fh["mean_minus"][...])
            m2_m = np.asarray(fh["m2_minus"][...])
            count_m = int(fh["count_minus"][()])
            state_minus = merge_aggregate(state_minus, mean_m, m2_m, count_m)

            sample_count += int(fh.attrs.get("sample_count", count_p))
            eps = fh.attrs.get("perturbation_epsilon")
            if eps is not None:
                eps_val = float(eps)
                if epsilon_ref is None:
                    epsilon_ref = eps_val
                elif abs(epsilon_ref - eps_val) > 0.0:
                    raise ValueError(
                        f"Mismatched perturbation_epsilon across workers: {epsilon_ref} vs {eps_val}"
                    )
            ptype = fh.attrs.get("perturbation_type")
            if ptype is not None:
                ptype_val = str(ptype)
                if perturbation_type_ref is None:
                    perturbation_type_ref = ptype_val
                elif perturbation_type_ref != ptype_val:
                    raise ValueError(
                        f"Mismatched perturbation_type across workers: {perturbation_type_ref} vs {ptype_val}"
                    )

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
        if perturbation_type_ref is not None:
            fh.attrs["perturbation_type"] = perturbation_type_ref
        if epsilon_ref is not None:
            fh.attrs["perturbation_epsilon"] = float(epsilon_ref)
        fh.attrs["sample_count"] = int(sample_count)
        fh.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")
        fh.flush()


if __name__ == "__main__":
    app()
