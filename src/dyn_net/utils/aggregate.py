from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class AggregateState:
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    count: int = 0


@dataclass
class WorkerFile:
    fh: h5py.File
    mean_dset: h5py.Dataset
    m2_dset: h5py.Dataset
    count_dset: h5py.Dataset


def count_stats_rows(p_int) -> int:
    n = int((p_int.tmax - p_int.tmin) / p_int.dt)
    count = n // int(p_int.stats_every)
    if p_int.write_stats_at_start:
        count += 1
    return count


def assign_worker(run_index: int, worker_id: int, num_workers: int) -> bool:
    return (run_index % num_workers) == worker_id


def init_worker_file(path: Path, fieldnames: list[str], n_rows: int) -> WorkerFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = h5py.File(path, "w", libver="latest")
    n_fields = len(fieldnames)
    chunk_rows = max(1, min(1024, n_rows))
    mean_dset = fh.create_dataset(
        "mean",
        shape=(n_rows, n_fields),
        maxshape=(n_rows, n_fields),
        chunks=(chunk_rows, n_fields),
        dtype=np.float64,
        compression="gzip",
        compression_opts=4,
    )
    m2_dset = fh.create_dataset(
        "m2",
        shape=(n_rows, n_fields),
        maxshape=(n_rows, n_fields),
        chunks=(chunk_rows, n_fields),
        dtype=np.float64,
        compression="gzip",
        compression_opts=4,
    )
    count_dset = fh.create_dataset("count", shape=(), dtype=np.int64)
    fh.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")
    fh.swmr_mode = True
    fh.flush()
    return WorkerFile(fh=fh, mean_dset=mean_dset, m2_dset=m2_dset, count_dset=count_dset)


def flush_worker(worker: WorkerFile, state: AggregateState) -> None:
    worker.mean_dset[...] = state.mean
    worker.m2_dset[...] = state.m2
    worker.count_dset[...] = int(state.count)
    worker.fh.flush()


def update_aggregate(state: AggregateState, stats_arr: np.ndarray) -> None:
    state.count += 1
    if state.mean is None:
        state.mean = stats_arr.astype(np.float64, copy=True)
        state.m2 = np.zeros_like(state.mean)
        return
    delta = stats_arr - state.mean
    state.mean += delta / state.count
    state.m2 += delta * (stats_arr - state.mean)


def merge_aggregate(
    state: AggregateState,
    mean_b: np.ndarray,
    m2_b: np.ndarray,
    count_b: int,
) -> AggregateState:
    if count_b <= 0:
        return state
    if state.mean is None:
        return AggregateState(mean=mean_b.copy(), m2=m2_b.copy(), count=int(count_b))
    total = state.count + count_b
    delta = mean_b - state.mean
    state.mean = state.mean + delta * (count_b / total)
    state.m2 = state.m2 + m2_b + delta * delta * (state.count * count_b / total)
    state.count = int(total)
    return state
