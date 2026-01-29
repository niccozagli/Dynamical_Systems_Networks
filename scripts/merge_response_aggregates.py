#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from dyn_net.utils.aggregate import AggregateState, merge_aggregate


def _load_aggregate(path: Path, key_prefix: str):
    with h5py.File(path, "r", libver="latest", swmr=True) as fh:
        mean = np.asarray(cast(h5py.Dataset, fh[f"mean_{key_prefix}"])[...])
        std = np.asarray(cast(h5py.Dataset, fh[f"std_{key_prefix}"])[...])
        count = int(cast(h5py.Dataset, fh[f"count_{key_prefix}"])[()])
        if count > 1:
            m2 = std * std * (count - 1)
        else:
            m2 = np.zeros_like(mean)
        attrs = dict(fh.attrs)
        run_ids = np.asarray(fh["run_ids"][...]) if "run_ids" in fh else None
        run_counts = np.asarray(fh["run_counts"][...]) if "run_counts" in fh else None
    return mean, m2, count, attrs, run_ids, run_counts


def _merge_run_counts(merged: dict[str, int], run_ids, run_counts) -> dict[str, int]:
    if run_ids is None or run_counts is None:
        return merged
    for rid, cnt in zip(run_ids.tolist(), run_counts.tolist()):
        key = str(rid)
        merged[key] = merged.get(key, 0) + int(cnt)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple response aggregate.h5 files into a single aggregate."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to aggregate.h5 files to merge",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output aggregate.h5 path",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    state_plus = AggregateState()
    state_minus = AggregateState()
    fieldnames = None
    perturb_type = None
    perturb_eps = None
    run_counts: dict[str, int] = {}
    total_worker_time_s = 0.0
    total_runs_done = 0

    for path in inputs:
        mean_p, m2_p, count_p, attrs_p, run_ids, run_cnts = _load_aggregate(path, "plus")
        mean_m, m2_m, count_m, attrs_m, _, _ = _load_aggregate(path, "minus")

        if fieldnames is None:
            fn = attrs_p.get("fieldnames")
            if fn is None:
                raise ValueError(f"Missing fieldnames in {path}")
            fieldnames = [s.decode("utf-8") for s in np.asarray(fn).tolist()]
        else:
            fn = attrs_p.get("fieldnames")
            if fn is not None:
                fn_list = [s.decode("utf-8") for s in np.asarray(fn).tolist()]
                if fn_list != fieldnames:
                    raise ValueError(f"Fieldnames mismatch in {path}")

        ptype = attrs_p.get("perturbation_type")
        peps = attrs_p.get("perturbation_epsilon")
        if ptype is not None:
            if perturb_type is None:
                perturb_type = str(ptype)
            elif perturb_type != str(ptype):
                raise ValueError(f"perturbation_type mismatch: {perturb_type} vs {ptype}")
        if peps is not None:
            peps_val = float(peps)
            if perturb_eps is None:
                perturb_eps = peps_val
            elif abs(perturb_eps - peps_val) > 0.0:
                raise ValueError(f"perturbation_epsilon mismatch: {perturb_eps} vs {peps_val}")

        state_plus = merge_aggregate(state_plus, mean_p, m2_p, count_p)
        state_minus = merge_aggregate(state_minus, mean_m, m2_m, count_m)

        run_counts = _merge_run_counts(run_counts, run_ids, run_cnts)

        total_worker_time_s += float(attrs_p.get("worker_time_s", 0.0))
        total_runs_done += int(attrs_p.get("runs_done", 0))

    if state_plus.mean is None or state_minus.mean is None:
        raise ValueError("No valid aggregates to merge.")
    if state_plus.m2 is None or state_minus.m2 is None:
        raise ValueError("Missing variance accumulators in merged aggregates.")
    if fieldnames is None:
        raise ValueError("Missing fieldnames after merge.")

    std_plus = np.sqrt(state_plus.m2 / max(1, state_plus.count - 1))
    std_minus = np.sqrt(state_minus.m2 / max(1, state_minus.count - 1))

    with h5py.File(output, "w", libver="latest") as fh:
        fh.create_dataset("mean_plus", data=state_plus.mean, compression="gzip", compression_opts=4)
        fh.create_dataset("std_plus", data=std_plus, compression="gzip", compression_opts=4)
        fh.create_dataset("count_plus", data=int(state_plus.count))
        fh.create_dataset("mean_minus", data=state_minus.mean, compression="gzip", compression_opts=4)
        fh.create_dataset("std_minus", data=std_minus, compression="gzip", compression_opts=4)
        fh.create_dataset("count_minus", data=int(state_minus.count))
        if run_counts:
            run_ids_sorted = sorted(run_counts)
            counts = np.asarray([run_counts[r] for r in run_ids_sorted], dtype=np.int64)
            fh.create_dataset(
                "run_ids",
                data=np.asarray(run_ids_sorted, dtype=h5py.string_dtype(encoding="utf-8")),
                compression="gzip",
                compression_opts=4,
            )
            fh.create_dataset("run_counts", data=counts, compression="gzip", compression_opts=4)
        fh.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")
        if perturb_type is not None:
            fh.attrs["perturbation_type"] = perturb_type
        if perturb_eps is not None:
            fh.attrs["perturbation_epsilon"] = float(perturb_eps)
        fh.attrs["graph_count"] = int(len(run_counts))
        fh.attrs["sample_count"] = int(sum(run_counts.values())) if run_counts else int(state_plus.count)
        fh.attrs["worker_time_s"] = float(total_worker_time_s)
        fh.attrs["runs_done"] = int(total_runs_done)
        fh.attrs["runs_per_s"] = float(total_runs_done / total_worker_time_s) if total_worker_time_s > 0 else 0.0
        fh.flush()

    print(f"Wrote merged aggregate to {output}")


if __name__ == "__main__":
    main()
