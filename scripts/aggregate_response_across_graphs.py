#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np

from dyn_net.utils.aggregate import AggregateState, merge_aggregate


def _load_graph_aggregate(path: Path):
    with h5py.File(path, "r") as fh:
        field_attr = fh.attrs.get("fieldnames")
        if field_attr is None:
            raise ValueError(f"Missing fieldnames in {path}")
        fieldnames = [s.decode("utf-8") for s in np.asarray(field_attr).tolist()]

        mean_plus = np.asarray(fh["mean_plus"][...])
        std_plus = np.asarray(fh["std_plus"][...])
        count_plus = int(fh["count_plus"][()])

        mean_minus = np.asarray(fh["mean_minus"][...])
        std_minus = np.asarray(fh["std_minus"][...])
        count_minus = int(fh["count_minus"][()])

        attrs = dict(fh.attrs)

    # Reconstruct m2 from std (sample std: std = sqrt(m2/(count-1)))
    if count_plus > 1:
        m2_plus = (std_plus ** 2) * (count_plus - 1)
    else:
        m2_plus = np.zeros_like(mean_plus)

    if count_minus > 1:
        m2_minus = (std_minus ** 2) * (count_minus - 1)
    else:
        m2_minus = np.zeros_like(mean_minus)

    return (
        fieldnames,
        mean_plus,
        m2_plus,
        count_plus,
        mean_minus,
        m2_minus,
        count_minus,
        attrs,
    )


def main():
    parser = argparse.ArgumentParser(description="Aggregate response results across graphs.")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory containing graph_* subfolders.",
    )
    parser.add_argument(
        "--eps-tag",
        required=True,
        help="Epsilon tag (e.g. 001).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for merged aggregate file (optional).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    eps_tag = args.eps_tag
    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")

    graph_dirs = sorted(base_dir.glob("graph_*") )
    if not graph_dirs:
        raise SystemExit(f"No graph_* folders found under {base_dir}")

    agg_paths = []
    config_paths = []
    for graph_dir in graph_dirs:
        agg_path = graph_dir / f"eps{eps_tag}" / "response" / "aggregate.h5"
        if agg_path.exists():
            agg_paths.append(agg_path)
        cfg_path = graph_dir / f"eps{eps_tag}" / "response" / "config_used.json"
        if cfg_path.exists():
            config_paths.append(cfg_path)
    if not agg_paths:
        raise SystemExit(f"No aggregate.h5 found for eps{eps_tag} under {base_dir}")

    state_plus = AggregateState()
    state_minus = AggregateState()
    fieldnames_ref = None
    sample_count = 0
    graph_count = 0
    attrs_ref = None

    for path in agg_paths:
        (
            fieldnames,
            mean_plus,
            m2_plus,
            count_plus,
            mean_minus,
            m2_minus,
            count_minus,
            attrs,
        ) = _load_graph_aggregate(path)

        if fieldnames_ref is None:
            fieldnames_ref = fieldnames
            attrs_ref = attrs
        elif fieldnames != fieldnames_ref:
            raise ValueError(f"Fieldnames mismatch in {path}")

        state_plus = merge_aggregate(state_plus, mean_plus, m2_plus, count_plus)
        state_minus = merge_aggregate(state_minus, mean_minus, m2_minus, count_minus)
        sample_count += int(count_plus)
        graph_count += 1

    if state_plus.mean is None or state_minus.mean is None:
        raise SystemExit("No data to aggregate.")

    std_plus = np.sqrt(state_plus.m2 / max(1, state_plus.count - 1))
    std_minus = np.sqrt(state_minus.m2 / max(1, state_minus.count - 1))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"aggregate_response_eps_{eps_tag}.h5"
    else:
        out_path = base_dir / f"aggregate_response_eps_{eps_tag}.h5"

    with h5py.File(out_path, "w") as fh:
        fh.create_dataset("mean_plus", data=state_plus.mean, compression="gzip", compression_opts=4)
        fh.create_dataset("std_plus", data=std_plus, compression="gzip", compression_opts=4)
        fh.create_dataset("count_plus", data=int(state_plus.count))
        fh.create_dataset("mean_minus", data=state_minus.mean, compression="gzip", compression_opts=4)
        fh.create_dataset("std_minus", data=std_minus, compression="gzip", compression_opts=4)
        fh.create_dataset("count_minus", data=int(state_minus.count))
        fh.attrs["fieldnames"] = np.asarray(fieldnames_ref, dtype="S")
        fh.attrs["graph_count"] = int(graph_count)
        fh.attrs["sample_count"] = int(sample_count)
        # propagate perturbation metadata if present
        if attrs_ref is not None:
            if "perturbation_type" in attrs_ref:
                fh.attrs["perturbation_type"] = attrs_ref["perturbation_type"]
            if "perturbation_epsilon" in attrs_ref:
                fh.attrs["perturbation_epsilon"] = attrs_ref["perturbation_epsilon"]

    if config_paths:
        cfg_out = out_path.with_name(f"config_used_eps_{eps_tag}.json")
        cfg_out.write_text(config_paths[0].read_text())

    print(f"Saved merged aggregate to {out_path} (graphs={graph_count})")


if __name__ == "__main__":
    main()
