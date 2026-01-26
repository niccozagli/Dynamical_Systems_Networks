#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import h5py
from typing import cast
import json


def read_network_seed(run_dir: Path) -> int | None:
    cfg_path = run_dir / "config_used.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    return cfg.get("network", {}).get("params", {}).get("seed")


def iter_run_dirs(root: Path):
    for path in sorted(root.iterdir()):
        if path.is_dir():
            state_path = path / "state.h5"
            if state_path.exists():
                yield path


def build_table(
    unperturbed_root: Path,
    output_path: Path,
    transient: float,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "state_path",
        "time_index",
        "t",
        "network.params.seed",
    ]
    count = 0
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for run_dir in iter_run_dirs(unperturbed_root):
            run_id = run_dir.name
            seed = read_network_seed(run_dir)
            state_path = run_dir / "state.h5"

            with h5py.File(state_path, "r") as h5f:
                times_dset = cast(h5py.Dataset, h5f["time"])
                times = times_dset[...]
                for idx, t in enumerate(times):
                    t_val = float(t)
                    if t_val < transient:
                        continue
                    row = {
                        "run_id": run_id,
                        "state_path": str(state_path),
                        "time_index": int(idx),
                        "t": t_val,
                        "network.params.seed": seed if seed is not None else "",
                    }
                    writer.writerow(row)
                    count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a response table from unperturbed state files (post-transient only)."
    )
    parser.add_argument(
        "--unperturbed-root",
        required=True,
        help="Root directory containing per-run folders with state.h5",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Output TSV path (overrides --output-dir).",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Output directory for response_samples.tsv.",
    )
    parser.add_argument(
        "--transient",
        type=float,
        default=0.0,
        help="Transient cutoff (keep t >= transient)",
    )

    args = parser.parse_args()
    root = Path(args.unperturbed_root)
    if args.output is None and args.output_dir is None:
        raise ValueError("Provide --output or --output-dir.")
    if args.output is not None and args.output_dir is not None:
        raise ValueError("Provide only one of --output or --output-dir.")
    if args.output is not None:
        output = Path(args.output)
    else:
        output = Path(args.output_dir) / "response_samples.tsv"

    count = build_table(root, output, float(args.transient))
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()
