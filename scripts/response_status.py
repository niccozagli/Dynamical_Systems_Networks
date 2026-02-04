#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py


def _fmt(val, width=10):
    if val is None:
        return " " * width
    if isinstance(val, float):
        return f"{val:>{width}.2f}"
    return f"{val:>{width}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show progress from response worker files.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory (contains response/worker_*.h5).",
    )
    parser.add_argument(
        "--pattern",
        default="worker_*.h5",
        help="Worker file pattern inside response/ (default: worker_*.h5).",
    )
    parser.add_argument(
        "--per-worker",
        action="store_true",
        help="Print per-worker rows (default: summary only).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / "response"
    paths = sorted(out_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No worker files found in {out_dir} matching {args.pattern}")

    total_runs = 0
    total_samples = 0
    if args.per_worker:
        header = (
            f"{'worker':>8} {'job':>6} {'wid':>6} {'runs':>10} {'samples':>10} "
            f"{'r/s':>10} {'time_s':>10}"
        )
        print(header)
        print("-" * len(header))
    for path in paths:
        with h5py.File(path, "r", swmr=True, libver="latest") as fh:
            runs_done = fh.attrs.get("runs_done")
            sample_count = fh.attrs.get("sample_count")
            runs_per_s = fh.attrs.get("runs_per_s")
            worker_time_s = fh.attrs.get("worker_time_s")
            job_id = fh.attrs.get("job_id")
            worker_id = fh.attrs.get("worker_id")
            global_worker_id = fh.attrs.get("global_worker_id")

        total_runs += int(runs_done or 0)
        total_samples += int(sample_count or 0)

        if args.per_worker:
            print(
                f"{_fmt(global_worker_id, 8)} {_fmt(job_id, 6)} {_fmt(worker_id, 6)} "
                f"{_fmt(runs_done, 10)} {_fmt(sample_count, 10)} {_fmt(runs_per_s, 10)} {_fmt(worker_time_s, 10)}"
            )

    if args.per_worker:
        print("-" * len(header))
    print(f"{'total':>8} {'':>6} {'':>6} {_fmt(total_runs, 10)} {_fmt(total_samples, 10)}")


if __name__ == "__main__":
    main()
