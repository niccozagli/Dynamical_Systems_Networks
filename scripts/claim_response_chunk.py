#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np


def _table_delim(path: Path) -> str:
    return "," if path.suffix.lower() == ".csv" else "\t"


def _load_used_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    used: set[str] = set()
    with path.open("r", newline="") as fh:
        for line in fh:
            s = line.strip()
            if s:
                used.add(s)
    return used


def _next_chunk_index(output_dir: Path, prefix: str) -> int:
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.tsv$")
    max_idx = 0
    for path in output_dir.glob(f"{prefix}_*.tsv"):
        m = pat.match(path.name)
        if not m:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def claim_chunk(
    table_path: Path,
    output_dir: Path,
    used_ids_path: Path,
    chunk_size: int,
    chunk_prefix: str,
    chunk_index: int | None,
    *,
    randomize: bool,
    seed: int | None,
) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    used_ids_path.parent.mkdir(parents=True, exist_ok=True)

    used_ids = _load_used_ids(used_ids_path)
    delim = _table_delim(table_path)

    if chunk_index is None:
        chunk_index = _next_chunk_index(output_dir, chunk_prefix)

    chunk_path = output_dir / f"{chunk_prefix}_{chunk_index:04d}.tsv"

    claimed_ids: list[str] = []
    written = 0
    rng = np.random.default_rng(seed)

    with table_path.open("r", newline="") as in_fh, chunk_path.open("w", newline="") as out_fh:
        lines = (line for line in in_fh if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(lines, delimiter=delim)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Response table has no header.")
        if "sample_id" not in fieldnames:
            raise ValueError("Response table is missing required column 'sample_id'.")

        writer = csv.DictWriter(out_fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        if not randomize:
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                if not sample_id or sample_id in used_ids:
                    continue
                writer.writerow(row)
                claimed_ids.append(sample_id)
                written += 1
                if written >= chunk_size:
                    break
        else:
            # Reservoir sampling over unused rows to avoid loading the full table.
            reservoir: list[dict[str, str]] = []
            seen_unused = 0
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                if not sample_id or sample_id in used_ids:
                    continue
                seen_unused += 1
                if len(reservoir) < chunk_size:
                    reservoir.append(row)
                    continue
                j = int(rng.integers(0, seen_unused))
                if j < chunk_size:
                    reservoir[j] = row

            for row in reservoir:
                sample_id = (row.get("sample_id") or "").strip()
                if not sample_id:
                    continue
                writer.writerow(row)
                claimed_ids.append(sample_id)
                written += 1

    if written == 0:
        # Leave an empty file only if nothing could be claimed.
        chunk_path.unlink(missing_ok=True)
        return chunk_path, 0

    with used_ids_path.open("a", newline="") as fh:
        for sample_id in claimed_ids:
            fh.write(sample_id + "\n")

    return chunk_path, written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claim the next unused chunk of response samples based on sample_id."
    )
    parser.add_argument("--table", required=True, help="Path to response_samples.tsv")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where chunk TSVs will be written",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Number of unused rows to claim",
    )
    parser.add_argument(
        "--used-ids",
        required=False,
        help="Path to used_sample_ids.txt (default: <output-dir>/used_sample_ids.txt)",
    )
    parser.add_argument(
        "--chunk-prefix",
        default="response_samples_chunk",
        help="Chunk filename prefix (default: response_samples_chunk)",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Optional explicit chunk index (otherwise auto-increment)",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomly sample unused rows via reservoir sampling (reads full table).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (only used with --randomize).",
    )

    args = parser.parse_args()

    table_path = Path(args.table)
    output_dir = Path(args.output_dir)
    used_ids_path = Path(args.used_ids) if args.used_ids else output_dir / "used_sample_ids.txt"

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be >= 1")

    chunk_path, written = claim_chunk(
        table_path=table_path,
        output_dir=output_dir,
        used_ids_path=used_ids_path,
        chunk_size=int(args.chunk_size),
        chunk_prefix=str(args.chunk_prefix),
        chunk_index=args.chunk_index,
        randomize=bool(args.randomize),
        seed=args.seed,
    )

    if written == 0:
        print("No unused samples available to claim.")
        return

    print(f"Claimed {written} samples -> {chunk_path}")
    print(f"Used IDs file: {used_ids_path}")


if __name__ == "__main__":
    main()
