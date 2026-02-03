#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

def _read_rows(path: Path) -> list[dict[str, str]]:
    delim = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", newline="") as fh:
        lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
        reader = csv.DictReader(lines, delimiter=delim)
        return list(reader)


def _parse_value(raw: str):
    s = raw.strip()
    if s == "":
        return None
    low = s.lower()
    if low in {"true", "false", "null"}:
        return json.loads(low)
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    try:
        if "." not in s and "e" not in s and "E" not in s:
            return int(s)
        return float(s)
    except ValueError:
        return s


def load_table_row(path: Path, row_index: int) -> dict[str, str]:
    if row_index <= 0:
        raise ValueError("row_index must be >= 1")
    delim = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", newline="") as fh:
        lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
        reader = csv.DictReader(lines, delimiter=delim)
        for idx, row in enumerate(reader, start=1):
            if idx == row_index:
                return row
    raise ValueError(f"Row {row_index} not found in {path}")


def apply_overrides(config: dict, row: dict[str, str]) -> None:
    for key, raw in row.items():
        if raw is None:
            continue
        val = _parse_value(raw)
        if val is None:
            continue
        parts = key.split(".")
        if len(parts) == 1:
            config[parts[0]] = val
            continue
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = val


def cmd_count_rows(args: argparse.Namespace) -> None:
    rows = _read_rows(Path(args.table))
    print(len(rows))


def cmd_row_id(args: argparse.Namespace) -> None:
    print(f"row_{args.row_index:04d}")


def cmd_write_config_used(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.config).read_text())
    row = load_table_row(Path(args.table), int(args.row_index))
    apply_overrides(config, row)

    config.setdefault("phase_diagram", {})
    config["phase_diagram"]["graph_realizations"] = int(args.graph_realizations)
    config["phase_diagram"]["noise_realizations"] = int(args.noise_realizations)

    run_dir = Path(args.output_dir) / str(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.json").write_text(json.dumps(config, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    count_parser = subparsers.add_parser("count-rows", help="Count rows in the sweep table.")
    count_parser.add_argument("--table", required=True, help="CSV/TSV params table path.")
    count_parser.set_defaults(func=cmd_count_rows)

    row_parser = subparsers.add_parser("row-id", help="Format a row-based run id.")
    row_parser.add_argument("--row-index", type=int, required=True, help="1-based row index.")
    row_parser.set_defaults(func=cmd_row_id)

    write_parser = subparsers.add_parser(
        "write-config-used",
        help="Write config_used.json for a sweep row with overrides applied.",
    )
    write_parser.add_argument("--config", required=True, help="Base JSON config path.")
    write_parser.add_argument("--table", required=True, help="CSV/TSV params table path.")
    write_parser.add_argument("--row-index", type=int, required=True, help="1-based row index.")
    write_parser.add_argument("--output-dir", required=True, help="Output directory.")
    write_parser.add_argument("--run-id", required=True, help="Run identifier (folder name).")
    write_parser.add_argument("--graph-realizations", type=int, required=True)
    write_parser.add_argument("--noise-realizations", type=int, required=True)
    write_parser.set_defaults(func=cmd_write_config_used)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
