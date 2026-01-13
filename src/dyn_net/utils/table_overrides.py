import csv
import json
from pathlib import Path
from typing import Any


def _parse_value(raw: str) -> Any:
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
