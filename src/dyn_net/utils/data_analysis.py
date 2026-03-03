from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResponseRecord:
    linear: pd.DataFrame
    quadratic: pd.DataFrame
    config: dict


def find_repo_root(start: Path | None = None, marker: str = "pyproject.toml") -> Path:
    start_path = (start or Path.cwd()).resolve()
    for parent in (start_path, *start_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find repo root from {start_path} (missing {marker})."
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _decode_fieldnames(field_attr: object) -> list[str]:
    if field_attr is None:
        raise KeyError("Missing 'fieldnames' attribute in aggregate response file.")
    arr = np.asarray(field_attr)
    values = arr.tolist()
    if isinstance(values, list):
        decoded = []
        for item in values:
            if isinstance(item, (bytes, bytearray)):
                decoded.append(item.decode("utf-8"))
            else:
                decoded.append(str(item))
        return decoded
    if isinstance(values, (bytes, bytearray)):
        return [values.decode("utf-8")]
    return [str(values)]


def load_aggregate_response(
    aggregate_path: Path,
    *,
    swmr: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing aggregate response file: {aggregate_path}")

    with h5py.File(aggregate_path, "r", swmr=swmr) as h5f:
        fieldnames = _decode_fieldnames(h5f.attrs.get("fieldnames"))
        df_plus = pd.DataFrame(h5f["mean_plus"][...], columns=fieldnames)
        df_minus = pd.DataFrame(h5f["mean_minus"][...], columns=fieldnames)

    return df_plus, df_minus


def compute_response_frames(
    df_plus: pd.DataFrame,
    df_minus: pd.DataFrame,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if epsilon == 0:
        raise ValueError("epsilon must be non-zero for response computation.")

    linear = (df_plus - df_minus) / (2 * epsilon)
    linear["t"] = df_plus["t"]

    quadratic = (df_plus + df_minus) / (2 * epsilon**2)
    quadratic["t"] = df_plus["t"]

    return linear, quadratic


def load_response_records(
    *,
    repo_root: Path,
    network_label: str,
    perturbation_type: str,
    settings: dict[str, Iterable[int]],
    eps_tag: str,
) -> dict[str, list[ResponseRecord]]:
    records: dict[str, list[ResponseRecord]] = {}

    for setting, ns in settings.items():
        setting_records: list[ResponseRecord] = []
        for n_nodes in ns:
            sim_output_dir = (
                repo_root
                / "results/linear_response"
                / network_label
                / "perturbed_runs"
                / perturbation_type
                / setting
                / f"n{n_nodes}"
            )

            aggregate_path = sim_output_dir / f"aggregate_response_eps_{eps_tag}.h5"
            config_path = sim_output_dir / f"config_used_eps_{eps_tag}.json"

            config = read_json(config_path)
            df_plus, df_minus = load_aggregate_response(aggregate_path)
            epsilon = config["perturbation"]["epsilon"]
            linear, quadratic = compute_response_frames(df_plus, df_minus, epsilon)

            setting_records.append(
                ResponseRecord(linear=linear, quadratic=quadratic, config=config)
            )

        records[setting] = setting_records

    return records
