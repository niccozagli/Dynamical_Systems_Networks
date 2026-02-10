#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from dyn_net.utils.criticality import (
    find_theta_c,
    find_theta_c_from_degree_distribution,
)
from dyn_net.utils.simulation_steps import prepare_network


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def _get_sigma(cfg: dict) -> float:
    noise = cfg.get("noise", {}).get("params", {})
    for key in ("sigma", "std", "noise_std"):
        if key in noise:
            return float(noise[key])
    raise ValueError("Could not find sigma in noise params")


def _theta_c_empirical(cfg: dict) -> float:
    A = prepare_network(cfg)
    deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(int)
    ks, counts = np.unique(deg, return_counts=True)
    P_k = counts / counts.sum()
    pi_k = (ks * P_k) / (ks * P_k).sum()
    sigma = _get_sigma(cfg)
    return float(
        find_theta_c(
            ks,
            pi_k,
            sigma,
            theta_bracket=(0.01, 5.0),
        )
    )


def _theta_c_meanfield(cfg: dict) -> float:
    deg_dist = cfg["network"]["params"]["degree_distribution"]
    sigma = _get_sigma(cfg)
    return float(
        find_theta_c_from_degree_distribution(
            deg_dist,
            sigma,
            theta_bracket=(0.01, 5.0),
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Check theta vs criticality per graph.")
    parser.add_argument(
        "--base-dir",
        default="results/linear_response/poisson/unperturbed_runs",
        help="Base directory containing setting/n*/graph_*/config_used.json",
    )
    parser.add_argument("--settings", nargs="+", default=["critical", "far"])
    parser.add_argument("--Ns", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument(
        "--limit-graphs",
        type=int,
        default=1,
        help="Max graphs per N to check (default 1).",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV output path.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    header = "setting,N,graph,theta,theta_c_meanfield,theta_c_empirical"
    print(header)
    rows = [header]
    for setting in args.settings:
        for n_val in args.Ns:
            n_dir = base_dir / setting / f"n{n_val}"
            if not n_dir.exists():
                continue
            graphs = sorted(n_dir.glob("graph_*"))
            if args.limit_graphs > 0:
                graphs = graphs[: args.limit_graphs]
            for graph_dir in graphs:
                cfg_path = graph_dir / "config_used.json"
                if not cfg_path.exists():
                    continue
                try:
                    cfg = _load_config(cfg_path)
                    theta = float(cfg["system"]["params"]["theta"])
                    theta_c_mf = _theta_c_meanfield(cfg)
                    theta_c_emp = _theta_c_empirical(cfg)
                except Exception as exc:
                    row = f"{setting},{n_val},{graph_dir.name},ERROR,{exc}"
                    print(row)
                    rows.append(row)
                    continue
                row = (
                    f"{setting},{n_val},{graph_dir.name},"
                    f"{theta:.6f},{theta_c_mf:.6f},{theta_c_emp:.6f}"
                )
                print(row)
                rows.append(row)

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
