#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import correlate


def _read_stats(stats_path: Path):
    with h5py.File(stats_path, "r") as fh:
        dset = fh["stats"]
        field_attr = dset.attrs.get("fieldnames")
        if field_attr is None:
            raise ValueError(f"Missing fieldnames attribute in {stats_path}")
        fieldnames = [s.decode("utf-8") for s in np.asarray(field_attr).tolist()]
        data = np.asarray(dset[...], dtype=float)
    return data, fieldnames


def _compute_corr(stats_path: Path, transient: float):
    data, fieldnames = _read_stats(stats_path)
    try:
        t_idx = fieldnames.index("t")
        x_idx = fieldnames.index("mean_x1")
    except ValueError as exc:
        raise ValueError(f"Required fields not found in {stats_path}: {exc}")

    t = data[:, t_idx]
    signal = data[:, x_idx]
    mask = t > float(transient)
    if not np.any(mask):
        raise ValueError(f"No samples with t > transient in {stats_path}")

    t = t[mask]
    signal = signal[mask]
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0

    signal = signal - np.mean(signal)
    corr = correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2 :] / signal.size
    return corr, dt


def _save_corr(output_path: Path, t: np.ndarray, corr_mean: np.ndarray, corr_std: np.ndarray, *,
               graph_count: int, transient: float, t_max: float | None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as fh:
        fh.create_dataset("t", data=t, compression="gzip", compression_opts=4)
        fh.create_dataset("corr_mean", data=corr_mean, compression="gzip", compression_opts=4)
        fh.create_dataset("corr_std", data=corr_std, compression="gzip", compression_opts=4)
        fh.attrs["graph_count"] = int(graph_count)
        fh.attrs["transient"] = float(transient)
        fh.attrs["stat"] = "mean_x1"
        if t_max is not None:
            fh.attrs["t_max"] = float(t_max)


def main():
    parser = argparse.ArgumentParser(description="Compute correlation functions for unperturbed runs.")
    parser.add_argument("--base-dir", default="results/linear_response/poisson/unperturbed_runs",
                        help="Base directory for unperturbed runs.")
    parser.add_argument("--settings", nargs="+", default=["critical", "far"],
                        help="Settings to process (e.g. critical far).")
    parser.add_argument("--Ns", nargs="+", type=int, default=[1000, 5000, 10000],
                        help="Network sizes to process.")
    parser.add_argument("--transient", type=float, default=5000.0,
                        help="Transient time to discard.")
    parser.add_argument("--output-name", default="correlation_mean_x1.h5",
                        help="Output filename to store averaged correlation.")
    parser.add_argument("--plot-name", default="correlation_mean_x1.png",
                        help="Plot filename (saved under base-dir).")
    parser.add_argument("--t-max", type=float, default=2000.0,
                        help="Max time shown/saved for the correlation function.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    results = {setting: {} for setting in args.settings}

    for setting in args.settings:
        for n_val in args.Ns:
            n_dir = base_dir / setting / f"n{n_val}"
            if not n_dir.exists():
                continue

            corrs = []
            dt_ref = None
            graph_count = 0

            for graph_dir in sorted(n_dir.glob("graph_*")):
                stats_path = graph_dir / "stats.h5"
                if not stats_path.exists():
                    continue
                try:
                    corr, dt = _compute_corr(stats_path, args.transient)
                except Exception as exc:
                    print(f"Skip {stats_path}: {exc}")
                    continue

                if dt_ref is None:
                    dt_ref = dt
                elif abs(dt - dt_ref) > 1e-10:
                    print(f"Skip {stats_path}: dt mismatch ({dt} vs {dt_ref})")
                    continue

                corrs.append(corr)
                graph_count += 1

            if not corrs:
                continue

            min_len = min(len(c) for c in corrs)
            corr_stack = np.stack([c[:min_len] for c in corrs], axis=0)
            corr_mean = corr_stack.mean(axis=0)
            if corr_stack.shape[0] > 1:
                corr_std = corr_stack.std(axis=0, ddof=1)
            else:
                corr_std = np.zeros_like(corr_mean)

            dt_ref = dt_ref or 0.0
            t = np.arange(min_len) * dt_ref
            if args.t_max is not None:
                mask = t <= float(args.t_max)
                if not np.any(mask):
                    print(f"Skip {n_dir}: no samples within t_max={args.t_max}")
                    continue
                t = t[mask]
                corr_mean = corr_mean[mask]
                corr_std = corr_std[mask]
            output_path = n_dir / args.output_name
            _save_corr(output_path, t, corr_mean, corr_std,
                       graph_count=graph_count, transient=args.transient, t_max=args.t_max)
            print(f"Saved {output_path} (graphs={graph_count})")
            results[setting][n_val] = (t, corr_mean, corr_std)

    if any(results.values()):
        fig, axes = plt.subplots(nrows=len(args.settings), figsize=(8, 4 * len(args.settings)), sharex=True)
        if len(args.settings) == 1:
            axes = [axes]

        for ax, setting in zip(axes, args.settings):
            n_vals = sorted(results.get(setting, {}).keys())
            if not n_vals:
                continue
            cmap = plt.get_cmap("viridis")
            for i, n_val in enumerate(n_vals):
                t, corr_mean, corr_std = results[setting][n_val]
                frac = 0.2 + 0.6 * (i / max(1, len(n_vals) - 1))
                color = cmap(frac)
                ax.plot(t, corr_mean, color=color, label=f"N={n_val}")
                ax.fill_between(t, corr_mean - corr_std, corr_mean + corr_std, color=color, alpha=0.2)

            ax.set_title(setting)
            ax.set_ylabel("corr(mean_x1)")
            ax.legend(frameon=False)

        axes[-1].set_xlabel("t")
        fig.tight_layout()
        plot_path = base_dir / args.plot_name
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot {plot_path}")


if __name__ == "__main__":
    main()
