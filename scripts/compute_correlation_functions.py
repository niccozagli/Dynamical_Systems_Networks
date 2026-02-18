#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal


def _read_stats(stats_path: Path):
    with h5py.File(stats_path, "r") as fh:
        dset = fh["stats"]
        field_attr = dset.attrs.get("fieldnames")
        if field_attr is None:
            raise ValueError(f"Missing fieldnames attribute in {stats_path}")
        fieldnames = [s.decode("utf-8") for s in np.asarray(field_attr).tolist()]
        data = np.asarray(dset[...], dtype=float)
    return data, fieldnames


def _load_time_series(graph_dir: Path, transient: float, field: str) -> tuple[np.ndarray, np.ndarray]:
    stats_path = graph_dir / "stats.h5"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats.h5 in {graph_dir}")
    data, fieldnames = _read_stats(stats_path)
    try:
        t_idx = fieldnames.index("t")
        x_idx = fieldnames.index(field)
    except ValueError as exc:
        raise ValueError(f"Required fields not found in {stats_path}: {exc}")
    t = data[:, t_idx]
    x = data[:, x_idx]
    mask = t > float(transient)
    if not np.any(mask):
        raise ValueError(f"No samples with t > transient in {stats_path}")
    return t[mask], x[mask]


def _annealed_series(n_dir: Path, transient: float, field: str) -> tuple[np.ndarray, np.ndarray] | None:
    graph_dirs = sorted(n_dir.glob("graph_*"))
    if not graph_dirs:
        return None
    series = []
    t_ref = None
    for graph_dir in graph_dirs:
        try:
            t, x = _load_time_series(graph_dir, transient, field)
        except Exception as exc:
            print(f"Skip {graph_dir}: {exc}")
            continue
        if t_ref is None:
            t_ref = t
        elif len(t) != len(t_ref) or np.max(np.abs(t - t_ref)) > 1e-8:
            print(f"Skip {graph_dir}: time grid mismatch")
            continue
        series.append(x)
    if not series or t_ref is None:
        return None
    mean_series = np.mean(np.stack(series, axis=0), axis=0)
    return t_ref, mean_series


def _representative_series(n_dir: Path, transient: float, field: str) -> tuple[np.ndarray, np.ndarray] | None:
    graph_dir = next(iter(sorted(n_dir.glob("graph_*"))), None)
    if graph_dir is None:
        return None
    try:
        return _load_time_series(graph_dir, transient, field)
    except Exception as exc:
        print(f"Skip {graph_dir}: {exc}")
        return None


def _normalized_autocorr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    corr = signal.correlate(x, x, mode="full", method="auto")
    mid = x.size - 1
    corr = corr[mid:]
    if corr[0] == 0:
        raise ValueError("C(0) is zero; cannot normalize correlation function.")
    return corr / corr[0]


def _corr_from_series(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Non-positive or invalid dt in time series.")
    corr = _normalized_autocorr(x)
    tau = np.arange(corr.size, dtype=float) * dt
    return tau, corr


def _quenched_correlation(
    n_dir: Path, transient: float, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    graph_dirs = sorted(n_dir.glob("graph_*"))
    if not graph_dirs:
        return None
    corrs = []
    tau_ref = None
    for graph_dir in graph_dirs:
        try:
            t, x = _load_time_series(graph_dir, transient, field)
            tau, corr = _corr_from_series(t, x)
        except Exception as exc:
            print(f"Skip {graph_dir}: {exc}")
            continue
        if tau_ref is None:
            tau_ref = tau
        elif len(tau) != len(tau_ref) or np.max(np.abs(tau - tau_ref)) > 1e-8:
            print(f"Skip {graph_dir}: time grid mismatch")
            continue
        corrs.append(corr)
    if not corrs or tau_ref is None:
        return None
    corr_stack = np.stack(corrs, axis=0)
    mean_corr = np.mean(corr_stack, axis=0)
    std_corr = np.std(corr_stack, axis=0)
    return tau_ref, mean_corr, std_corr


def _annealed_correlation(
    n_dir: Path, transient: float, field: str
) -> tuple[np.ndarray, np.ndarray] | None:
    result = _annealed_series(n_dir, transient, field)
    if result is None:
        return None
    t, mean_series = result
    return _corr_from_series(t, mean_series)


def _color_for(setting: str, idx: int, total: int):
    cmap = plt.get_cmap("Reds" if setting == "critical" else "Blues")
    if total <= 1:
        shade = 0.7
    else:
        shade = 0.3 + 0.7 * (idx / (total - 1))
    return cmap(shade)


def main():
    parser = argparse.ArgumentParser(description="Plot annealed and quenched order parameters.")
    parser.add_argument("--base-dir", default="results/linear_response/poisson/unperturbed_runs",
                        help="Base directory for unperturbed runs.")
    parser.add_argument("--settings", nargs="+", default=["critical", "far"],
                        help="Settings to process (e.g. critical far).")
    parser.add_argument("--Ns", nargs="+", type=int, default=[1000, 5000, 10000],
                        help="Network sizes to process.")
    parser.add_argument("--transient", type=float, default=5000.0,
                        help="Transient time to discard.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    summary_dir = base_dir / "correlation_functions_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        ("mean_x1", "global mean_x1"),
        ("deg_weighted_mean_x1", "degree-weighted mean_x1"),
    ]

    # Annealed order parameter: average across graphs at each time
    fig, axes = plt.subplots(
        nrows=len(args.settings),
        ncols=len(fields),
        figsize=(10, 3.5 * len(args.settings)),
        sharex=True,
    )
    if len(args.settings) == 1:
        axes = [axes]
    for row, setting in enumerate(args.settings):
        setting_dir = base_dir / setting
        n_vals = [n for n in args.Ns if (setting_dir / f"n{n}").exists()]
        for col, (field, label) in enumerate(fields):
            ax = axes[row][col] if len(fields) > 1 else axes[row]
            for n_val in n_vals:
                n_dir = setting_dir / f"n{n_val}"
                result = _annealed_series(n_dir, args.transient, field)
                if result is None:
                    continue
                t, mean_series = result
                ax.plot(t, mean_series, label=f"N={n_val}")
            ax.set_title(f"{setting} (annealed, {label})")
            ax.set_ylabel(label)
            ax.legend(frameon=False)
    for col in range(len(fields)):
        axes[-1][col].set_xlabel("t")
    fig.tight_layout()
    plot_path = summary_dir / "annealed_order_parameter.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot {plot_path}")

    # Quenched representative: one graph per N
    fig, axes = plt.subplots(
        nrows=len(args.settings),
        ncols=len(fields),
        figsize=(10, 3.5 * len(args.settings)),
        sharex=True,
    )
    if len(args.settings) == 1:
        axes = [axes]
    for row, setting in enumerate(args.settings):
        setting_dir = base_dir / setting
        n_vals = [n for n in args.Ns if (setting_dir / f"n{n}").exists()]
        for col, (field, label) in enumerate(fields):
            ax = axes[row][col] if len(fields) > 1 else axes[row]
            for n_val in n_vals:
                n_dir = setting_dir / f"n{n_val}"
                result = _representative_series(n_dir, args.transient, field)
                if result is None:
                    continue
                t, series = result
                ax.plot(t, series, label=f"N={n_val}")
            ax.set_title(f"{setting} (quenched representative, {label})")
            ax.set_ylabel(label)
            ax.legend(frameon=False)
    for col in range(len(fields)):
        axes[-1][col].set_xlabel("t")
    fig.tight_layout()
    plot_path = summary_dir / "quenched_order_parameter_representative.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot {plot_path}")

    # Overlay annealed vs quenched representative in critical setting (window 20000-25000)
    critical_dir = base_dir / "critical"
    if critical_dir.exists():
        fig, axes = plt.subplots(
            nrows=2,
            ncols=len(fields),
            figsize=(10, 6),
            sharex=True,
        )
        if len(fields) == 1:
            axes = [axes]
        for col, (field, label) in enumerate(fields):
            ax = axes[0][col] if len(fields) > 1 else axes[0]
            ax2 = axes[1][col] if len(fields) > 1 else axes[1]
            n_vals = [n for n in args.Ns if n == 10000 and (critical_dir / f"n{n}").exists()]
            for n_val in n_vals:
                n_dir = critical_dir / f"n{n_val}"
                ann = _annealed_series(n_dir, args.transient, field)
                rep = _representative_series(n_dir, args.transient, field)
                if ann is None or rep is None:
                    continue
                t_a, x_a = ann
                t_r, x_r = rep
                # apply window
                mask_a = (t_a >= 20000) & (t_a <= 25000)
                mask_r = (t_r >= 20000) & (t_r <= 25000)
                if np.any(mask_a):
                    ax.plot(t_a[mask_a], x_a[mask_a], label=f"N={n_val} annealed")
                if np.any(mask_r):
                    ax2.plot(t_r[mask_r], x_r[mask_r], label=f"N={n_val} quenched")
            ax.set_title(f"critical (annealed, {label})")
            ax2.set_title(f"critical (quenched rep, {label})")
            ax.set_ylabel(label)
            ax2.set_ylabel(label)
            ax.legend(frameon=False)
            ax2.legend(frameon=False)
        for col in range(len(fields)):
            axes[1][col].set_xlabel("t")
        fig.tight_layout()
        plot_path = summary_dir / "critical_annealed_vs_quenched_window.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot {plot_path}")

    # Correlation functions for degree-weighted order parameter
    field = "deg_weighted_mean_x1"

    # Quenched: per-graph correlations, then mean/std
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for setting in args.settings:
        setting_dir = base_dir / setting
        n_vals = [n for n in args.Ns if (setting_dir / f"n{n}").exists()]
        n_vals = sorted(n_vals)
        for idx, n_val in enumerate(n_vals):
            n_dir = setting_dir / f"n{n_val}"
            result = _quenched_correlation(n_dir, args.transient, field)
            if result is None:
                continue
            tau, mean_corr, std_corr = result
            color = _color_for(setting, idx, len(n_vals))
            ax.plot(tau, mean_corr, color=color, label=f"{setting} N={n_val}")
            ax.fill_between(tau, mean_corr - std_corr, mean_corr + std_corr, color=color, alpha=0.2)
    ax.set_title("Correlation functions (quenched, degree-weighted order)")
    ax.set_xlabel("t")
    ax.set_ylabel("C(t) / C(0)")
    ax.legend(frameon=False)
    fig.tight_layout()
    plot_path = summary_dir / "correlation_functions_quenched.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot {plot_path}")

    # Annealed: mean order parameter across graphs first, then correlation
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for setting in args.settings:
        setting_dir = base_dir / setting
        n_vals = [n for n in args.Ns if (setting_dir / f"n{n}").exists()]
        n_vals = sorted(n_vals)
        for idx, n_val in enumerate(n_vals):
            n_dir = setting_dir / f"n{n_val}"
            result = _annealed_correlation(n_dir, args.transient, field)
            if result is None:
                continue
            tau, corr = result
            color = _color_for(setting, idx, len(n_vals))
            ax.plot(tau, corr, color=color, label=f"{setting} N={n_val}")
    ax.set_title("Correlation functions (annealed, degree-weighted order)")
    ax.set_xlabel("t")
    ax.set_ylabel("C(t) / C(0)")
    ax.legend(frameon=False)
    fig.tight_layout()
    plot_path = summary_dir / "correlation_functions_annealed.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot {plot_path}")


if __name__ == "__main__":
    main()
