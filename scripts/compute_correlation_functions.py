#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import correlate

from dyn_net.utils.simulation_steps import prepare_network


def _read_stats(stats_path: Path):
    with h5py.File(stats_path, "r") as fh:
        dset = fh["stats"]
        field_attr = dset.attrs.get("fieldnames")
        if field_attr is None:
            raise ValueError(f"Missing fieldnames attribute in {stats_path}")
        fieldnames = [s.decode("utf-8") for s in np.asarray(field_attr).tolist()]
        data = np.asarray(dset[...], dtype=float)
    return data, fieldnames


def _compute_corr_from_stats(stats_path: Path, transient: float):
    data, fieldnames = _read_stats(stats_path)
    try:
        t_idx = fieldnames.index("t")
        x_idx = fieldnames.index("deg_weighted_mean_x1")
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
               graph_count: int, transient: float, t_max: float | None, stat_name: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as fh:
        fh.create_dataset("t", data=t, compression="gzip", compression_opts=4)
        fh.create_dataset("corr_mean", data=corr_mean, compression="gzip", compression_opts=4)
        fh.create_dataset("corr_std", data=corr_std, compression="gzip", compression_opts=4)
        fh.attrs["graph_count"] = int(graph_count)
        fh.attrs["transient"] = float(transient)
        fh.attrs["stat"] = stat_name
        if t_max is not None:
            fh.attrs["t_max"] = float(t_max)


def _load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def _load_final_x1(state_path: Path) -> np.ndarray:
    with h5py.File(state_path, "r") as h5f:
        state_dset = h5f["state"]
        last_idx = int(state_dset.shape[0] - 1)
        if last_idx < 0:
            raise ValueError(f"Empty state dataset in {state_path}")
        x = np.asarray(state_dset[last_idx, :], dtype=float).reshape(-1)
    return x


def _load_degrees(config_path: Path) -> np.ndarray:
    config = _load_config(config_path)
    A = prepare_network(config)
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    return deg


def _plot_rho_k_by_bins(ax, x1: np.ndarray, deg: np.ndarray, *, bins=60):
    # Bin by degree quantiles (low/mid/high)
    q = np.quantile(deg, [0.0, 0.33, 0.66, 1.0])
    labels = ["low k", "mid k", "high k"]
    for (lo, hi), label in zip(zip(q[:-1], q[1:]), labels):
        mask = (deg >= lo) & (deg <= hi)
        if not np.any(mask):
            continue
        hist, bin_edges = np.histogram(x1[mask], bins=bins, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(centers, hist, label=label)


def _plot_rho_k_heatmap(ax, x1: np.ndarray, deg: np.ndarray, *, x_bins=60, k_bins=40):
    # 2D histogram of k vs x1 (density normalized per-k)
    k_vals = deg.astype(float)
    hist, x_edges, k_edges = np.histogram2d(x1, k_vals, bins=[x_bins, k_bins], density=False)
    # Normalize per-k bin to get rho_k(x)
    col_sums = hist.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        hist = np.divide(hist, col_sums, where=col_sums > 0)
    extent = [k_edges[0], k_edges[-1], x_edges[0], x_edges[-1]]
    im = ax.imshow(
        hist,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("x1")
    return im


def _aggregate_and_save(
    *,
    base_dir: Path,
    settings: list[str],
    n_vals: list[int],
    transient: float,
    t_max: float | None,
    output_name: str,
    stat_name: str,
    compute_fn,
    plot_name: str,
    normalized_plot_name: str,
    tau_plot_name: str,
    ylabel: str,
):
    results = {setting: {} for setting in settings}
    tau_norm_results = {setting: {} for setting in settings}

    for setting in settings:
        for n_val in n_vals:
            n_dir = base_dir / setting / f"n{n_val}"
            if not n_dir.exists():
                continue

            corrs = []
            dt_ref = None
            graph_count = 0

            for graph_dir in sorted(n_dir.glob("graph_*")):
                try:
                    corr, dt = compute_fn(graph_dir, transient)
                except Exception as exc:
                    print(f"Skip {graph_dir}: {exc}")
                    continue

                if dt_ref is None:
                    dt_ref = dt
                elif abs(dt - dt_ref) > 1e-10:
                    print(f"Skip {graph_dir}: dt mismatch ({dt} vs {dt_ref})")
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
            if t_max is not None:
                mask = t <= float(t_max)
                if not np.any(mask):
                    print(f"Skip {n_dir}: no samples within t_max={t_max}")
                    continue
                t = t[mask]
                corr_mean = corr_mean[mask]
                corr_std = corr_std[mask]

            output_path = n_dir / output_name
            _save_corr(
                output_path,
                t,
                corr_mean,
                corr_std,
                graph_count=graph_count,
                transient=transient,
                t_max=t_max,
                stat_name=stat_name,
            )
            print(f"Saved {output_path} (graphs={graph_count})")
            results[setting][n_val] = (t, corr_mean, corr_std)

    summary_dir = base_dir / "correlation_functions_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if any(results.values()):
        fig, axes = plt.subplots(nrows=len(settings), figsize=(8, 4 * len(settings)), sharex=True)
        if len(settings) == 1:
            axes = [axes]

        for ax, setting in zip(axes, settings):
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
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False)

        axes[-1].set_xlabel("t")
        fig.tight_layout()
        plot_path = summary_dir / plot_name
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot {plot_path}")

        fig, axes = plt.subplots(nrows=len(settings), figsize=(8, 4 * len(settings)), sharex=True)
        if len(settings) == 1:
            axes = [axes]

        for ax, setting in zip(axes, settings):
            n_vals = sorted(results.get(setting, {}).keys())
            if not n_vals:
                continue
            cmap = plt.get_cmap("viridis")
            for i, n_val in enumerate(n_vals):
                t, corr_mean, corr_std = results[setting][n_val]
                if corr_mean.size == 0:
                    continue
                norm = corr_mean[0]
                if norm == 0:
                    continue
                corr_mean_n = corr_mean / norm
                corr_std_n = corr_std / abs(norm)
                frac = 0.2 + 0.6 * (i / max(1, len(n_vals) - 1))
                color = cmap(frac)
                ax.plot(t, corr_mean_n, color=color, label=f"N={n_val}")
                ax.fill_between(t, corr_mean_n - corr_std_n, corr_mean_n + corr_std_n, color=color, alpha=0.2)
                tau_norm_results[setting][n_val] = float(np.trapz(corr_mean_n, t))

            ax.set_title(f"{setting} (normalized)")
            ax.set_ylabel("C(t) / C(0)")
            ax.legend(frameon=False)

        axes[-1].set_xlabel("t")
        fig.tight_layout()
        plot_path = summary_dir / normalized_plot_name
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot {plot_path}")

    if any(tau_norm_results.values()):
        fig, axes = plt.subplots(nrows=len(settings), figsize=(6, 3.5 * len(settings)), sharex=True)
        if len(settings) == 1:
            axes = [axes]

        for ax, setting in zip(axes, settings):
            n_vals = sorted(tau_norm_results.get(setting, {}).keys())
            if not n_vals:
                continue
            taus = [tau_norm_results[setting][n] for n in n_vals]
            ax.plot(n_vals, taus, marker="o")
            ax.set_title(setting)
            ax.set_ylabel(r"$\tau_{\mathrm{corr}}$")

        axes[-1].set_xlabel("N")
        fig.tight_layout()
        tau_plot_path = summary_dir / tau_plot_name
        fig.savefig(tau_plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot {tau_plot_path}")


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
    parser.add_argument("--output-name", default="correlation_degree_weighted_mean_x1.h5",
                        help="Output filename to store averaged correlation.")
    parser.add_argument("--plot-name", default="correlation_degree_weighted_mean_x1.png",
                        help="Plot filename (saved under base-dir).")
    parser.add_argument("--t-max", type=float, default=2000.0,
                        help="Max time shown/saved for the correlation function.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    def _compute_stats_corr(graph_dir: Path, transient: float):
        stats_path = graph_dir / "stats.h5"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing stats.h5 in {graph_dir}")
        return _compute_corr_from_stats(stats_path, transient)

    def _compute_degree_corr(graph_dir: Path, transient: float):
        state_path = graph_dir / "state.h5"
        config_path = graph_dir / "config_used.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing state.h5 in {graph_dir}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config_used.json in {graph_dir}")
        return _compute_corr_degree_weighted(state_path, config_path, transient)

    _aggregate_and_save(
        base_dir=base_dir,
        settings=args.settings,
        n_vals=args.Ns,
        transient=args.transient,
        t_max=args.t_max,
        output_name=args.output_name,
        stat_name="deg_weighted_mean_x1",
        compute_fn=_compute_stats_corr,
        plot_name=args.plot_name,
        normalized_plot_name="correlation_degree_weighted_mean_x1_normalized.png",
        tau_plot_name="correlation_time_degree_weighted_mean_x1_normalized.png",
        ylabel="corr(deg_weighted_mean_x1)",
    )

    # Plot degree-weighted mean_x1 time series for critical setting (all graphs per N)
    critical_dir = base_dir / "critical"
    if critical_dir.exists():
        n_vals = [n for n in args.Ns if (critical_dir / f"n{n}").exists()]
        if n_vals:
            fig, axes = plt.subplots(nrows=len(n_vals), figsize=(8, 3 * len(n_vals)), sharex=True)
            if len(n_vals) == 1:
                axes = [axes]
            for ax, n_val in zip(axes, n_vals):
                n_dir = critical_dir / f"n{n_val}"
                graph_dirs = sorted(n_dir.glob("graph_*"))
                if not graph_dirs:
                    continue
                for graph_dir in graph_dirs:
                    stats_path = graph_dir / "stats.h5"
                    if not stats_path.exists():
                        continue
                    try:
                        data, fieldnames = _read_stats(stats_path)
                        t_idx = fieldnames.index("t")
                        x_idx = fieldnames.index("deg_weighted_mean_x1")
                    except Exception as exc:
                        print(f"Skip {stats_path}: {exc}")
                        continue
                    t = data[:, t_idx]
                    x = data[:, x_idx]
                    mask = t > float(args.transient)
                    t = t[mask]
                    x = x[mask]
                    ax.plot(t, x, alpha=0.6)
                ax.set_ylabel("deg_weighted_mean_x1")
            axes[0].set_title("Degree-weighted mean_x1 (critical, all graphs, post-transient)")
            axes[-1].set_xlabel("t")
            fig.tight_layout()
            summary_dir = base_dir / "correlation_functions_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            plot_path = summary_dir / "critical_deg_weighted_mean_x1_timeseries.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot {plot_path}")

    # Empirical measure at final timestep for critical and far (one graph per N)
    settings_for_empirical = ["critical", "far"]
    fig, axes = plt.subplots(nrows=len(settings_for_empirical), ncols=2, figsize=(12, 4 * len(settings_for_empirical)), sharex=True)
    if len(settings_for_empirical) == 1:
        axes = [axes]
    for row_ax, setting in zip(axes, settings_for_empirical):
        ax_unw = row_ax[0]
        ax_w = row_ax[1]
        setting_dir = base_dir / setting
        if not setting_dir.exists():
            continue
        cmap = plt.get_cmap("viridis")
        n_vals = [n for n in args.Ns if (setting_dir / f"n{n}").exists()]
        for i, n_val in enumerate(n_vals):
            n_dir = setting_dir / f"n{n_val}"
            graph_dir = next(iter(sorted(n_dir.glob("graph_*"))), None)
            if graph_dir is None:
                continue
            state_path = graph_dir / "state.h5"
            config_path = graph_dir / "config_used.json"
            if not state_path.exists():
                continue
            if not config_path.exists():
                continue
            try:
                x = _load_final_x1(state_path)
            except Exception as exc:
                print(f"Skip {state_path}: {exc}")
                continue
            # Handle 2N state (x1, x2) vs N state (x1)
            if x.size % 2 == 0:
                n_nodes = x.size // 2
                x1 = x.reshape(n_nodes, 2)[:, 0]
            else:
                x1 = x
            frac = 0.2 + 0.6 * (i / max(1, len(n_vals) - 1))
            color = cmap(frac)
            hist, bin_edges = np.histogram(x1, bins=60, density=True)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax_unw.plot(centers, hist, color=color, label=f"N={n_val} ({graph_dir.name})")

            # Degree-weighted histogram
            try:
                deg = _load_degrees(config_path)
                if deg.size != x1.size:
                    raise ValueError("Degree size does not match x1 size.")
                hist_w, bin_edges_w = np.histogram(x1, bins=60, weights=deg, density=False)
                if hist_w.sum() > 0:
                    hist_w = hist_w / hist_w.sum() / np.diff(bin_edges_w)
                centers_w = 0.5 * (bin_edges_w[:-1] + bin_edges_w[1:])
                ax_w.plot(centers_w, hist_w, color=color, label=f"N={n_val} ({graph_dir.name})")
            except Exception as exc:
                print(f"Skip weighted histogram {config_path}: {exc}")

        ax_unw.set_title(f"{setting} (unweighted)")
        ax_unw.set_ylabel("empirical density")
        ax_unw.legend(frameon=False)
        ax_w.set_title(f"{setting} (degree-weighted)")
        ax_w.legend(frameon=False)

    axes[-1][0].set_xlabel("x1 (final)")
    axes[-1][1].set_xlabel("x1 (final)")
    fig.tight_layout()
    summary_dir = base_dir / "correlation_functions_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_path = summary_dir / "empirical_measure_final_x1.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot {plot_path}")

    # rho_k(x): degree-binned curves + heatmap (critical and far)
    fig, axes = plt.subplots(nrows=len(settings_for_empirical), ncols=2, figsize=(12, 4 * len(settings_for_empirical)))
    if len(settings_for_empirical) == 1:
        axes = [axes]
    for row_ax, setting in zip(axes, settings_for_empirical):
        ax_bins = row_ax[0]
        ax_heat = row_ax[1]
        setting_dir = base_dir / setting
        if not setting_dir.exists():
            continue
        # Use one representative graph per N (first found) and plot the largest N available
        n_vals = [n for n in args.Ns if (setting_dir / f"n{n}").exists()]
        if not n_vals:
            continue
        n_val = max(n_vals)
        n_dir = setting_dir / f"n{n_val}"
        graph_dir = next(iter(sorted(n_dir.glob("graph_*"))), None)
        if graph_dir is None:
            continue
        state_path = graph_dir / "state.h5"
        config_path = graph_dir / "config_used.json"
        if not state_path.exists() or not config_path.exists():
            continue
        try:
            x = _load_final_x1(state_path)
            deg = _load_degrees(config_path)
        except Exception as exc:
            print(f"Skip rho_k {graph_dir}: {exc}")
            continue
        if x.size % 2 == 0:
            n_nodes = x.size // 2
            x1 = x.reshape(n_nodes, 2)[:, 0]
        else:
            x1 = x
        if deg.size != x1.size:
            print(f"Skip rho_k {graph_dir}: degree size mismatch")
            continue

        _plot_rho_k_by_bins(ax_bins, x1, deg, bins=60)
        ax_bins.set_title(f"{setting} (rho_k(x), N={n_val})")
        ax_bins.set_xlabel("x1")
        ax_bins.set_ylabel("density")
        ax_bins.legend(frameon=False)

        im = _plot_rho_k_heatmap(ax_heat, x1, deg, x_bins=60, k_bins=40)
        ax_heat.set_title(f"{setting} (rho_k heatmap, N={n_val})")
        fig.colorbar(im, ax=ax_heat, label="rho_k(x)")

    fig.tight_layout()
    plot_path = summary_dir / "rho_k_x1.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot {plot_path}")


if __name__ == "__main__":
    main()
