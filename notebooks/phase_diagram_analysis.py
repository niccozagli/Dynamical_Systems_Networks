import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import json
    from tqdm import tqdm
    import h5py
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from dyn_net.utils.criticality import (
        find_theta_c_from_degree_distribution,
        solve_mean_field_from_config,
    )
    from dyn_net.utils.data_analysis import find_repo_root

    def read_config(config_path: Path) -> dict:
        return json.loads(Path(config_path).read_text())

    def read_aggregate_dfs(agg_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        with h5py.File(agg_path, "r", swmr=True) as fh:
            mean_dset = fh["mean"]
            std_dset = fh["std"]
            mean_dset.refresh()
            std_dset.refresh()
            field_attr = fh.attrs.get("fieldnames")
            if field_attr is None:
                raise ValueError(f"Missing fieldnames in {agg_path}")
            fieldnames = [s.decode("utf-8") for s in np.asarray(field_attr).tolist()]
            mean_df = pd.DataFrame(mean_dset[...], columns=fieldnames)
            std_df = pd.DataFrame(std_dset[...], columns=fieldnames)
            return mean_df, std_df

    def list_phase_diagram_runs(base_dir: Path) -> list[Path]:
        base_dir = Path(base_dir)
        run_dirs = [
            p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("row_")
        ]
        return sorted(run_dirs, key=lambda p: p.name)

    def summarize_mean_x1_aggregate(
        agg_path: Path,
        *,
        frac: float = 0.8,
        use_abs: bool = False,
    ) -> dict:
        mean_df, std_df = read_aggregate_dfs(agg_path)
        with h5py.File(agg_path, "r", swmr=True) as fh:
            count = int(fh["count"][()])
        col = "deg_weighted_mean_x1_abs" if use_abs else "deg_weighted_mean_x1"
        if col not in mean_df.columns:
            raise ValueError(f"{col} column missing from aggregate.")
        if "t" not in mean_df.columns:
            raise ValueError("t column missing from aggregate.")
        t_max = float(mean_df["t"].max())
        t_start = frac * t_max
        tail = mean_df[mean_df["t"] >= t_start]
        if tail.empty:
            raise ValueError("No samples in tail window; check frac/t_max.")
        series = tail[col]
        time_var = float(series.var(ddof=0))
        time_std = float(np.sqrt(time_var))
        tail_std = std_df.loc[tail.index, col]
        return {
            "mean_x1": float(series.mean()),
            "time_var_x1": time_var,
            "time_std_x1": time_std,
            "var_x1": float((tail_std ** 2).mean()),
            "t_start": float(tail["t"].iloc[0]),
            "t_end": float(tail["t"].iloc[-1]),
            "count": count,
        }

    def compute_theta_c_from_config_data(
        config_used: dict, *, theta_bracket: tuple[float, float] = (1e-6, 1.0)
    ) -> float:
        degree_distribution = config_used["network"]["params"]["degree_distribution"]
        sigma = float(config_used["noise"]["params"]["sigma"])
        return find_theta_c_from_degree_distribution(
            degree_distribution=degree_distribution,
            sigma=sigma,
            theta_bracket=theta_bracket,
        )

    def compute_theta_c_from_config(
        config_path: Path, *, theta_bracket: tuple[float, float] = (1e-6, 1.0)
    ) -> float:
        config_used = read_config(config_path)
        return compute_theta_c_from_config_data(
            config_used, theta_bracket=theta_bracket
        )

    def build_phase_diagram(
        sweep_dir: Path,
        *,
        frac: float = 0.8,
        theta_c: float | None = None,
    ) -> pd.DataFrame:
        sweep_dir = Path(sweep_dir)
        rows: list[dict] = []

        run_dirs = list_phase_diagram_runs(sweep_dir)
        if not run_dirs:
            raise ValueError(f"No row_* subfolders found in {sweep_dir}")

        if theta_c is None:
            first_cfg = read_config(run_dirs[0] / "config_used.json")
            theta_c = compute_theta_c_from_config_data(first_cfg)

        for run_dir in run_dirs:
            agg_path = run_dir / "aggregate.h5"
            if not agg_path.exists():
                print(f"{agg_path} not found")
                continue
            cfg_path = run_dir / "config_used.json"
            if not cfg_path.exists():
                print(f"{cfg_path} not found")
                continue
            config_used = read_config(cfg_path)
            theta = float(config_used["system"]["params"]["theta"])
            use_abs = theta > theta_c
            summary = summarize_mean_x1_aggregate(
                agg_path, frac=frac, use_abs=use_abs
            )
            rows.append(
                {
                    "run": run_dir.name,
                    "theta": theta,
                    "use_abs": use_abs,
                    **summary,
                }
            )

        phase_df = pd.DataFrame(rows)
        if not phase_df.empty:
            phase_df = phase_df.sort_values("theta").reset_index(drop=True)
        return phase_df

    return (
        build_phase_diagram,
        compute_theta_c_from_config,
        find_repo_root,
        list_phase_diagram_runs,
        np,
        plt,
        read_config,
        solve_mean_field_from_config,
        tqdm,
    )


@app.cell
def _(
    build_phase_diagram,
    compute_theta_c_from_config,
    find_repo_root,
    list_phase_diagram_runs,
    np,
    read_config,
    solve_mean_field_from_config,
    tqdm,
):
    NETWORK_LABEL = "poisson_annealed"
    N = 5000
    FRAC = 0.7

    repo_root = find_repo_root()
    phase_dir = repo_root / f"results/phase_diagram/{NETWORK_LABEL}/n{N}"
    run_dirs = list_phase_diagram_runs(phase_dir)
    if not run_dirs:
        raise ValueError(f"No runs found in {phase_dir}")

    first_cfg = run_dirs[0] / "config_used.json"

    theta_c = compute_theta_c_from_config(first_cfg)
    phase_df = build_phase_diagram(phase_dir, frac=FRAC, theta_c=theta_c)
    phase_df["theta_scaled"] = phase_df["theta"] / theta_c
    phase_df["std"] = np.sqrt(phase_df["var_x1"])
    _config_used = read_config(first_cfg)
    _thetas = phase_df["theta"].to_numpy()
    _m_vals = []
    for _theta in tqdm(_thetas, desc="Mean-field solve"):
        _m_vals.append(
            solve_mean_field_from_config(
                _config_used,
                theta=float(_theta),
                bracket=(0.0, 2.0),
                prefer_nonzero=True,
                positive_root=True,
            )
        )
    phase_df["mean_field_m"] = _m_vals
    return phase_df, repo_root, theta_c


@app.cell
def _(phase_df, plt, repo_root):
    fig, ax = plt.subplots(figsize=(6, 4))


    ax.plot(
        phase_df["theta_scaled"],
        phase_df["mean_field_m"],
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    phase_df.plot(x="theta_scaled", y="mean_x1", marker="o", linestyle="",ax=ax, legend=False)
    ax.fill_between(
        phase_df["theta_scaled"],
        phase_df["mean_x1"] - phase_df["std"],
        phase_df["mean_x1"] + phase_df["std"],
        alpha=0.3,
    )
    ax.set_xlabel(r"$\theta/\theta_c$",size=16)
    ax.set_ylabel(r"$\langle |x| \rangle_0$",size=16)
    ax.set_title("Poisson Annealed")
    ax.grid(linestyle='--',alpha=0.4)
    plt.show()
    fig.tight_layout()
    fig.savefig(
        fname=repo_root/"figures/phase_diagram_poisson_annealed.png",
        dpi=400,
        bbox_inches="tight")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Poisson
    """)
    return


@app.cell
def _(
    build_phase_diagram,
    compute_theta_c_from_config,
    list_phase_diagram_runs,
    np,
    read_config,
    repo_root,
    solve_mean_field_from_config,
    theta_c,
    tqdm,
):
    _NETWORK_LABEL = "poisson"
    _N = 10000
    _FRAC = 0.7

    _phase_dir = repo_root / f"results/phase_diagram/{_NETWORK_LABEL}/n{_N}"
    _run_dirs = list_phase_diagram_runs(_phase_dir)
    if not _run_dirs:
        raise ValueError(f"No runs found in {_phase_dir}")

    _first_cfg = _run_dirs[0] / "config_used.json"

    _theta_c = compute_theta_c_from_config(_first_cfg)
    _phase_df = build_phase_diagram(_phase_dir, frac=_FRAC, theta_c=_theta_c)
    _phase_df["theta_scaled"] = _phase_df["theta"] / theta_c
    _phase_df["std"] = np.sqrt(_phase_df["var_x1"])
    _config_used = read_config(_first_cfg)
    _thetas = _phase_df["theta"].to_numpy()
    _m_vals = []
    for _theta in tqdm(_thetas, desc="Mean-field solve (Poisson)"):
        _m_vals.append(
            solve_mean_field_from_config(
                _config_used,
                theta=float(_theta),
                bracket=(0.0, 2.0),
                prefer_nonzero=True,
                positive_root=True,
            )
        )
    _phase_df["mean_field_m"] = _m_vals

    phase_df_poisson = _phase_df.copy()
    return (phase_df_poisson,)


@app.cell
def _(phase_df, phase_df_poisson, plt, repo_root):
    fig_poisson, ax_poisson = plt.subplots(figsize=(6, 4))
    phase_df_poisson.plot(x="theta_scaled", y="mean_x1", marker="o", ax=ax_poisson, legend=False)
    if "mean_field_m" in phase_df.columns:
        ax_poisson.plot(
            phase_df_poisson["theta_scaled"],
            phase_df_poisson["mean_field_m"],
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
    ax_poisson.fill_between(
        phase_df_poisson["theta_scaled"],
        phase_df_poisson["mean_x1"] - phase_df_poisson["std"],
        phase_df_poisson["mean_x1"] + phase_df_poisson["std"],
        alpha=0.2,
    )
    ax_poisson.set_xlabel(r"$\theta/\theta_c$",size=16)
    ax_poisson.set_ylabel(r"$\langle |x| \rangle_0$",size=16)
    ax_poisson.set_title("Poisson Quenched")

    ax_poisson.grid(linestyle='--',alpha=0.4)
    fig_poisson.tight_layout()
    plt.show()
    fig_poisson.savefig(
        fname=repo_root/"figures/phase_diagram_poisson_quenched.png",
        dpi=400,
        bbox_inches="tight")

    return


@app.cell
def _(list_phase_diagram_runs, read_config, repo_root):
    _NETWORK_LABEL = "poisson"
    _N=10000
    _phase_dir = repo_root / f"results/phase_diagram/{_NETWORK_LABEL}/n{_N}"
    _run_dirs = list_phase_diagram_runs(_phase_dir)
    _first_cfg = _run_dirs[0] / "config_used.json"
    _config_used = read_config(_first_cfg)
    _config_used
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
