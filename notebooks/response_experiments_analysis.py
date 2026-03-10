import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from matplotlib import pyplot as plt
    import numpy as np
    from dyn_net.utils.data_analysis import find_repo_root, load_response_records

    def plot_records(
        records,
        *,
        title: str | None = None,
        variable_name: str = "deg_weighted_mean_x1",
    ):
        fig, ax = plt.subplots()
        for setting, setting_records in records.items():
            sorted_records = sorted(
                setting_records,
                key=lambda record: record.config["network"]["params"]["n"],
            )
            cmap = plt.get_cmap("Reds" if setting == "critical" else "Blues")
            n_records = len(sorted_records)
            shade_values = (
                np.linspace(0.4, 0.9, n_records) if n_records > 1 else [0.65]
            )

            for record, shade in zip(sorted_records, shade_values):
                record.linear.plot(
                    x="t",
                    y=variable_name,
                    ax=ax,
                    color=cmap(shade),
                    legend=False,
                )
        if title:
            ax.set_title(title)
        return fig, ax

    return find_repo_root, load_response_records, np, plot_records, plt


@app.cell
def _(find_repo_root, load_response_records):
    NETWORK_LABEL = "poisson_annealed"
    N_BY_SETTING = {"critical": [1000, 5000, 10000], "far": [1000, 10000]}
    EPS_TAG = "01"

    repo_root = find_repo_root()
    records_constant = load_response_records(
        repo_root=repo_root,
        network_label=NETWORK_LABEL,
        perturbation_type="constant",
        settings=N_BY_SETTING,
        eps_tag=EPS_TAG,
    )
    return EPS_TAG, NETWORK_LABEL, records_constant, repo_root


@app.cell
def _(plot_records, plt, records_constant, repo_root):
    fig_constant, ax_constant = plot_records(
        records=records_constant,
        variable_name="deg_weighted_mean_x1",
        title="Constant perturbation",
    )

    ax_constant.grid(linestyle="--", alpha=0.6)
    ax_constant.set_xlabel(xlabel=r"$t$", size=16)
    ax_constant.set_ylabel(ylabel=r"$\tilde{G}_x(t)$", size=16)
    fig_constant.tight_layout()
    fig_constant.savefig(
        fname=repo_root/"figures/GreensFunction_Poisson_Annealed_Constant.png",
        dpi=400,
        bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(EPS_TAG, NETWORK_LABEL, load_response_records, repo_root):
    N_BY_SETTING_ROT = {"critical": [1000, 10000], "far": [1000, 10000]}

    records_alpha_rot = load_response_records(
        repo_root=repo_root,
        network_label=NETWORK_LABEL,
        perturbation_type="alpha_rot",
        settings=N_BY_SETTING_ROT,
        eps_tag=EPS_TAG,
    )
    return (records_alpha_rot,)


@app.cell
def _(plot_records, plt, records_alpha_rot, repo_root):
    fig_alpha_rot, ax_alpha_rot = plot_records(
        records=records_alpha_rot,
        variable_name="deg_weighted_mean_x1x2_x1sq_minus1",
        title="Rotational perturbation",
    )

    ax_alpha_rot.grid(linestyle="--", alpha=0.6)
    ax_alpha_rot.set_xlabel(xlabel=r"$t$", size=16)
    ax_alpha_rot.set_ylabel(ylabel=r"$\tilde{G}_{\gamma x}(t)$", size=16)
    ax_alpha_rot.set_xlim((-1, 10))

    fig_alpha_rot.tight_layout()
    fig_alpha_rot.savefig(
        fname=repo_root/"figures/GreensFunction_Poisson_Annealed_Rotation.png",
        dpi=400,
        bbox_inches="tight")

    plt.show()
    return


@app.cell
def _(plot_records, plt, records_alpha_rot, repo_root):
    fig_alpha_rot_x  ,ax_alpha_rot_x = plot_records(
        records=records_alpha_rot,
        variable_name="deg_weighted_mean_x1",
        title="Rotational perturbation")
    ax_alpha_rot_x.grid(linestyle="--", alpha=0.6)
    ax_alpha_rot_x.set_xlabel(xlabel=r"$t$", size=16)
    ax_alpha_rot_x.set_ylabel(ylabel=r"$\tilde{G}_{x}(t)$", size=16)
    ax_alpha_rot_x.set_ylim((-0.01,0.01))
    ax_alpha_rot_x.set_xlim((-1,10))

    fig_alpha_rot_x.tight_layout()

    fig_alpha_rot_x.savefig(
        fname=repo_root/"figures/GreenFunctions_Poisson_Annealed_Rotation_X.png",
        dpi=400,
        bbox_inches="tight"
    )

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Spectral Analysis
    """)
    return


@app.cell
def _(np, plt, records_constant, repo_root):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = plt.subplots()
    _ax_inset = inset_axes(
        ax,
        width="58%",
        height="52%",
        loc="upper right",
        bbox_to_anchor=(0.32, 0.41, 0.62, 0.55),
        bbox_transform=ax.transAxes,
    )
    _ax_integral = inset_axes(
        ax,
        width="58%",
        height="52%",
        loc="lower right",
        bbox_to_anchor=(0.32, 0.17, 0.62, 0.55),
        bbox_transform=ax.transAxes,
    )
    for _setting, _records in records_constant.items():
        _sorted_records = sorted(
            _records, key=lambda record: record.config["network"]["params"]["n"]
        )
        _cmap = plt.get_cmap("Reds" if _setting == "critical" else "Blues")
        _n_records = len(_sorted_records)
        _shade_values = (
            np.linspace(0.4, 0.9, _n_records) if _n_records > 1 else [0.65]
        )

        for _record, _shade in zip(_sorted_records, _shade_values):
            _t = _record.linear["t"].to_numpy()
            _dt = _t[1] - _t[0]
            _g = _record.linear["deg_weighted_mean_x1"].to_numpy()
            _omega = 2 * np.pi * np.fft.fftfreq(_t.size, d=_dt)
            _chi = _dt * np.fft.fft(_g)
            _chi = np.conj(_chi)
            _chi *= np.exp(1j * _omega * _dt)  # account for time shift

            _omega_plot = _omega.copy()
            _omega_plot[_omega_plot == 0] = 5e-5
            _order = np.argsort(_omega)
            _omega = _omega[_order]
            _omega_plot = _omega_plot[_order]
            _chi = _chi[_order]
            _mask = _omega_plot > 0
            _n_label = _record.config["network"]["params"]["n"]
            ax.plot(
                _omega_plot[_mask],
                np.real(_chi[_mask]),
                color=_cmap(_shade),
            )

            _mask_small = np.abs(_omega) <= 0.05
            _ax_inset.plot(
                _omega[_mask_small],
                _omega[_mask_small] * np.imag(_chi[_mask_small]),
                color=_cmap(_shade),
            )

            _c = -0.02
            _mask_int = _omega >= _c
            _omega_int = _omega[_mask_int]
            _chi_int = _chi[_mask_int]
            _y = np.real(_chi_int)
            _dx = np.diff(_omega_int)
            _traps = 0.5 * (_y[1:] + _y[:-1]) * _dx
            _int_vals = np.concatenate(([0.0], np.cumsum(_traps)))
            _ax_integral.plot(
                _omega_int,
                _int_vals,
                color=_cmap(_shade),
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$",size=13)
    ax.set_ylabel(r"$\mathbf{Re} \chi(\omega)$",size=13)
    ax.grid(linestyle="--", alpha=0.6)
    _ax_inset.set_xlim((-0.012, 0.012))
    _ax_inset.set_xlabel(r"$\omega$", size=13)
    _ax_inset.set_ylabel(r"$\omega\,\mathbf{Im} \chi(\omega)$", size=13)
    _ax_inset.grid(linestyle="--", alpha=0.4)
    _ax_integral.set_xlim((-0.015, 0.015))
    _ax_integral.set_ylim((-0.1,1.1))
    _ax_integral.set_xlabel(r"$\omega$", size=13)
    _ax_integral.set_ylabel(r"$\int_{-c}^{\omega} \mathbf{Re}\chi(s)\,ds$", size=13)
    _ax_integral.grid(linestyle="--", alpha=0.4)

    fig.savefig(
        fname=repo_root/"figures/susceptibility_Poisson_Annealed_constant.png",
        dpi=400,
        bbox_inches="tight"
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Poisson (Quenched)
    """)
    return


@app.cell
def _(load_response_records, repo_root):
    _NETWORK_LABEL = "poisson"
    _N_BY_SETTING = {"critical": [1000, 5000, 10000], "far": [1000, 10000]}
    _EPS_TAG = "001"

    records_constant_poisson = load_response_records(
        repo_root=repo_root,
        network_label=_NETWORK_LABEL,
        perturbation_type="constant",
        settings=_N_BY_SETTING,
        eps_tag=_EPS_TAG,
    )
    return (records_constant_poisson,)


@app.cell
def _(plot_records, plt, records_constant_poisson, repo_root):
    fig_poisson, ax_poisson = plot_records(
        records=records_constant_poisson,
        variable_name="deg_weighted_mean_x1",
        title="Constant perturbation",
    )

    ax_poisson.grid(linestyle="--", alpha=0.6)
    ax_poisson.set_xlabel(xlabel=r"$t$", size=16)
    ax_poisson.set_ylabel(ylabel=r"$\tilde{G}_x(t)$", size=16)
    fig_poisson.savefig(
        fname=repo_root/"figures/GreensFunctions_PoissonQuenched_Constant.png",
        dpi=400,
        bbox_inches="tight"
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
