import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from matplotlib import pyplot as plt
    import numpy as np
    from dyn_net.utils.data_analysis import find_repo_root, load_response_records

    def plot_records(records, *, title: str | None = None,variable_name : str = "deg_weighted_mean_x1"):
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
        return fig , ax

    return find_repo_root, load_response_records, plot_records, plt


@app.cell
def _(find_repo_root, load_response_records, plot_records, plt):
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
    fig_constant , ax_constant = plot_records(
    records=records_constant,
    variable_name="deg_weighted_mean_x1",
    title="Constant perturbation")

    ax_constant.grid(linestyle="--", alpha=0.6)
    ax_constant.set_xlabel(xlabel=r"$t$", size=16)
    ax_constant.set_ylabel(ylabel=r"$\tilde{G}_x(t)$", size=16)
    plt.show()
    return EPS_TAG, NETWORK_LABEL, repo_root


@app.cell
def _(
    EPS_TAG,
    NETWORK_LABEL,
    load_response_records,
    plot_records,
    plt,
    repo_root,
):
    N_BY_SETTING_ROT = {"critical": [1000, 10000], "far": [1000, 10000]}

    records_alpha_rot = load_response_records(
        repo_root=repo_root,
        network_label=NETWORK_LABEL,
        perturbation_type="alpha_rot",
        settings=N_BY_SETTING_ROT,
        eps_tag=EPS_TAG,
    )
    fig_alpha_rot  ,ax_alpha_rot = plot_records(
        records=records_alpha_rot,
        variable_name="deg_weighted_mean_x1x2_x1sq_minus1",
        title="Rotational perturbation")

    ax_alpha_rot.grid(linestyle="--", alpha=0.6)
    ax_alpha_rot.set_xlabel(xlabel=r"$t$", size=16)
    ax_alpha_rot.set_ylabel(ylabel=r"$\tilde{G}_{\gamma x}(t)$", size=16)
    ax_alpha_rot.set_xlim((-1,10))
    plt.show()
    return (records_alpha_rot,)


@app.cell
def _(plot_records, plt, records_alpha_rot):
    fig_alpha_rot_x  ,ax_alpha_rot_x = plot_records(
        records=records_alpha_rot,
        variable_name="deg_weighted_mean_x1",
        title="Rotational perturbation")
    ax_alpha_rot_x.grid(linestyle="--", alpha=0.6)
    ax_alpha_rot_x.set_xlabel(xlabel=r"$t$", size=16)
    ax_alpha_rot_x.set_ylabel(ylabel=r"$\tilde{G}_{x}(t)$", size=16)
    ax_alpha_rot_x.set_ylim((-0.01,0.01))
    ax_alpha_rot_x.set_xlim((-1,10))
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
