"""P7: regenerate the four manuscript figures from the corrected outputs.

The original figures were produced by inline heredoc code (session 20261108_164233_00ccd654,
ext-claude-3 state.db, messages 3939/3943/3947; no script file existed). The plotting code below
reproduces that code verbatim except for (i) the corrected input files, (ii) the cell counts in the
titles, (iii) fig_a4: log y-axis (corrected medians reach 233) and a third panel with the share of
attempted fits without a valid estimate (all attempts), as requested for the correction.
Pre-correction PDFs are kept in latex/pre_correction/.
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/papers/jams-frn50k")
CORR = f"{ROOT}/correction_20261122"
SB = f"{CORR}/sandbox/main"
OUT = f"{ROOT}/latex"


def fig_dose_and_decomp():
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    df = pd.read_csv(f"{SB}/pending_final_dose_wildci.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = df["quintile"].values
    ax.bar(x, df["naive"], width=-0.18, align="edge", label="naive (observed)", color="#888888")
    ax.bar(x, df["recovered"], width=0.18, align="edge", label="recovered (M)", color="#1f77b4")
    ax.set_xlabel("Discount-depth quintile")
    ax.set_ylabel("Mean log(1+sales)")
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Dose-response: naive vs. recovered, by discount depth", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_dose_response.pdf")
    plt.close(fig)

    # ponytail: original used decomp_cells.csv bias_pct_* (dist step NOT on |b_M|); caption and Table
    # commondenom use the common denominator, so the figure now reads the common-denominator columns.
    dc = pd.read_csv(f"{SB}/pending_table4_commondenom.csv").dropna(subset=["beta_rec_depth", "beta_rec_promo"])
    n = len(dc)
    comp_cols_depth = ["cd_omission_depth", "cd_dist_depth", "cd_pure_cens_depth"]
    comp_cols_promo = ["cd_omission_promo", "cd_dist_promo", "cd_pure_cens_promo"]
    labels = ["Covariate\nomission", "Distributional/\nlink form", "Censoring\ncorrection"]
    med_depth = [dc[c].abs().median() for c in comp_cols_depth]
    med_promo = [dc[c].abs().median() for c in comp_cols_promo]
    print("decomposition medians depth", np.round(med_depth, 4), "promo", np.round(med_promo, 4), "n", n)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xpos = np.arange(len(labels)); w = 0.35
    ax.bar(xpos - w / 2, med_depth, w, label="depth", color="#1f77b4")
    ax.bar(xpos + w / 2, med_promo, w, label="promo", color="#ff7f0e")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median |component| (on $|b_M|$ denominator)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Decomposition: median component magnitude ({n} cells)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_decomposition.pdf")
    plt.close(fig)


def fig_a4():
    plt.rcdefaults()
    df = pd.read_csv(f"{CORR}/out/p5_e2/corrected/e2_headline_tbl2.csv")
    a4 = df[df["arm"] == "A4_nonignorable"]
    fd = pd.read_csv(f"{CORR}/out/fit_diagnostics.csv", low_memory=False,
                     usecols=["source", "arm", "param", "model", "corrected_status"])
    e = fd[(fd.source == "E2") & (fd.arm == "A4_nonignorable")]
    fail = 100 * (1 - e.assign(v=e.corrected_status.eq("valid")).groupby(["model", "param"]).v.mean())
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    estimators = ["naive", "g", "m", "nb2", "mix"]
    colors = {"naive": "#888888", "g": "#2ca02c", "m": "#1f77b4", "nb2": "#d62728", "mix": "#9467bd"}
    labels_map = {"naive": "naive", "g": "G", "m": "M", "nb2": "NB2", "mix": "MIX"}
    for ax, coef, title in zip(axes[:2], ["d", "p"], ["Depth coefficient", "Promo coefficient"]):
        sub = a4[a4["coef"] == coef].sort_values("param")
        for est in estimators:
            ax.plot(sub["param"], sub[est], marker="o", label=labels_map[est], color=colors[est])
        ax.set_xlabel(r"Shared-shock strength $\rho$")
        ax.set_title(title + " (valid fits)", fontsize=10)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yscale("log")
    axes[1].sharey(axes[0])
    axes[0].set_ylabel("Median relative error (log scale)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[2]
    for est in ["m", "nb2", "mix"]:
        s = fail.loc[est.upper()].sort_index()
        ax.plot(s.index, s.values, marker="o", label=labels_map[est], color=colors[est])
    ax.plot([0, 0.5, 1.0], [0, 0, 0], marker="o", color="#888888", label="naive, G")
    ax.set_ylim(-3, 103)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel(r"Shared-shock strength $\rho$")
    ax.set_ylabel("Fits without valid estimate (%)")
    ax.set_title("All attempts (2,700 per $\\rho$)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.suptitle("E2 arm A4 (non-ignorable stockout): sensitivity to shared-shock strength", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/fig_a4_sensitivity.pdf")
    plt.close(fig)
    print("A4 failure %:\n", fail.round(1).unstack().to_string())


def fig_hourly():
    plt.rcdefaults()
    summ = pd.read_csv(f"{CORR}/out/p4_hourly_summary.csv")
    v = summ[summ.corrected_M_status == "valid"]
    nv, rc = [], []
    for _, r in v.iterrows():
        z = np.load(f"{CORR}/out/p4/{int(r['city'])}_{int(r['cat'])}_hourly.npz")
        nv.append(z["naive_h"]); rc.append(z["rec_h_HB"])
    naive_mean, rec_mean = np.mean(nv, 0), np.mean(rc, 0)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    hours = np.arange(24)
    ax.plot(hours, naive_mean, marker="o", ms=3, label="naive (observed)", color="#888888")
    ax.plot(hours, rec_mean, marker="o", ms=3, label="recovered (M)", color="#1f77b4")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean promotion coefficient (per-hour)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Hour-of-day profile: naive vs. recovered ({len(v)} cells)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hourly_profile.pdf")
    plt.close(fig)
    print("hourly cells", len(v))


if __name__ == "__main__":
    fig_dose_and_decomp(); fig_a4(); fig_hourly()
