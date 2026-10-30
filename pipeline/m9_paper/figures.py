"""
figures.py: M9: the figures of the paper, drawn from the M8 result tables

Module: M9 (camera-ready)

1. What it does
    Draws every figure the paper includes, from results/m8/*.csv (and, for the propensity-score figure, one refit
    of the main propensity model). File names are the ones the LaTeX source includes.

4. Outputs
    results/figures/*.png, also copied to paper/CameraReady/LaTeX/figures_png/.

7. How to run
    python -m pipeline.m9_paper.figures
"""
from __future__ import annotations

import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from pipeline.common.runlog import base_parser, setup_logging  # noqa: E402
from pipeline.m8_analysis import estimators as est  # noqa: E402
from pipeline.m8_analysis.run_analysis import analysis_sample, covariates  # noqa: E402

M8 = config.RESULTS / "m8"
OUT = config.RESULTS / "figures"
PAPER = config.ROOT / "paper" / "CameraReady" / "LaTeX" / "figures_png"
SIG, NS = "#1f77b4", "#9a9a9a"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "savefig.dpi": 200, "savefig.bbox": "tight"})

OUTCOME_LABELS = {
    "1+_mainspace_edit_14d": "1+ article edit (14d)", "n_mainspace_edits_14d": "N article edits (14d)",
    "active_days_30d": "Active days (30d)", "unique_namespace_14d": "Unique namespaces (14d)",
    "reverted_any_14d": "Any edit reverted (14d)", "2+_active_days_14d": "2+ active days (14d)"}
SG_ORDER = ["Technical", "Referent (Q2)", "Appraisal (Q3)", "Normative (Q4)", "Non-substantive",
            "Own work (Q5=Y)", "No own work (Q5=N)"]


def _save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_ps(log) -> None:
    D = analysis_sample()
    X, d = covariates(D), D["y_treat"].astype(float)
    ps = est.fit_propensity(X[:, X.std(axis=0) > 0], d)
    e = ps.e_hat
    cs = pd.read_csv(M8 / "A5_common_support.csv").iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    bins = np.linspace(0, 1, 61)
    ax[0].hist(e[d == 0], bins=bins, density=True, alpha=.6, color="#e06666", label=f"Control (N={int((d == 0).sum()):,})")
    ax[0].hist(e[d == 1], bins=bins, density=True, alpha=.6, color="#6fa8dc", label=f"Treated (N={int((d == 1).sum()):,})")
    for v in (cs.lo, cs.hi):
        ax[0].axvline(v, ls=":", lw=.8, color="gray")
    ax[0].set(title=f"Propensity score distribution (AUC = {ps.auc:.3f})", xlabel="Propensity score", ylabel="Density")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].boxplot([e[d == 0], e[d == 1]], labels=["Control", "Treated"])
    ax[1].set(title="Propensity score by group", ylabel="Propensity score")
    _save(fig, "fig1_ps_combined.png")
    log.info("fig1: AUC %.4f", ps.auc)


def fig_love() -> None:
    b = pd.read_csv(M8 / "A4_covariate_balance.csv")
    b["raw_abs"], b["strat_abs"] = b.raw.abs(), b.strat.abs()
    top = b.sort_values("raw_abs", ascending=False).head(20).iloc[::-1].copy()
    # readable names: embedding components numbered from 1 as in the covariate table, politeness markers shortened
    top["label"] = (top["var"].str.replace(r"^emb_pc(\d+)$", lambda m: f"emb_pc{int(m.group(1)) + 1}", regex=True)
                    .str.replace(r"^q_poly_feature_politeness_==(.*)==$", lambda m: "politeness: " + m.group(1).replace("_", " "), regex=True))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    y = np.arange(len(top))
    ax.scatter(top.raw_abs, y, color="#d62728", label="Raw", zorder=3)
    ax.scatter(top.strat_abs, y, color=SIG, label="Stratified (K=10)", zorder=3)
    for yi, a, s in zip(y, top.raw_abs, top.strat_abs):
        ax.plot([s, a], [yi, yi], color="lightgray", zorder=1)
    ax.axvline(.1, ls="--", color="gray", lw=.8)
    ax.set_yticks(y, top["label"], fontsize=8)
    ax.set(xlabel="|Standardized mean difference|", title="Covariate balance (20 most imbalanced of 164)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig2_love_plot_top20.png")


def _forest(ax, labels, att, lo, hi, scale=1.0, fmt="{:+.1f}pp", sig=None):
    y = np.arange(len(labels))[::-1]
    sig = [(l > 0 or h < 0) for l, h in zip(lo, hi)] if sig is None else sig
    for yi, a, l, h, s in zip(y, att, lo, hi, sig):
        c = SIG if s else NS
        ax.plot([l * scale, h * scale], [yi, yi], color=c, lw=2.5)
        ax.plot(a * scale, yi, "o", color=c, ms=8)
        ax.text(max(hi) * scale * 1.08, yi, fmt.format(a * scale) + (" *" if s else ""), va="center", fontsize=9)
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", alpha=.3)


def fig_overall_forest() -> None:
    m = pd.read_csv(M8 / "B_main_results.csv")
    m = m[m.Outcome.isin(["1+_mainspace_edit_14d", "2+_active_days_14d", "reverted_any_14d"])]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    _forest(ax, [OUTCOME_LABELS[o] for o in m.Outcome], m.ATT, m.CI_lo, m.CI_hi, scale=100)
    ax.set(xlabel="ATT (percentage points)")
    ax.set_xlim(right=m.CI_hi.max() * 100 * 1.35)
    _save(fig, "fig4_overall_forest.png")


def fig_robustness() -> None:
    r = pd.read_csv(M8 / "C6_robustness_summary.csv")
    r = r[~r.Spec.str.startswith("Pseudo")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(r))[::-1]
    for yi, (_, row) in zip(y, r.iterrows()):
        main = row.Spec.startswith("Main")
        c = SIG if main else "#555555"
        ax.plot([row.CI_lo, row.CI_hi], [yi, yi], color=c, lw=3 if main else 1.6)
        ax.plot(row.ATT, yi, "o" if main else "s", color=c, ms=9 if main else 6)
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(y, r.Spec, fontsize=9)
    ax.set(xlabel="ATT (1+ article edit within 14 days)", title="ATT across specifications")
    ax.grid(axis="x", alpha=.3)
    _save(fig, "fig5_robustness.png")


def fig_subgroup_forest() -> None:
    s = pd.read_csv(M8 / "subgroup_att.csv")
    s = s[s.Outcome == "1+_mainspace_14d"].set_index("Subgroup")
    full = s.loc["Full sample"]
    s = s.loc[SG_ORDER]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [f"{g} (N={int(n):,})" for g, n in zip(s.index, s.N)]
    _forest(ax, labels, s.ATT, s.CI_lo, s.CI_hi, fmt="{:+.3f}")
    ax.axvline(full.ATT, ls="--", color="gray", lw=1, label=f"Full sample ATT = {full.ATT:+.3f}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=False)
    ax.set(xlabel="ATT (1+ article edit within 14 days)")
    ax.set_xlim(right=s.CI_hi.max() * 1.45)
    _save(fig, "fig6_subgroup_forest.png")


def fig_strata_sensitivity() -> None:
    s = pd.read_csv(M8 / "sensitivity_strata.csv")
    s = s[s.DV == "primary"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for g in ["Full sample"] + SG_ORDER:
        x = s[s.Subgroup == g].sort_values("K")
        if len(x):
            ax.plot(x.K, x.ATT, marker="o", label=g, lw=2.2 if g == "Full sample" else 1.2)
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="Number of strata K", ylabel="ATT (1+ article edit, 14d)", xticks=sorted(s.K.unique()))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3)
    _save(fig, "fig7_strata_sensitivity.png")


def fig_reply_features() -> None:
    r = pd.read_csv(M8 / "reply_association.csv")
    r = r.reindex(r.Beta_std.abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(7, 6))
    lo, hi = r.Beta_std - 1.96 * r.SE, r.Beta_std + 1.96 * r.SE
    _forest(ax, list(r.Feature), r.Beta_std, lo, hi, fmt="{:+.3f}", sig=list(r.p < .05))
    ax.set(xlabel="Standardized coefficient (1+ article edit within 14 days)")
    ax.set_xlim(right=hi.max() * 2.2)
    _save(fig, "fig8_reply_features.png")


def fig_monthly() -> None:
    """Appendix figure: share of the one-year cohort who edit an article in each month (followup_monthly.csv).
    Description only; the tests of a lasting effect are the later periods (followup_cohort_periods.csv)."""
    t = pd.read_csv(M8 / "followup_monthly.csv")
    treated, control, ink, muted, grid = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e6e5e0"
    with plt.rc_context({"font.size": 7.5, "axes.edgecolor": muted, "xtick.color": muted, "ytick.color": muted,
                         "axes.labelcolor": ink}):
        fig, ax = plt.subplots(figsize=(3.3, 2.1))
        x = t.month.to_numpy()
        ax.plot(x, 100 * t.Treat_share, color=treated, lw=1.6, marker="o", ms=3.5, label="Received a reply")
        ax.plot(x, 100 * t.Ctrl_share_reweighted, color=control, lw=1.6, marker="o", ms=3.5, mfc="white", mew=1.2,
                label="No reply (reweighted)")
        ax.set_ylim(0, 26)
        ax.set_xticks(x)
        ax.set_ylabel("Edited an article (%)")
        ax.set_xlabel("Month after the reply (30-day periods)")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        _save(fig, "fig_monthly_activity.png")


def fig_heatmap() -> None:
    s = pd.read_csv(M8 / "subgroup_att.csv")
    cols = ["1+_mainspace_14d", "n_edits_14d", "active_days_30d", "unique_ns_14d", "reverted_any_14d", "2+_active_days_14d"]
    att = s.pivot(index="Subgroup", columns="Outcome", values="ATT").loc[SG_ORDER, cols]
    dd = s.pivot(index="Subgroup", columns="Outcome", values="Cohen_d").loc[SG_ORDER, cols]
    sig = s.assign(sg=s.CI_lo.gt(0) | s.CI_hi.lt(0)).pivot(index="Subgroup", columns="Outcome", values="sg").loc[SG_ORDER, cols]
    fig, ax = plt.subplots(figsize=(11, 4))
    v = np.nanmax(np.abs(dd.to_numpy()))
    im = ax.imshow(dd.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(att.shape[0]):
        for j in range(att.shape[1]):
            a = att.iat[i, j]
            ax.text(j, i, (f"{a:+.3f}" if abs(a) < 0.1 else f"{a:+.2f}") + ("*" if sig.iat[i, j] else ""),
                    ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)), ["1+ edit\n14d", "N edits\n14d", "Active days\n30d", "Unique NS\n14d",
                                     "Reverted\n14d", "2+ active\ndays 14d"])
    ax.set_yticks(range(len(SG_ORDER)), SG_ORDER)
    fig.colorbar(im, ax=ax, shrink=.8, label="Cohen's d (cell text: ATT)")
    ax.set_title("Within-subgroup ATT by question type and outcome (* = 95% CI excludes zero)")
    _save(fig, "fig_heatmap_subgroup_dv.png")


def fig_mentor_pool() -> None:
    """Appendix figure: mentors on the list by month (top) and askers per mentor in the automatic pool (bottom),
    from descriptive_mentor_pool_by_month.csv. Dotted lines mark the months in which the share of new accounts that
    could see the mentorship entry points rose (10%, 25%, 50%, then 75% and 100% in February 2025)."""
    import matplotlib.transforms as mt
    t = pd.read_csv(M8 / "descriptive_mentor_pool_by_month.csv")
    x = pd.to_datetime(t.month + "-01")
    rollout = [("2022-03", "10%"), ("2023-07", "25%"), ("2023-10", "50%"), ("2025-02", "75% / 100%")]
    auto, manual, ink, muted, grid = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e6e5e0"
    with plt.rc_context({"font.size": 7.5, "axes.edgecolor": muted, "xtick.color": muted, "ytick.color": muted,
                         "axes.labelcolor": ink}):
        fig, ax = plt.subplots(2, 1, figsize=(3.3, 3.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
        ax[0].plot(x, t.auto_pool, color=auto, lw=1.6)
        ax[0].plot(x, t.manual_only, color=manual, lw=1.6)
        ax[0].text(pd.Timestamp("2021-05-15"), 112, "Automatic\nassignment pool", color=ink, fontsize=7, va="bottom")
        ax[0].text(pd.Timestamp("2025-04-01"), 92, "Manual claims\nonly", color=ink, fontsize=7, va="bottom")
        ax[0].set_ylim(0, 230)
        ax[0].set_ylabel("Mentors on the list")
        ax[1].plot(x, t.askers_per_pool_mentor, color=ink, lw=1.4)
        ax[1].set_ylim(0, None)
        ax[1].set_ylabel("Askers per month\nper pool mentor")
        for i, a in enumerate(ax):
            tr = mt.blended_transform_factory(a.transData, a.transAxes)
            for mth, lab in rollout:
                a.axvline(pd.Timestamp(mth + "-01"), color=muted, lw=0.6, ls=":")
                if i == 0:
                    a.text(pd.Timestamp(mth + "-01"), 1.02, lab, color=muted, fontsize=6.5, va="bottom",
                           ha="right" if lab == "25%" else "left", transform=tr)
            a.grid(axis="y", color=grid, lw=0.6)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        fig.tight_layout()
        _save(fig, "fig_mentor_pool.png")


def main(argv: list[str] | None = None) -> None:
    base_parser("M9: paper figures from the M8 result tables.").parse_args(argv)
    log = setup_logging("m9_figures")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ps(log)
    for f in (fig_love, fig_overall_forest, fig_robustness, fig_subgroup_forest, fig_strata_sensitivity,
              fig_reply_features, fig_heatmap, fig_monthly, fig_mentor_pool):
        f()
    PAPER.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.png"):
        shutil.copy2(p, PAPER / p.name)
    log.info("figures written to %s and %s", OUT, PAPER)


if __name__ == "__main__":
    main(sys.argv[1:])
