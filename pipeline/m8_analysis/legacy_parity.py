"""
legacy_parity.py: Stage A. Use the new estimator functions to reproduce every result table of the paper version.

Module: M8 (PLAN.md section 0.2, item 4, "reproduce first, then fix"; CODE_PLAN.md section 9, step 2)

1. What it does
    Reads the paper-version PSM dataset (psm_dataset.npz from git 23fea07). It then uses
    the newly written functions in estimators.py / regression.py to recompute every result
    table that the four paper-version scripts produced, in the same order those scripts used:
        s13  main effects, balance, common support, ablation, number of strata, 48 hours,
             trimming, windows, E-values, pseudo start times (the paper-version Rosenbaum
             bounds are no longer computed);
        s14  PSM within subgroups (Table 5) and the number-of-strata sensitivity for subgroups;
        a2   interaction-term tests (the script in the 2026-5-7 folder);
        s15  mixed models for reply features (Tables 6, 7 and the appendix tables).
    It then compares each table, cell by cell, with the paper-version output table saved in
    git 23fea07, and writes a comparison report.

2. Why it is designed this way
    The rewritten code must first show that it "gets the same numbers as the paper version on
    the same data". Only then can each later change in the numbers be credited to a specific
    fix, and not to some difference introduced during the rewrite.

    Because we require a cell-by-cell match:
        - Random numbers: each paper-version script uses its own RandomState(42), and draws are
          used up in the same call order as the original script (for s13: ablation, then main
          results, then number of strata, then 48h, then trimming, then windows, then pseudo
          start times);
        - Propensity scores: s13 and s14 use max_iter=2000 and random_state=42;
          a2 uses max_iter=1000, sets no random_state, and clips scores to [0.01, 0.99];
        - Balance checks: reproduced with the paper-version weighting (weight each stratum by
          its total size). Results with ATT weighting (I-35) are also given, for the fixed version;
        - E-values: both old paper-version methods and the fixed method are given (I-19, I-36).

3. Inputs
    config.EXTERNAL["paper_npz"]: the paper-version PSM dataset;
    the s15 part also needs paper_reply_features, paper_s11 (row order), and paper_s7 (mentor names);
    paper-version output tables: read with `git show 23fea07:<path>`, so that we compare against
    the paper version itself.

4. Outputs (results/m8_legacy/)
    Result tables with the same names as in the paper version (for example B_main_results.csv,
    subgroup_att.csv);
    evalues.csv and balance_att_weighting.csv (fixed-version definitions);
    parity_report.md and parity_summary.csv: the results of the table-by-table comparison.

5. Key parameters
    --parts (which parts to run), --s15-dvs (main outcome only, or all 6),
    config.N_BOOT (bootstrap draws), config.N_STRATA (number of strata).
    Processing steps:
    run_s13() -> run_s14() -> run_a2() -> run_s15() -> compare_all() -> write_report()

6. Possible errors and how they are handled
    A table does not match: no error is raised. Instead the report lists the cells with the
    largest differences, and a person decides the cause (for example, the paper-version output
    table itself came from an earlier run).

7. How to run
    python -m pipeline.m8_analysis.legacy_parity              # s13, s14, a2, and the s15 main result (about 10 to 20 minutes)
    python -m pipeline.m8_analysis.legacy_parity --parts s13  # run only one part
    python -m pipeline.m8_analysis.legacy_parity --s15-dvs all  # run s15 for all 6 outcome variables (slow)
    python run_all.py --legacy

8. How to check
    results/m8_legacy/parity_report.md; tests/test_m8.py.
"""
from __future__ import annotations

import io as _io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from pipeline.common.io import read_jsonl, write_table
from pipeline.common.manifest import check_sources, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis import regression as reg

OUT = config.RESULTS / "m8_legacy"
log = None  # set by main()

# Outcome variables: (name in the paper-version output table, key in the npz). Same order as DIMS in paper-version s13.
DIMS_S13 = [
    ("1+_mainspace_edit_14d", "primary"),
    ("n_mainspace_edits_14d", "n_mainspace_edits_14d"),
    ("active_days_30d", "active_days_30d"),
    ("unique_namespace_14d", "unique_ns"),
    ("reverted_any_14d", "reverted_any"),
    ("2+_active_days_14d", "cross_day_any_14d"),
]
DIMS_S14 = [
    ("1+_mainspace_14d", "primary"), ("n_edits_14d", "n_mainspace_edits_14d"),
    ("active_days_30d", "active_days_30d"), ("unique_ns_14d", "unique_ns"),
    ("reverted_any_14d", "reverted_any"), ("2+_active_days_14d", "cross_day_any_14d"),
]
DIMS_A2 = DIMS_S13[:5]


# ════════════════════════════════════════════════════════════════════════════
# Reading
# ════════════════════════════════════════════════════════════════════════════


def load_npz() -> dict:
    """Read the paper-version PSM dataset and return a plain dict (arrays are loaded into memory)."""
    path = check_sources(["paper_npz"])["paper_npz"]
    with np.load(path, allow_pickle=True) as z:
        D = {k: z[k] for k in z.files}
    log.info("loaded %s: N=%d, treated=%d", path.name, len(D["y_treat"]), int(D["y_treat"].sum()))
    return D


def legacy_table(relpath: str, repo: Path | None = None) -> pd.DataFrame | None:
    """Read an output table from the paper-version commit with git show. Return None if the file does not exist.

    Args:
        relpath: path inside the repo, for example "data/s13/tables/B_main_results.csv".
        repo: repo folder, default config.PAPER_REPO. The a2 tables are also in this repo.
    """
    repo = repo or config.PAPER_REPO
    r = subprocess.run(["git", "-C", str(repo), "show", f"{config.PAPER_COMMIT}:{relpath}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return pd.read_csv(_io.StringIO(r.stdout))


def blocks(D: dict) -> dict[str, np.ndarray]:
    """Covariate blocks of the paper-version npz."""
    return {k: D[f"X_{k}"] for k in ("E", "Qtext", "Qpersp", "Qtype", "emb20", "temporal", "M", "emb_full")}


def covariate_names(D: dict) -> list[str]:
    """Names of the 164 covariates, in the same order as X_main (same as ALL_COLS in the paper version)."""
    return (list(D["E_cols"]) + list(D["Qtext_cols"]) + list(D["Qpersp_cols"]) +
            list(D["Qtype_cols"]) + [f"emb_pc{i}" for i in range(20)] + list(D["temporal_cols"]))


# ════════════════════════════════════════════════════════════════════════════
# s13: main effects and robustness
# ════════════════════════════════════════════════════════════════════════════


def _boot(y, d, st, clusters, rng, B=None):
    """Run the bootstrap and return a percentile confidence interval. B defaults to config.N_BOOT (read at call time). When debugging, B is capped at config.N_BOOT."""
    B = config.N_BOOT if B is None else min(B, config.N_BOOT)
    return est.percentile_ci(est.bootstrap_att(y, d, st, clusters, rng, B))


def run_s13(D: dict) -> dict[str, pd.DataFrame]:
    """Recompute every result table in the same order as the paper-version s13_psm_analysis.py.

    Random numbers: one RandomState(42), used in this order: A7 ablation (16 x 500), B main results
    (6 outcomes x (500 mentee + 500 mentor)), C1 number of strata (6 x 3 x 500), C2 48 hours,
    C3 trimming (4 x 500), C4 windows (7 x 500), C7 pseudo start times (5 x 500). The order is exactly
    the same as in the paper version.
    """
    rng = np.random.RandomState(42)
    B = blocks(D)
    d = D["y_treat"].astype(float)
    d48 = D["y_treat_48h"].astype(float)
    names = covariate_names(D)
    X = np.hstack([B["E"], B["Qtext"], B["Qpersp"], B["Qtype"], B["emb20"], B["temporal"]])
    mentees = est.ClusterIndex(D["mentee_ids"])
    mentors = est.ClusterIndex(D["mentor_ids"])
    out: dict[str, pd.DataFrame] = {}

    # A1 propensity scores and strata
    ps = est.fit_propensity(X, d)
    st = est.make_strata(ps.e_hat, config.N_STRATA)
    log.info("[s13] AUC = %.4f", ps.auc)

    # A3 per-stratum details
    rows = []
    for k in range(st.max() + 1):
        m = st == k
        rows.append({"Stratum": k, "N": int(m.sum()), "Treated": int(d[m].sum()),
                     "Control": int((1 - d[m]).sum()), "T_pct": d[m].mean() * 100,
                     "PS_lo": ps.e_hat[m].min(), "PS_hi": ps.e_hat[m].max()})
    out["A3_strata_diagnostics"] = pd.DataFrame(rows)

    # A4 balance: paper-version weighting (by total stratum size), plus the fixed-version ATT weighting
    out["A4_covariate_balance"] = est.smd_table(X, d, st, names, weighting="legacy")
    out["balance_att_weighting"] = est.smd_table(X, d, st, names, weighting="att")

    # A5 common support
    out["A5_common_support"] = pd.DataFrame([est.common_support(ps.e_hat, d)])

    # A7 ablation (16 specifications; these use up the first batch of random numbers)
    specs = {
        "Full model (E+Qtext+Qpersp+Qtype+Emb+T)": ["E", "Qtext", "Qpersp", "Qtype", "emb20", "temporal"],
        "Full – Qtype (no LLM annotation)": ["E", "Qtext", "Qpersp", "emb20", "temporal"],
        "Full – temporal": ["E", "Qtext", "Qpersp", "Qtype", "emb20"],
        "Full – embeddings": ["E", "Qtext", "Qpersp", "Qtype", "temporal"],
        "Full – Qtext": ["E", "Qpersp", "Qtype", "emb20", "temporal"],
        "Full – Qpersp": ["E", "Qtext", "Qtype", "emb20", "temporal"],
        "Full + full emb (1024d)": ["E", "Qtext", "Qpersp", "Qtype", "emb_full", "temporal"],
        "Edit history only": ["E"],
        "E + embeddings": ["E", "emb20"],
        "E + Qtext": ["E", "Qtext"],
        "E + Qpersp": ["E", "Qpersp"],
        "E + Qtype": ["E", "Qtype"],
        "Question features only (Qtext+Qpersp+Qtype)": ["Qtext", "Qpersp", "Qtype"],
        "Embeddings only": ["emb20"],
        "Qtype only": ["Qtype"],
        "Full + mentor features": ["E", "Qtext", "Qpersp", "Qtype", "emb20", "temporal", "M"],
    }
    y1 = D["oc_primary"].astype(float)
    rows = []
    for name, keys in specs.items():
        Xa = np.hstack([B[k] for k in keys])
        psa = est.fit_propensity(Xa, d)
        sta = est.make_strata(psa.e_hat, config.N_STRATA)
        att = est.att_stratified(y1, d, sta).att
        lo, hi = _boot(y1, d, sta, mentees, rng)
        smd = est.smd_table(Xa, d, sta, [f"x{j}" for j in range(Xa.shape[1])], weighting="legacy")
        rows.append({"Model": name, "p": Xa.shape[1], "AUC": psa.auc, "ATT": att, "CI_lo": lo, "CI_hi": hi,
                     "Sig": "*" if est.is_significant(lo, hi) else "", "SMD_gt_0.1": int((smd["strat"] > 0.1).sum())})
        log.info("[s13] ablation %-45s AUC=%.4f ATT=%+.4f", name, psa.auc, att)
    out["A7_feature_ablation"] = pd.DataFrame(rows)

    # B main results (mentee bootstrap and mentor bootstrap alternate, as in the paper version)
    Xs = StandardScaler().fit_transform(X)
    rows, mrows = [], []
    for nm, key in DIMS_S13:
        y = D[f"oc_{key}"].astype(float)
        att = est.att_stratified(y, d, st).att
        dr = est.dr_stratified_legacy(y, d, st, Xs)
        lo, hi = _boot(y, d, st, mentees, rng)
        mlo, mhi = _boot(y, d, st, mentors, rng)
        sd = y[d == 0].std()
        rows.append({"Outcome": nm, "Naive": est.naive_diff(y, d), "ATT": att, "DR": dr, "CI_lo": lo, "CI_hi": hi,
                     "Cohen_d": att / sd if sd > 0 else 0.0, "d_CI_lo": lo / sd, "d_CI_hi": hi / sd,
                     "Sig": "*" if est.is_significant(lo, hi) else "",
                     "variant": "primary" if key == "primary" else "supplementary"})
        mrows.append({"Outcome": nm, "ATT": att, "Mentee_CI_lo": lo, "Mentee_CI_hi": hi,
                      "Mentee_Sig": "*" if est.is_significant(lo, hi) else "",
                      "Mentor_CI_lo": mlo, "Mentor_CI_hi": mhi,
                      "Mentor_Sig": "*" if est.is_significant(mlo, mhi) else ""})
    out["B_main_results"] = pd.DataFrame(rows)
    out["B_mentor_boot"] = pd.DataFrame(mrows)
    main_att, main_lo = rows[0]["ATT"], rows[0]["CI_lo"]

    # C1 number of strata
    rows = []
    for nm, key in DIMS_S13:
        y = D[f"oc_{key}"].astype(float)
        for K in (5, 10, 20):
            stk = est.make_strata(ps.e_hat, K)
            att = est.att_stratified(y, d, stk).att
            lo, hi = _boot(y, d, stk, mentees, rng)
            rows.append({"Outcome": nm, "Strata": K, "ATT": att, "CI_lo": lo, "CI_hi": hi,
                         "Sig": "*" if est.is_significant(lo, hi) else ""})
    out["C1_strata_sensitivity"] = pd.DataFrame(rows)

    # C2 48-hour definition
    ps48 = est.fit_propensity(X, d48)
    st48 = est.make_strata(ps48.e_hat, config.N_STRATA)
    att48 = est.att_stratified(y1, d48, st48).att
    lo48, hi48 = _boot(y1, d48, st48, mentees, rng)

    # C3 trimming
    rows = []
    for lo_t, hi_t in ((0.01, 0.99), (0.05, 0.95), (0.10, 0.90), (0.15, 0.85)):
        tm = (ps.e_hat >= lo_t) & (ps.e_hat <= hi_t)
        stt = est.make_strata(ps.e_hat[tm], config.N_STRATA)
        att = est.att_stratified(y1[tm], d[tm], stt).att
        lo, hi = _boot(y1[tm], d[tm], stt, est.ClusterIndex(D["mentee_ids"][tm]), rng)
        rows.append({"Range": f"[{lo_t},{hi_t}]",          # same format as the paper version, for example "[0.1,0.9]"
                     "Trimmed": int((~tm).sum()), "N_remain": int(tm.sum()), "ATT": att, "CI_lo": lo, "CI_hi": hi,
                     "Sig": "*" if est.is_significant(lo, hi) else ""})
    out["C3_ps_trimming"] = pd.DataFrame(rows)

    # C4 windows (the paper-version npz did not save a "full follow-up" flag, so all observations are used, as in the paper version)
    rows = []
    for w in config.OUTCOME_WINDOWS_DAYS:
        y = D[f"oc_mainspace_{w}d"].astype(float)
        att = est.att_stratified(y, d, st).att
        lo, hi = _boot(y, d, st, mentees, rng)
        rows.append({"window": w, "N": len(y), "ATT": att, "CI_lo": lo, "CI_hi": hi,
                     "Sig": "*" if est.is_significant(lo, hi) else ""})
    out["C4_window_sensitivity"] = pd.DataFrame(rows)

    # Three ways to compute the E-value (decided 2026-10-07: keep only E-values for sensitivity analysis, no longer compute Rosenbaum bounds)
    p1 = y1[d == 1].mean()
    p0 = y1[d == 0].mean()
    ev_fixed = est.e_value_att(p1, main_att, main_lo)
    ev_old = est.e_value_legacy_one_plus_att(main_att, main_lo)
    ev_cur = est.e_value_s13_current(p1, p0, p0, main_lo)
    out["evalues"] = pd.DataFrame([
        {"method": "Used in the paper (version 430f20b: RR = 1 + ATT, wrong)", **ev_old},
        {"method": "23fea07 s13 version (point estimate uses unadjusted p1/p0, lower bound uses (p0+τ_lo)/p0)", **ev_cur},
        {"method": "Fixed version (RR = p1 / (p1 - τ), ATT definition)", "e_point": ev_fixed["e_point"],
         "e_ci": ev_fixed["e_ci"], "rr": ev_fixed["rr"], "rr_lo": ev_fixed["rr_lo"]},
    ])

    # C7 pseudo start times
    rows = [{"Strategy": "Random draw (main)", "ATT": main_att, "CI_lo": rows_main_lo(out),
             "CI_hi": out["B_main_results"].iloc[0]["CI_hi"]}]
    for key, label in (("median", "Median lag"), ("p25", "25th pctl lag"), ("p75", "75th pctl lag"),
                       ("zero", "Zero lag (immediate)"), ("multi", "Multi-draw avg")):
        y = D[f"oc_primary__{key}"].astype(float)
        att = est.att_stratified(y, d, st).att
        lo, hi = _boot(y, d, st, mentees, rng)
        rows.append({"Strategy": label, "ATT": att, "CI_lo": lo, "CI_hi": hi})
    c7 = pd.DataFrame(rows)
    c7["Sig"] = ["*" if est.is_significant(a, b) else "" for a, b in zip(c7.CI_lo, c7.CI_hi)]
    out["C7_pseudo_time"] = c7

    # C6 summary (same components as the paper version)
    rob = [{"Spec": "Main (K=10)", "ATT": main_att, "CI_lo": main_lo, "CI_hi": out["B_main_results"].iloc[0]["CI_hi"]},
           {"Spec": "D=48h", "ATT": att48, "CI_lo": lo48, "CI_hi": hi48}]
    rob += [{"Spec": f"Trim {r.Range}", "ATT": r.ATT, "CI_lo": r.CI_lo, "CI_hi": r.CI_hi}
            for r in out["C3_ps_trimming"].itertuples()]
    c1p = out["C1_strata_sensitivity"].query("Outcome == '1+_mainspace_edit_14d'")
    rob += [{"Spec": f"K={int(r.Strata)}", "ATT": r.ATT, "CI_lo": r.CI_lo, "CI_hi": r.CI_hi} for r in c1p.itertuples()]
    rob += [{"Spec": f"Pseudo: {r.Strategy}", "ATT": r.ATT, "CI_lo": r.CI_lo, "CI_hi": r.CI_hi}
            for r in c7.iloc[1:].itertuples()]
    c6 = pd.DataFrame(rob)
    c6["Sig"] = ["*" if est.is_significant(a, b) else "" for a, b in zip(c6.CI_lo, c6.CI_hi)]
    out["C6_robustness_summary"] = c6

    # Diagnostic numbers that the paper text cites directly but that have no table of their own
    a4 = out["A4_covariate_balance"]
    out["s13_text_numbers"] = pd.DataFrame([{
        "auc": ps.auc, "n": len(d), "n_treated": int(d.sum()), "n_control": int((1 - d).sum()),
        "treated_pct": d.mean() * 100, "control_baseline_primary": p0,
        "relative_effect_primary_pct": main_att / p0 * 100,
        "relative_effect_2plus_days_pct": out["B_main_results"].iloc[5]["ATT"] /
        D["oc_cross_day_any_14d"][d == 0].mean() * 100,
        "n_raw_smd_gt_0.1": int((a4.raw > 0.1).sum()), "max_raw_smd": a4.raw.max(),
        "max_strat_smd_legacy": a4.strat.max(),
        "max_strat_smd_att": out["balance_att_weighting"].strat.max(),
        "n_outside_common_support": out["A5_common_support"].n_outside.iloc[0],
    }])
    return out


def rows_main_lo(out: dict) -> float:
    """Lower bound of the main-result confidence interval (the first row of C7 matches the paper version, so take it straight from table B)."""
    return float(out["B_main_results"].iloc[0]["CI_lo"])




# ════════════════════════════════════════════════════════════════════════════
# s14: PSM within subgroups
# ════════════════════════════════════════════════════════════════════════════


def subgroups(D: dict) -> list[tuple[str, np.ndarray]]:
    """The 8 subgroups of paper-version s14 (do not change the order: it decides the order in which random numbers are used)."""
    Q = D["X_Qtype"]
    q_sub, q_ref, q_apr, q_nor, q_own = (Q[:, j] for j in range(5))
    n = len(Q)
    return [
        ("Full sample", np.ones(n, bool)),
        ("Technical", (q_sub == 1) & (q_ref == 0) & (q_apr == 0) & (q_nor == 0)),
        ("Referent (Q2)", q_ref == 1),
        ("Appraisal (Q3)", q_apr == 1),
        ("Normative (Q4)", q_nor == 1),
        ("Non-substantive", q_sub == 0),
        ("Own work (Q5=Y)", q_own == 1),
        ("No own work (Q5=N)", q_own == 0),
    ]


def run_s14(D: dict) -> dict[str, pd.DataFrame]:
    """Recompute Part 1 and Part 2 of the paper-version s14_1_heterogeneous_effects.py.

    Part 1 (Table 5): re-estimate the propensity score within each subgroup (covariates without Q1 to Q5),
        K = min(5, max(3, n_c // 30)), 500 bootstrap draws per outcome.
    Part 2 (appendix Fig A3): K in {3, 5, 7, 10}, 200 bootstrap draws per setting;
        skipped when n_c < 15K (as in the paper version).
    Random numbers: one RandomState(42), Part 1 first, then Part 2.
    """
    rng = np.random.RandomState(42)
    B = blocks(D)
    d_all = D["y_treat"].astype(float)
    Xsub = np.hstack([B["E"], B["Qtext"], B["Qpersp"], B["emb20"], B["temporal"]])
    out_rows, diag, sens = [], [], []
    sgs = subgroups(D)
    for name, mask in sgs:
        idx = np.where(mask)[0]
        d = d_all[idx]
        nt, nc = int(d.sum()), int(len(d) - d.sum())
        if nc < 50 or nt < 50:
            continue
        K = min(config.N_STRATA_SUBGROUP, max(3, nc // 30))
        ps = est.fit_propensity(Xsub[idx], d)
        st = est.make_strata(ps.e_hat, K)
        cl = est.ClusterIndex(D["mentee_ids"][idx])
        smd = est.smd_table(Xsub[idx], d, st, [f"x{j}" for j in range(Xsub.shape[1])], weighting="legacy")
        n_bad = int((smd["strat"] > 0.1).sum())   # same as the paper version: number of variables with |SMD| > 0.1 after stratification
        diag.append({"Subgroup": name, "N": len(idx), "Treated": nt, "Control": nc,
                     "T_pct": round(nt / len(idx) * 100, 1), "AUC": round(ps.auc, 4),
                     "K": int(st.max() + 1), "SMD_bad": n_bad})
        for nm, key in DIMS_S14:
            y = D[f"oc_{key}"][idx].astype(float)
            att = est.att_stratified(y, d, st).att
            lo, hi = _boot(y, d, st, cl, rng)
            sd = y[d == 0].std()
            out_rows.append({"Subgroup": name, "N": len(idx), "T": nt, "C": nc, "Outcome": nm,
                             "Ctrl_mean": y[d == 0].mean(), "Treat_mean": y[d == 1].mean(),
                             "ATT": att, "CI_lo": lo, "CI_hi": hi, "Cohen_d": att / sd if sd > 0 else 0.0,
                             "Sig": "*" if est.is_significant(lo, hi) else ""})
        log.info("[s14] %-20s N=%d K=%d ATT(primary)=%+.4f", name, len(idx), K, out_rows[-6]["ATT"])
    # Part 2
    for name, mask in sgs:
        idx = np.where(mask)[0]
        d = d_all[idx]
        nc = int(len(d) - d.sum())
        if nc < 50 or d.sum() < 50:
            continue
        cl = est.ClusterIndex(D["mentee_ids"][idx])
        for nm, key in DIMS_S14:
            dv_label = {"1+_mainspace_14d": "primary"}.get(nm, nm)
            y = D[f"oc_{key}"][idx].astype(float)
            for K in config.SUBGROUP_STRATA_SENSITIVITY:
                if nc < K * 15:
                    continue
                ps = est.fit_propensity(Xsub[idx], d)
                st = est.make_strata(ps.e_hat, K)
                att = est.att_stratified(y, d, st).att
                lo, hi = _boot(y, d, st, cl, rng, B=200)
                sens.append({"Subgroup": name, "DV": dv_label, "K": K, "ATT": att, "CI_lo": lo, "CI_hi": hi,
                             "Sig": "*" if est.is_significant(lo, hi) else ""})
    return {"subgroup_att": pd.DataFrame(out_rows), "subgroup_diagnostics": pd.DataFrame(diag),
            "sensitivity_strata": pd.DataFrame(sens)}


# ════════════════════════════════════════════════════════════════════════════
# a2: interaction-term tests
# ════════════════════════════════════════════════════════════════════════════


def run_a2(D: dict) -> dict[str, pd.DataFrame]:
    """Recompute the interaction term δ as in 2026-5-7/a2_interaction_regression.py (PLAN M8.8, item 2).

    Propensity scores: max_iter=1000, no random_state, clipped to [0.01, 0.99], K=10 (as in a2).
    Random numbers: one RandomState(42), used in the order "moderator (outer loop) x outcome (inner loop)".
    """
    rng = np.random.RandomState(42)
    B = blocks(D)
    d = D["y_treat"].astype(float)
    X = np.hstack([B["E"], B["Qtext"], B["Qpersp"], B["Qtype"], B["emb20"], B["temporal"]])
    ps = est.fit_propensity(X, d, max_iter=1000, lr_random_state=None, clip=(0.01, 0.99))
    st = est.make_strata(ps.e_hat, config.N_STRATA)
    Xr = np.hstack([B["Qtext"], B["Qpersp"], B["Qtype"], B["emb20"]])
    rps = est.fit_propensity(Xr, d, max_iter=1000, lr_random_state=None).e_hat
    Q = D["X_Qtype"]
    q_sub, q_ref, q_apr, q_nor, q_own = (Q[:, j] for j in range(5))
    breadth = q_ref + q_apr + q_nor
    tech = ((q_sub == 1) & (q_ref == 0) & (q_apr == 0) & (q_nor == 0)).astype(float)
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731  as in a2, uses the population standard deviation
    mods = [("M1: Substantive (Q1)", q_sub, "binary"),
            ("M2: Info-seeking breadth (Q2+Q3+Q4)", z(breadth), "continuous (z)"),
            ("M3: Own work (Q5)", q_own, "binary"),
            ("M4: Reply propensity", z(rps), "continuous (z)"),
            ("M5: Referent (Q2)", q_ref, "binary"),
            ("M6: Appraisal (Q3)", q_apr, "binary"),
            ("M7: Normative (Q4)", q_nor, "binary"),
            ("M8: Pure technical", tech, "binary")]
    cl = est.ClusterIndex(D["mentee_ids"])
    rows = []
    stf = st.astype(float)
    # Parallel run: first advance rng in serial order and record each task's starting state, so results match a serial run bit for bit
    tasks, meta = [], []
    for mname, mv, mtype in mods:
        for nm, key in DIMS_A2:
            tasks.append((D[f"oc_{key}"].astype(float), d, mv, stf, cl, rng.get_state(), config.N_BOOT))
            meta.append((mname, mtype, nm))
            est.advance_rng(rng, cl, config.N_BOOT)
    if config.N_JOBS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(config.N_JOBS) as ex:
            results = list(ex.map(est.interaction_task, tasks))
    else:
        results = [est.interaction_task(t) for t in tasks]
    for (mname, mtype, nm), (b, draws) in zip(meta, results):
        lo, hi = (np.percentile(draws, 2.5), np.percentile(draws, 97.5)) if len(draws) > 50 else (np.nan, np.nan)
        rows.append({"moderator": mname, "mod_type": mtype, "DV": nm, "beta_D": b[1], "beta_DxM": b[3],
                     "boot_mean": draws.mean(), "boot_SE": draws.std(), "CI_lo": lo, "CI_hi": hi,
                     "sig": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
        if nm == DIMS_A2[-1][0]:
            log.info("[a2] %-40s δ(primary)=%+.4f", mname, rows[-len(DIMS_A2)]["beta_DxM"])
    return {"a2_interaction_results": pd.DataFrame(rows)}


# ════════════════════════════════════════════════════════════════════════════
# s15: mixed models for reply features
# ════════════════════════════════════════════════════════════════════════════

REPLY_FEATURES = [
    ("r_words", "Reply word count"), ("r_pronoun_you_rate", "\"You\" rate (per word)"),
    ("r_pronoun_i_rate", "\"I\" rate (per word)"), ("r_pronoun_we_rate", "\"We\" rate (per word)"),
    ("r_vader_compound", "VADER compound sentiment"), ("r_tb_subjectivity", "TextBlob subjectivity"),
    ("r_flesch_kincaid", "Flesch-Kincaid grade"), ("r_n_imperatives", "Imperative sentences"),
    ("r_n_resources", "Resource links (total)"), ("r_n_wikilink", "Wikilinks"),
    ("r_n_policy", "Policy references"), ("r_n_help", "Help page references"),
    ("r_reply_lag_hours", "Reply lag (hours)"), ("r_reply_q_word_ratio", "Reply/question word ratio"),
    ("r_has_greeting", "Has greeting"), ("r_polite_gratitude", "Gratitude expression"),
    ("r_polite_has_positive", "Positive language"), ("r_polite_deference", "Deference"),
    ("r_has_list", "Has list"), ("r_n_question_marks", "Question marks in reply"),
    ("r_qr_cosine_sim", "Q-R semantic similarity"),
]
DVS_S15 = [("primary", "1+ mainspace edit (14d)"), ("n_mainspace_edits_14d", "N mainspace edits (14d)"),
           ("active_days_30d", "Active days (30d)"), ("unique_ns", "Unique namespaces (14d)"),
           ("reverted_any", "Reverted any (14d)"), ("cross_day_any_14d", "2+ active days (14d)")]


def run_s15(D: dict, dvs: str = "primary") -> dict[str, pd.DataFrame]:
    """Recompute Part 1 and Part 3 of the paper-version s15_2_reply_association.py.

    Preprocessing is exactly the same as in the paper version:
        - Row alignment: conversation_id in reply_features.csv -> s11 row number -> PSM row number (treated group only);
        - NaN and infinity are replaced with 0; reply lag is capped at the 99th percentile;
        - Reply features are standardized on the whole treated group (not re-standardized within subgroups); same for controls;
        - Features with VIF > 10 are dropped;
        - Mentor names come from paper-version s7.
    Args:
        dvs: "primary" runs only the main outcome (paper Tables 6, 7 and the appendix tables use only the main outcome); "all" runs all 6.
    """
    src = check_sources(["paper_reply_features", "paper_s11", "paper_s7"])
    B = blocks(D)
    d = D["y_treat"]
    Xall = np.hstack([B["E"], B["Qtext"], B["Qpersp"], B["Qtype"], B["emb20"], B["temporal"]])
    s11_cids = [r["conversation_id"] for r in read_jsonl(src["paper_s11"])]
    cid_mentor = {r["conversation_id"]: r.get("mentor", "unknown") for r in read_jsonl(src["paper_s7"])}
    treated_map = {s11_cids[i]: i for i in np.where(d == 1)[0]}
    rf = pd.read_csv(src["paper_reply_features"])
    cols = [c for c, _ in REPLY_FEATURES]
    names = [n for _, n in REPLY_FEATURES]
    pos = [(ri, treated_map[c]) for ri, c in enumerate(rf["conversation_id"]) if c in treated_map]
    rfi = np.array([p[0] for p in pos])
    psi = np.array([p[1] for p in pos])
    mentors = np.array([cid_mentor.get(rf["conversation_id"].iloc[i], "unknown") for i in rfi])
    Xc = StandardScaler().fit_transform(Xall[psi])
    Xr = np.nan_to_num(rf.iloc[rfi][cols].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    j = cols.index("r_reply_lag_hours")
    Xr[:, j] = np.clip(Xr[:, j], 0, np.percentile(Xr[:, j], 99))
    Xr = StandardScaler().fit_transform(Xr)
    Xr, kept, vif = reg.vif_screen(Xr, names)
    log.info("[s15] matched %d treated rows, %d mentors; VIF dropped: %s", len(psi), len(set(mentors)),
             list(vif.loc[vif.dropped, "feature"]))
    qt = rf.iloc[rfi][["Q0", "Q2", "Q3", "Q4", "Q5"]].values
    q_sub, q_ref, q_apr, q_nor, q_own = (qt[:, k] == "Y" for k in range(5))
    sgs = [("Technical", q_sub & ~q_ref & ~q_apr & ~q_nor), ("Referent (Q2)", q_ref),
           ("Normative (Q4)", q_nor), ("Non-substantive", qt[:, 0] == "N"),
           ("Own work (Q5=Y)", q_own), ("No own work (Q5=N)", ~q_own)]
    dv_list = DVS_S15 if dvs == "all" else DVS_S15[:1]
    full, bysg = [], []
    for key, dvname in dv_list:
        y = D[f"oc_{key}"][psi].astype(float)
        t0 = time.time()
        res = reg.mixedlm_with_controls(y, Xr, Xc, mentors)
        log.info("[s15] full %-26s %s in %.0fs", dvname, res.method, time.time() - t0)
        for k, nm in enumerate(kept):
            full.append({"DV": dvname, "DV_key": key, "Feature": nm, "Beta_std": res.beta[k], "SE": res.se[k],
                         "z": res.z[k], "p": res.p[k], "Sig": reg.sig_stars(res.p[k]), "ICC": res.icc,
                         "method": res.method, "Subgroup": "Full treated"})
        for sg, m in sgs:
            if m.sum() < 200:
                continue
            t0 = time.time()
            r = reg.mixedlm_with_controls(y[m], Xr[m], Xc[m], mentors[m])
            log.info("[s15] %-18s %-26s n=%d %s in %.0fs", sg, dvname, m.sum(), r.method, time.time() - t0)
            for k, nm in enumerate(kept):
                bysg.append({"Subgroup": sg, "N": int(m.sum()), "DV": dvname, "DV_key": key, "Feature": nm,
                             "Beta_std": r.beta[k], "SE": r.se[k], "z": r.z[k], "p": r.p[k],
                             "Sig": reg.sig_stars(r.p[k]), "ICC": r.icc, "method": r.method})
    return {"association_results": pd.DataFrame(full), "association_by_subgroup": pd.DataFrame(bysg),
            "s15_vif": vif}


# ════════════════════════════════════════════════════════════════════════════
# Comparison with the paper-version output tables
# ════════════════════════════════════════════════════════════════════════════

#: (new table name, paper-version path, key columns, numeric columns to compare, tolerance)
#: Paper-version s14 and s15 tables round numbers to 4 decimals, so the tolerance is 6e-5.
COMPARISONS = [
    ("A3_strata_diagnostics", "data/s13/tables/A3_strata_diagnostics.csv", ["Stratum"],
     ["N", "Treated", "Control", "PS_lo", "PS_hi"], 1e-9),
    ("A4_covariate_balance", "data/s13/tables/A4_covariate_balance.csv", ["var"], ["raw", "strat"], 1e-9),
    ("A7_feature_ablation", "data/s13/tables/A7_feature_ablation.csv", ["Model"],
     ["p", "AUC", "ATT", "CI_lo", "CI_hi", "SMD_gt_0.1"], 1e-9),
    ("B_main_results", "data/s13/tables/B_main_results.csv", ["Outcome"],
     ["Naive", "ATT", "DR", "CI_lo", "CI_hi", "Cohen_d", "d_CI_lo", "d_CI_hi"], 1e-9),
    ("B_mentor_boot", "data/s13/tables/B_mentor_boot.csv", ["Outcome"],
     ["ATT", "Mentee_CI_lo", "Mentee_CI_hi", "Mentor_CI_lo", "Mentor_CI_hi"], 1e-9),
    ("C1_strata_sensitivity", "data/s13/tables/C1_strata_sensitivity.csv", ["Outcome", "Strata"],
     ["ATT", "CI_lo", "CI_hi"], 1e-9),
    ("C3_ps_trimming", "data/s13/tables/C3_ps_trimming.csv", ["Range"],
     ["Trimmed", "N_remain", "ATT", "CI_lo", "CI_hi"], 1e-9),
    ("C4_window_sensitivity", "data/s13/tables/C4_window_sensitivity.csv", ["window"],
     ["N", "ATT", "CI_lo", "CI_hi"], 1e-9),
    ("C6_robustness_summary", "data/s13/tables/C6_robustness_summary.csv", ["Spec"],
     ["ATT", "CI_lo", "CI_hi"], 1e-9),
    ("C7_pseudo_time", "data/s13/tables/C7_pseudo_time.csv", ["Strategy"], ["ATT", "CI_lo", "CI_hi"], 1e-9),
    ("subgroup_att", "data/s14/tables/subgroup_att.csv", ["Subgroup", "Outcome"],
     ["N", "T", "C", "Ctrl_mean", "Treat_mean", "ATT", "CI_lo", "CI_hi", "Cohen_d"], 6e-5),
    ("subgroup_diagnostics", "data/s14/tables/subgroup_diagnostics.csv", ["Subgroup"],
     ["N", "Treated", "Control", "AUC", "K", "SMD_bad"], 6e-5),
    ("sensitivity_strata", "data/s14/tables/sensitivity_strata.csv", ["Subgroup", "DV", "K"],
     ["ATT", "CI_lo", "CI_hi"], 6e-5),
    ("a2_interaction_results", "data/a2/tables/a2_interaction_results.csv", ["moderator", "DV"],
     ["beta_D", "beta_DxM", "boot_mean", "boot_SE", "CI_lo", "CI_hi"], 1e-9),
    ("association_results", "data/s15/association_results.csv", ["DV_key", "Feature"],
     ["Beta_std", "SE", "p"], 6e-5),
    ("association_by_subgroup", "data/s15/association_by_subgroup.csv", ["Subgroup", "DV_key", "Feature"],
     ["N", "Beta_std", "SE", "p"], 6e-5),
]


#: Known differences that are not code problems. They go into the report to help a person check.
KNOWN_DIFFERENCES = {
    "A7_feature_ablation": "Only the \"Full + full emb (1024d)\" row has a tiny difference (AUC differs by about 1e-6). The logistic "
                           "regression with 1168 variables does not fully converge within max_iter=2000, so numbers "
                           "jitter slightly between runs. The other 15 rows match cell by cell.",
    "subgroup_att": "The paper-version subgroup table was made at 2026-05-15 08:56, before the last rebuild of the paper-version "
                    "npz (2026-05-15 23:42). The s13 tables (2026-05-16 00:25) were run after the rebuild, so they match "
                    "the new code cell by cell. The old npz was not kept, so this table cannot be reproduced cell by cell. "
                    "Supporting evidence: all 6 outcomes of the Referent subgroup match cell by cell; the main-result "
                    "point estimates differ by at most 0.0002.",
    "subgroup_diagnostics": "Same as subgroup_att (different data version).",
    "sensitivity_strata": "Same as subgroup_att (different data version).",
    "a2_interaction_results": "The a2 table came from an earlier npz (the script notes give a sample size of 35,260; it is now 35,381), so it cannot be reproduced cell by cell.",
    "association_by_subgroup": "Coefficients differ by at most 1.4e-4. The largest difference is in the smallest subgroup, Referent (n = 3,560). "
                               "This is a numerical difference in where the mixed-model optimizer (lbfgs, maxiter=200) "
                               "stops. The coefficients agree at the precision shown in the paper.",
}


def compare(new: pd.DataFrame, old: pd.DataFrame | None, keys: list[str], cols: list[str],
            tol: float) -> dict:
    """Compare the new table with the paper-version table, cell by cell.

    Returns:
        {status, n_rows_new, n_rows_old, n_matched, n_cells, n_bad_cells, max_abs_diff, worst}
        status: match (all within tolerance), differs, or missing_old (the paper version has no such table).
    """
    if old is None:
        return {"status": "missing_old"}
    new = new.copy()
    old = old.copy()
    for k in keys:
        new[k] = new[k].astype(str)
        old[k] = old[k].astype(str)
    m = new.merge(old, on=keys, how="outer", suffixes=("_new", "_old"), indicator=True)
    both = m[m["_merge"] == "both"]
    worst, max_diff, n_bad, n_cells = None, 0.0, 0, 0
    for c in cols:
        if f"{c}_new" not in both or f"{c}_old" not in both:
            continue
        a = pd.to_numeric(both[f"{c}_new"], errors="coerce")
        b = pd.to_numeric(both[f"{c}_old"], errors="coerce")
        diff = (a - b).abs()
        same_nan = a.isna() & b.isna()
        bad = (diff > tol) | (a.isna() ^ b.isna())
        n_cells += int((~same_nan).sum())
        n_bad += int(bad.sum())
        if diff.max(skipna=True) > max_diff:
            max_diff = float(diff.max(skipna=True))
            i = diff.idxmax()
            worst = {"column": c, **{k: both.loc[i, k] for k in keys},
                     "new": float(a.loc[i]), "old": float(b.loc[i])}
    return {"status": "match" if (n_bad == 0 and len(both) == len(old) == len(new)) else "differs",
            "n_rows_new": len(new), "n_rows_old": len(old), "n_matched": len(both),
            "n_cells": n_cells, "n_bad_cells": n_bad, "max_abs_diff": max_diff, "worst": worst}


def _md(df: pd.DataFrame, index: bool = False) -> str:
    """Turn a DataFrame into a markdown table. If tabulate is not installed, fall back to fixed-width text."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def _rel(p) -> str:
    """A path relative to the 0Wiki main folder (config.WIKI_BASE), so reports do not show local home folders."""
    try:
        return str(Path(p).resolve().relative_to(config.WIKI_BASE))
    except ValueError:
        return Path(p).name


def write_report(results: dict[str, pd.DataFrame], summary: list[dict]) -> Path:
    """Write parity_report.md: the table-by-table comparison, and where the fixed version differs from the paper version."""
    lines = ["# Stage A: paper-version reproduction report", "",
             f"- Data: `{_rel(config.EXTERNAL['paper_npz'])}` (paper-version commit `{config.PAPER_COMMIT}`)",
             f"- Generated: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC", "",
             "## Table-by-table comparison (new code vs paper-version output tables)", "",
             "| Table | Status | Rows (new/paper/matched) | Mismatched cells | Max diff | Cell with max diff |", "|---|---|---|---|---|---|"]
    notes: list[str] = []
    for s in summary:
        if s.get("status") == "missing_old":
            lines.append(f"| {s['table']} | paper version has no such table | | | | |")
            continue
        w = s.get("worst") or {}
        if s["status"] != "match" and s["table"] in KNOWN_DIFFERENCES:
            notes.append(f"- **{s['table']}**: {KNOWN_DIFFERENCES[s['table']]}")
        wtxt = ", ".join(f"{k}={v}" for k, v in w.items()) if w else ""
        lines.append(f"| {s['table']} | {'✅ match' if s['status'] == 'match' else '⚠️ differs'} | "
                     f"{s['n_rows_new']}/{s['n_rows_old']}/{s['n_matched']} | {s['n_bad_cells']}/{s['n_cells']} | "
                     f"{s['max_abs_diff']:.2e} | {wtxt} |")
    if notes:
        lines += ["", "## Known differences and their causes", ""] + notes
    if "evalues" in results:
        lines += ["", "## Three ways to compute the E-value (I-19, I-36)", "", _md(results["evalues"])]
    if "s13_text_numbers" in results:
        lines += ["", "## Diagnostic numbers cited in the paper text", "",
                  _md(results["s13_text_numbers"].T.rename(columns={0: "value"}), index=True)]
    path = OUT / "parity_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ════════════════════════════════════════════════════════════════════════════
# Main program
# ════════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    global log
    ap = base_parser("Stage A: reproduce the paper-version result tables with the new estimators.")
    ap.add_argument("--parts", default="s13,s14,a2,s15", help="comma-separated: s13,s14,a2,s15")
    ap.add_argument("--s15-dvs", default="primary", choices=["primary", "all"])
    args = ap.parse_args(argv)
    log = setup_logging("m8_legacy")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = set(args.parts.split(",")) - {"none", ""}   # --parts none: recompute nothing, only re-compare and rebuild the report

    D = load_npz() if parts else None
    results: dict[str, pd.DataFrame] = {}
    if "s13" in parts:
        results.update(run_s13(D))
    if "s14" in parts:
        results.update(run_s14(D))
    if "a2" in parts:
        results.update(run_a2(D))
    if "s15" in parts:
        results.update(run_s15(D, args.s15_dvs))

    for name, df in results.items():
        write_table(OUT / f"{name}.csv", df)

    # For tables not recomputed in this run, read the results left by the last run, so the report always covers every reproduced table
    for name, *_ in COMPARISONS:
        if name not in results and (OUT / f"{name}.csv").exists():
            results[name] = pd.read_csv(OUT / f"{name}.csv")
    for extra in ("evalues", "s13_text_numbers"):
        if extra not in results and (OUT / f"{extra}.csv").exists():
            results[extra] = pd.read_csv(OUT / f"{extra}.csv")

    summary = []
    for name, rel, keys, cols, tol in COMPARISONS:
        if name not in results:
            continue
        new = results[name]
        if name.startswith("association"):          # compare only the outcome variables actually run this time
            old = legacy_table(rel)
            old = old[old["DV_key"].isin(new["DV_key"].unique())] if old is not None else None
            if old is not None and "Subgroup" in keys:
                old = old[old["Subgroup"].isin(new["Subgroup"].unique())]
        else:
            old = legacy_table(rel)
        if name == "sensitivity_strata" and old is not None:
            old = old[old["Subgroup"].isin(new["Subgroup"].unique())]
        s = {"table": name, **compare(new, old, keys, cols, tol)}
        summary.append(s)
        log.info("compare %-26s %s  bad=%s/%s max|diff|=%s", name, s["status"], s.get("n_bad_cells"),
                 s.get("n_cells"), f"{s.get('max_abs_diff', float('nan')):.2e}" if "max_abs_diff" in s else "")
    write_table(OUT / "parity_summary.csv", pd.DataFrame(
        [{k: (str(v) if isinstance(v, dict) else v) for k, v in s.items()} for s in summary]))
    rep = write_report(results, summary)
    log.info("report: %s", rep)
    write_manifest("m8_legacy", inputs=[config.EXTERNAL["paper_npz"]],
                   outputs=sorted(OUT.glob("*.csv")) + [rep],
                   params={"parts": sorted(parts), "s15_dvs": args.s15_dvs},
                   counts={s["table"]: s.get("status") for s in summary}, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
