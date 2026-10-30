"""
run_analysis.py: M8 runs all of the paper's analyses on the new data, plus the robustness checks in PLAN M8.6

Module: M8 (PLAN.md M8; CODE_PLAN.md section 5, M8)
Rules: PLAN M8.1 to M8.9. Issues: I-10, I-11, I-19, I-24, I-26, I-35, I-36

1. What it does
    1. Reads the npz in the paper's layout exported by M7 (psm_dataset_legacy_layout.npz) and keeps the main
       analysis sample: in_sample and a full 14-day follow-up (analysis_sample, I-10). Each outcome then uses only
       people whose own window ends before the cutoff (add_followup_masks): active_days_30d needs 30 days, the
       window table needs w days, and the control group's fixed-lag pseudo start times move the window start.
    2. Runs the code in legacy_parity, which matches the paper version cell by cell:
        run_s13: propensity score, stratification, balance (both the paper's definition and the ATT definition), ablation,
                 main results (newcomer bootstrap and mentor bootstrap), number of strata, 48-hour definition,
                 trimming, windows, E-values (three methods; the corrected one is RR = p1/(p1 - tau)), pseudo start time.
        run_s14: subgroup ATT, subgroup diagnostics, sensitivity of subgroups to the number of strata.
        run_a2 : interaction tests (differences between groups, I-24).
        reply_association: mixed model of reply features (random intercept per mentor), VIF, BH correction,
                 subgroups (paper Tables 6 and 7; I-26). The input is now the M6 reply features, aligned on the
                 key first_question_revid (I-33).
    3. Robustness. For every setting we re-estimate the propensity score, re-stratify, and run 500 bootstrap draws
       clustered by newcomer:
        - the sample without the D10 restriction (R2.5);
        - treatment counts only replies from the assigned mentor (R5.3): assigned mentor replied vs no reply at all,
          with newcomers answered only by other editors left out and treated newcomers measured from the assigned
          mentor's first reply (add_mentor_outcome); mentor_reply_versions() also reports the simpler versions;
        - drop people fetched late (late_fetch, M3);
        - only questions asked at least 30 days before the cutoff (I-11: questions near the cutoff did not yet have time to get a reply);
        - also keep people without a full 14-day follow-up (the main sample before I-10, for comparison);
        - only people whose LLM labels, Perspective scores and embedding are all present (the subsample with no imputed values);
        - only people who are also in the paper sample (in_paper), for a direct comparison with the paper.

2. Why it is designed this way
    We reuse code that already reproduces the paper version cell by cell. So any difference between old and new
    results comes from the data itself. Each new check answers a problem listed in PLAN.

3. Inputs
    data/analytic/psm_dataset_legacy_layout.npz; data/interim/m6/reply_features.parquet.

4. Outputs
    results/m8/*.csv (one file per table); results/m8/summary.md (main results, robustness, comparison with the paper)

7. How to run
    python -m pipeline.m8_analysis.run_analysis            # final results (B = 500), about 4 minutes with 8 processes
    python -m pipeline.m8_analysis.run_analysis --boot 100 # debugging: B = 100, writes to results/m8_b100/, does not overwrite the final results
    python -m pipeline.m8_analysis.run_analysis --jobs 1   # no multiprocessing (a2 interactions run one task per process; results are bit-identical to a serial run)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.preprocessing import StandardScaler

import config
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis import legacy_parity as lp
from pipeline.m8_analysis import regression as reg
from pipeline.m5_sample.outcomes import build_outcomes, edit_arrays

OUT = config.RESULTS / "m8"


def load(path=None) -> dict:
    with np.load(path or config.ANALYTIC / "psm_dataset_legacy_layout.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def subset(D: dict, mask: np.ndarray) -> dict:
    """Take a subset of rows (applies to every array whose first dimension equals the sample size)."""
    n = len(D["y_treat"])
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in D.items()}


#: Window length (days) of each outcome used in the M8 tables; every other one is a 14-day window.
WINDOW_DAYS = {"active_days_30d": 30, **{f"mainspace_{w}d": w for w in config.OUTCOME_WINDOWS_DAYS}}


def add_followup_masks(D: dict) -> dict:
    """Add keep__{key}: the window of outcome oc_{key} ends before the cutoff (I-10).

    The main start time uses eligible_{w}d from M5. For the fixed-lag versions (oc_{key}__median and so on) a
    control's window starts at q_ts + that lag, so the flag is recomputed; the treated group keeps the main flag.
    The multi-draw version averages 20 draws and keeps the main flag.
    """
    lags = dict(zip(("median", "p25", "p75", "zero"), D["pseudo_lag_info"][:4]))
    q = pd.to_datetime(pd.Series(D["q_ts"]), utc=True)
    cutoff = pd.Timestamp(config.DATA_CUTOFF)
    treated = D["y_treat"] == 1
    keys = {k for _, k in lp.DIMS_S13 + lp.DIMS_S14} | {f"mainspace_{w}d" for w in config.OUTCOME_WINDOWS_DAYS}
    for key in keys:
        w = WINDOW_DAYS.get(key, 14)
        main = D[f"eligible_{w}d"] == 1
        D[f"keep__{key}"] = main
        D[f"keep__{key}__multi"] = main
        for name, lag in lags.items():
            ok = (q + pd.Timedelta(days=float(lag) + w) <= cutoff).to_numpy()
            D[f"keep__{key}__{name}"] = (treated & main) | (~treated & ok)
    return D


def add_mentor_outcome(D: dict) -> dict:
    """Outcomes for the assigned-mentor treatment (R5.3), added to D.

    others_only: someone replied, but never the assigned mentor (dropped from this comparison).
    oc_primary__mentor: the primary outcome with the window starting at the assigned mentor's first reply. It differs
        from oc_primary only for newcomers whose first reply came from someone else; for them it is recomputed from
        the M3 edits (the primary outcome does not use revert information).
    keep__primary__mentor: the 14-day window from that start ends before the cutoff.
    """
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet",
                        columns=["first_question_revid", "mentee_uid", "q_ts", "first_reply_ts", "mentor_first_reply_ts"])
    s = s.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[D["cid_order"].astype(np.int64)]
    first = pd.to_datetime(s.first_reply_ts, utc=True)
    mentor = pd.to_datetime(s.mentor_first_reply_ts, utc=True)
    D["others_only"] = (first.notna() & mentor.isna()).to_numpy()
    later = (mentor.notna() & (mentor > first)).to_numpy()
    y = D["oc_primary"].astype(float).copy()
    uids = s.mentee_uid.astype(np.int64).to_numpy()
    e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "revid", "timestamp", "ns", "tags"], filter=ds.field("uid").isin(list(uids[later]))).to_pandas()
    e["timestamp"] = pd.to_datetime(e.timestamp, utc=True).dt.tz_localize(None)
    groups, empty = dict(tuple(e.groupby("uid"))), e.iloc[0:0]
    for i in np.where(later)[0]:
        y[i] = build_outcomes(edit_arrays(groups.get(uids[i], empty)), mentor.iloc[i].to_pydatetime())["primary"]
    D["oc_primary__mentor"] = y
    ok = (mentor + pd.Timedelta(days=14) <= pd.Timestamp(config.DATA_CUTOFF)).to_numpy()
    D["keep__primary__mentor"] = np.where(later, ok, D["keep__primary"])
    return D


def analysis_sample(Dall: dict | None = None) -> dict:
    """The main analysis sample: in_sample with a full 14-day follow-up, with the per-outcome masks added."""
    Dall = add_followup_masks(load() if Dall is None else Dall)
    return subset(Dall, Dall["in_sample"] & Dall["keep__primary"])


def covariates(D: dict) -> np.ndarray:
    B = lp.blocks(D)
    return np.hstack([B["E"], B["Qtext"], B["Qpersp"], B["Qtype"], B["emb20"], B["temporal"]])


def psm_att(D: dict, y: np.ndarray, d: np.ndarray, rng, label: str) -> dict:
    """One setting: propensity score (5-fold cross-fitting), then 10 strata, then ATT, then 500 bootstrap draws clustered by newcomer."""
    X = covariates(D)
    keep = X.std(axis=0) > 0
    ps = est.fit_propensity(X[:, keep], d)
    st = est.make_strata(ps.e_hat, config.N_STRATA)
    att = est.att_stratified(y, d, st).att
    lo, hi = est.percentile_ci(est.bootstrap_att(y, d, st, est.ClusterIndex(D["mentee_ids"]), rng, config.N_BOOT))
    return {"Spec": label, "N": int(len(y)), "Treated": int(d.sum()), "AUC": ps.auc, "ATT": att, "CI_lo": lo, "CI_hi": hi,
            "Sig": "*" if est.is_significant(lo, hi) else "", "Ctrl_mean": float(y[d == 0].mean()),
            "Treat_mean": float(y[d == 1].mean())}


def robustness(Dall: dict, log) -> pd.DataFrame:
    """The settings in section 1, item 3 (main outcome oc_primary)."""
    rng = np.random.RandomState(config.SEED)
    full14 = Dall["keep__primary"]
    main = Dall["in_sample"] & full14
    cutoff30 = (pd.Timestamp(config.DATA_CUTOFF) - pd.Timedelta(days=30)).isoformat()
    q_ts = pd.to_datetime(pd.Series(Dall["q_ts"]), utc=True)
    specs = [
        ("Main (D10 sample)", main, "y_treat", "oc_primary"),
        ("No D10 restriction (R2.5)", Dall["in_sample_unrestricted"] & full14, "y_treat", "oc_primary"),
        ("Treatment = assigned mentor replied (vs no reply)",
         Dall["in_sample"] & Dall["keep__primary__mentor"] & ~Dall["others_only"], "y_treat_mentor", "oc_primary__mentor"),
        ("Exclude late-fetched askers (M3)", main & ~Dall["late_fetch"], "y_treat", "oc_primary"),
        ("Questions ≥ 30 days before the cutoff (I-11)", main & (q_ts < pd.Timestamp(cutoff30)).to_numpy(), "y_treat",
         "oc_primary"),
        ("Also people without a full 14-day follow-up (before I-10)", Dall["in_sample"], "y_treat", "oc_primary"),
        ("Complete text annotations (no imputation)", main & Dall["text_complete"], "y_treat", "oc_primary"),
        ("Askers also in the paper sample", main & Dall["in_paper"], "y_treat", "oc_primary"),
    ]
    rows = []
    for label, m, dkey, ykey in specs:
        D = subset(Dall, m)
        r = psm_att(D, D[ykey].astype(float), D[dkey].astype(float), rng, label)
        log.info("[robust] %-48s N=%6d ATT=%+.4f [%+.4f, %+.4f]", label, r["N"], r["ATT"], r["CI_lo"], r["CI_hi"])
        rows.append(r)
    return pd.DataFrame(rows)


def mentor_reply_versions(Dall: dict, log) -> pd.DataFrame:
    """Four ways to compare by who replied (primary outcome). Groups of treated newcomers: the assigned mentor replied
    first; someone else replied first and the mentor later; only others replied (others_only). Controls: no reply."""
    rng = np.random.RandomState(config.SEED)
    full14 = Dall["in_sample"] & Dall["keep__primary"]
    specs = [("Anyone replied vs no reply (main)", full14, "y_treat", "oc_primary"),
             ("Assigned mentor replied vs no reply or only others", full14, "y_treat_mentor", "oc_primary"),
             ("Assigned mentor replied vs no reply (only others dropped)", full14 & ~Dall["others_only"], "y_treat_mentor",
              "oc_primary"),
             ("Same, measured from the assigned mentor's first reply",
              Dall["in_sample"] & Dall["keep__primary__mentor"] & ~Dall["others_only"], "y_treat_mentor", "oc_primary__mentor")]
    rows = []
    for label, m, dkey, ykey in specs:
        D = subset(Dall, m)
        r = psm_att(D, D[ykey].astype(float), D[dkey].astype(float), rng, label)
        log.info("[mentor] %-58s N=%6d ATT=%+.4f [%+.4f, %+.4f]", label, r["N"], r["ATT"], r["CI_lo"], r["CI_hi"])
        rows.append(r)
    return pd.DataFrame(rows)


def windows_followup(D: dict, log) -> pd.DataFrame:
    """Window sensitivity, using only people with full follow-up for each window (I-10)."""
    rng = np.random.RandomState(config.SEED)
    rows = []
    for w in config.OUTCOME_WINDOWS_DAYS:
        key = f"eligible_{w}d"
        m = D[key] == 1
        S = subset(D, m)
        r = psm_att(S, S[f"oc_mainspace_{w}d"].astype(float), S["y_treat"].astype(float), rng, f"{w}d (eligible: {key})")
        rows.append({"window": w, **r})
        log.info("[window] %3dd N=%6d ATT=%+.4f", w, r["N"], r["ATT"])
    return pd.DataFrame(rows)


def reply_association(D: dict, rf: pd.DataFrame, log) -> dict[str, pd.DataFrame]:
    """Paper s15 Parts 1 and 3: within the treated group, a mixed model of reply features and the outcome
    (random intercept per mentor), controlling for all covariates.

    Same preprocessing as legacy_parity.run_s15 (NaN to 0, reply delay capped at the 99th percentile,
    standardization, drop features with VIF > 10). The only change: the input is the M6 reply features,
    aligned on first_question_revid (I-33).
    """
    d = D["y_treat"] == 1
    idx = {int(rv): i for i, rv in enumerate(D["cid_order"])}
    cols = [c for c, _ in lp.REPLY_FEATURES]
    names = [n for _, n in lp.REPLY_FEATURES]
    rf = rf[rf.first_question_revid.map(lambda r: int(r) in idx and d[idx[int(r)]])]
    psi = np.array([idx[int(r)] for r in rf.first_question_revid])
    Xc = StandardScaler().fit_transform(covariates(D)[psi])
    Xr = np.nan_to_num(rf[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    j = cols.index("r_reply_lag_hours")
    Xr[:, j] = np.clip(Xr[:, j], 0, np.percentile(Xr[:, j], 99))
    Xr = StandardScaler().fit_transform(Xr)
    Xr, kept, vif = reg.vif_screen(Xr, names)
    mentors = D["mentor_ids"][psi]
    qt = D["X_Qtype"][psi]
    q_sub, q_ref, q_apr, q_nor, q_own = (qt[:, k] == 1 for k in range(5))
    sgs = [("Technical", q_sub & ~q_ref & ~q_apr & ~q_nor), ("Referent (Q2)", q_ref), ("Normative (Q4)", q_nor),
           ("Non-substantive", ~q_sub), ("Own work (Q5=Y)", q_own), ("No own work (Q5=N)", ~q_own)]
    y = D["oc_primary"][psi].astype(float)
    log.info("[reply] %d treated with reply features; %d mentors; VIF dropped %s", len(psi), len(set(mentors)),
             list(vif.loc[vif.dropped, "feature"]))
    res = reg.mixedlm_with_controls(y, Xr, Xc, mentors)
    full = [{"Feature": nm, "Beta_std": res.beta[k], "SE": res.se[k], "z": res.z[k], "p": res.p[k],
             "Sig": reg.sig_stars(res.p[k]), "ICC": res.icc, "method": res.method} for k, nm in enumerate(kept)]
    full = pd.DataFrame(full)
    full["q_BH"] = reg.bh_qvalues(full.p.to_numpy())
    bysg = []
    for sg, m in sgs:
        if m.sum() < 200:
            continue
        r = reg.mixedlm_with_controls(y[m], Xr[m], Xc[m], mentors[m])
        for k, nm in enumerate(kept):
            bysg.append({"Subgroup": sg, "N": int(m.sum()), "Feature": nm, "Beta_std": r.beta[k], "SE": r.se[k],
                         "p": r.p[k], "Sig": reg.sig_stars(r.p[k]), "method": r.method})
    return {"reply_association": full, "reply_association_by_subgroup": pd.DataFrame(bysg), "reply_vif": vif}


def summary_md(out: dict[str, pd.DataFrame]) -> str:
    """Main results, E-values, robustness, and the comparison with the paper's numbers."""
    b = out["B_main_results"]
    lines = ["# M8 results summary (new data)", "", f"Cutoff {config.DATA_CUTOFF_ISO}; sample {int(out['_n'])} people, "
             f"treated group {int(out['_t'])} people; bootstrap B = {config.N_BOOT}"
             + (" (debug run; final results use 500)" if config.N_BOOT != config.N_BOOT_FINAL else "") + ".", "", "## Main results (paper Table 4)", "",
             "| Outcome | Naive difference | ATT | 95% CI | Cohen d |", "|---|---|---|---|---|"]
    for r in b.itertuples():
        lines.append(f"| {r.Outcome} | {r.Naive:+.4f} | {r.ATT:+.4f}{r.Sig} | [{r.CI_lo:+.4f}, {r.CI_hi:+.4f}] | {r.Cohen_d:+.3f} |")
    lines += ["", "For a line-by-line comparison of paper Table 4 with the new data, see results/paper_vs_new.md (M9). For the cell-by-cell reproduction of the paper npz, see results/m8_legacy/.", "",
              "## E-values", "", out["evalues"].to_markdown(index=False), "", "## Robustness", "",
              out["robustness"][["Spec", "N", "Treated", "AUC", "ATT", "CI_lo", "CI_hi", "Sig"]].to_markdown(index=False,
                                                                                                         floatfmt="+.4f"),
              "", "## Subgroups (paper Table 5)", "",
              out["subgroup_att"].query("Outcome == Outcome").head(40).to_markdown(index=False, floatfmt=".4f")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    global OUT
    ap = base_parser("M8: all analyses on the new dataset.")
    ap.add_argument("--boot", type=int, default=config.N_BOOT, help="number of bootstrap draws (500 for final results; 100 is fine for debugging)")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS, help="number of parallel processes")
    args = ap.parse_args(argv)
    config.N_BOOT, config.N_JOBS = args.boot, args.jobs  # the estimator functions read these two values when they are called
    debug = config.N_BOOT != config.N_BOOT_FINAL
    if debug:
        OUT = config.RESULTS / f"m8_b{config.N_BOOT}"
    log = setup_logging("m8" if not debug else f"m8_b{config.N_BOOT}")
    if debug:
        log.warning("debug run: B = %d (final results use %d); writing to %s", config.N_BOOT, config.N_BOOT_FINAL, OUT)
    lp.log = log                                         # functions in legacy_parity use a module-level log
    t0 = time.time()
    check_upstream("m8", ["m7"])
    OUT.mkdir(parents=True, exist_ok=True)
    Dall = add_mentor_outcome(add_followup_masks(load()))
    D = analysis_sample(Dall)
    out: dict[str, pd.DataFrame] = {}
    out.update(lp.run_s13(D))
    out.update(lp.run_s14(D))
    out.update(lp.run_a2(D))
    out["robustness"] = robustness(Dall, log)
    out["mentor_reply_versions"] = mentor_reply_versions(Dall, log)
    out["window_followup_complete"] = windows_followup(subset(Dall, Dall["in_sample"]), log)
    rf = pd.read_parquet(config.INTERIM / "m6" / "reply_features.parquet")
    out.update(reply_association(D, rf, log))
    for k, v in out.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    out["_n"], out["_t"] = len(D["y_treat"]), D["y_treat"].sum()
    (OUT / "summary.md").write_text(summary_md(out))
    main_row = out["B_main_results"].iloc[0]
    counts = {"N": int(out["_n"]), "treated": int(out["_t"]), "ATT_primary": float(main_row.ATT),
              "CI": [float(main_row.CI_lo), float(main_row.CI_hi)]}
    log.info("done: %s", counts)
    write_manifest("m8" if not debug else f"m8_b{config.N_BOOT}", inputs=[config.ANALYTIC / "psm_dataset_legacy_layout.npz"], outputs=[OUT / "summary.md"],
                   params={"n_boot": config.N_BOOT, "seed": config.SEED, "jobs": config.N_JOBS}, counts=counts, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
