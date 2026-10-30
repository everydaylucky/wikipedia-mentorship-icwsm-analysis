"""
followup.py: later-period outcomes and a question-time anchor (camera-ready analyses for the reviewers)

Module: M8 (analyses added for the ICWSM 2027 camera-ready, answering the meta-review and reviewers 1 to 3)

1. What it does
    Four causal analyses with exactly the main design (propensity score with 5-fold cross-fitting, 10 strata,
    stratified ATT, 500 bootstrap draws clustered by newcomer; run_analysis.psm_att):
    a. Later, non-overlapping periods. Outcome = at least one (or at least five) mainspace edits in days 15 to 28,
       29 to 60 and 61 to 180 after the anchor. Each period uses only people whose follow-up reaches the end of the
       period before the data cutoff. Benjamini-Hochberg q values over the six tests.
    b. Active periods. Among people with 180 days of follow-up: in how many of the four periods (days 1 to 14,
       15 to 28, 29 to 60, 61 to 180) the newcomer made a mainspace edit (0 to 4), and whether that number is at
       least 3.
    c. Question-time anchor. Both groups start the clock at the question, so no pseudo reply time is needed.
       Outcome = at least one mainspace edit in the 14 days after the question.
    d. Long-term effect on one fixed cohort: people whose anchor + 365 days is before the data cutoff, so every
       period is observed for the same people. One propensity model and one set of strata for the cohort.
       Rules fixed before this analysis was run:
         - outcome: at least one (secondary: at least five) mainspace edits in each period;
         - periods: days 15 to 28, 29 to 60, 61 to 180 (the later periods of analysis a) and 181 to 365;
         - inference: Benjamini-Hochberg q values over the eight later estimates (four periods x two thresholds),
           as for the other families of tests in the paper; COHORT_BOOT bootstrap draws, so that the bootstrap
           p values are precise enough for the correction (with 500 draws they move in steps of 0.004);
         - days 1 to 14 are reported to compare the cohort with the full sample, outside the correction.
       Monthly description for the appendix figure: month k = days (30(k-1), 30k], k = 1 to 12; treated share and
       control share reweighted to the treated newcomers' strata. The ATT per month is saved with pointwise 95%
       intervals and a simultaneous 95% band over the 12 months (sup-t band from the same bootstrap draws,
       Montiel Olea and Plagborg-Moller 2019), but is not used for any claim.
       Every outcome uses the same bootstrap draws (the generator is reset to config.SEED for each), so the draws
       are joint across months and periods.

2. Why it is designed this way
    Reviewer 2 noted that cumulative windows (any edit within 7 to 180 days) are nested, so they are significant
    once the 14-day window is. Non-overlapping periods answer whether the effect lasts. Activity two to six
    months later follows the idea of newcomer "survival" in Halfaker et al. (2013). The question-time anchor
    answers reviewer 1's concern about the pseudo reply time; it is conservative, because the treated
    newcomers' window includes the wait before the reply.

3. Inputs
    data/analytic/psm_dataset_legacy_layout.npz (M7); data/interim/m3/mentee_edits/ (M3) for the question-time anchor.

4. Outputs (results/m8/, or results/m8_b<B>/ with --boot)
    followup_later_periods.csv, followup_active_periods.csv, followup_question_anchor.csv,
    followup_monthly.csv, followup_cohort_periods.csv

5. Key parameters
    The periods below (PERIODS); config.N_BOOT; config.SEED.

6. Possible errors and how they are handled
    A period with fewer than 100 eligible people is skipped with a warning.

7. How to run
    python -m pipeline.m8_analysis.followup              # final (B = 500)
    python -m pipeline.m8_analysis.followup --boot 100   # quick check

8. How to check
    checks/m8_check.ipynb (section 7) and tests/test_m8.py.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import config
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis import regression as reg
from pipeline.m8_analysis.run_analysis import covariates, load, psm_att, subset

#: (label, outcome key suffix, last day of the period, eligibility flag)
PERIODS = [("days 15-28", "15_28d", 28), ("days 29-60", "29_60d", 60), ("days 61-180", "61_180d", 180)]


def att_with_p(D: dict, y: np.ndarray, d: np.ndarray, rng, label: str) -> dict:
    """Same steps as run_analysis.psm_att, plus a two-sided bootstrap p value: 2 * min(share of draws <= 0, share >= 0)."""
    X = covariates(D)
    keep = X.std(axis=0) > 0
    ps = est.fit_propensity(X[:, keep], d)
    st = est.make_strata(ps.e_hat, config.N_STRATA)
    att = est.att_stratified(y, d, st).att
    draws = est.bootstrap_att(y, d, st, est.ClusterIndex(D["mentee_ids"]), rng, config.N_BOOT)
    lo, hi = est.percentile_ci(draws)
    p = min(1.0, 2 * min(float((draws <= 0).mean()), float((draws >= 0).mean())))
    return {"Spec": label, "N": int(len(y)), "Treated": int(d.sum()), "AUC": ps.auc, "ATT": att, "CI_lo": lo, "CI_hi": hi,
            "Sig": "*" if est.is_significant(lo, hi) else "", "p_boot": p, "Ctrl_mean": float(y[d == 0].mean()),
            "Treat_mean": float(y[d == 1].mean())}


def later_periods(D: dict, log) -> pd.DataFrame:
    """Analysis a: ATT for activity in each later period, at thresholds of 1 and 5 mainspace edits.

    The p value comes from the bootstrap draws (with B = 500 its smallest non-zero value is 0.004);
    Benjamini-Hochberg q values are computed over the six tests.
    """
    rng = np.random.RandomState(config.SEED)
    rows = []
    for label, key, end in PERIODS:
        m = D[f"eligible_{end}d"] == 1
        if m.sum() < 100:
            log.warning("[later] %s: only %d eligible, skipped", label, int(m.sum()))
            continue
        S = subset(D, m)
        for thr in ("1plus", "5plus"):
            r = att_with_p(S, S[f"oc_th_{thr}_{key}"].astype(float), S["y_treat"].astype(float), rng,
                           f"{label}, {'1+' if thr == '1plus' else '5+'} mainspace edits")
            rows.append({"period": label, "threshold": thr, **r})
            log.info("[later] %-40s N=%6d ATT=%+.4f [%+.4f, %+.4f]", r["Spec"], r["N"], r["ATT"], r["CI_lo"], r["CI_hi"])
    out = pd.DataFrame(rows)
    out["q_BH"] = reg.bh_qvalues(out.p_boot.to_numpy())
    return out


def active_periods(D: dict, log) -> pd.DataFrame:
    """Analysis b: number of the four periods with a mainspace edit, for people with 180 days of follow-up."""
    rng = np.random.RandomState(config.SEED)
    S = subset(D, D["eligible_180d"] == 1)
    parts = [S["oc_primary"]] + [S[f"oc_th_1plus_{key}"] for _, key, _ in PERIODS]
    n_active = np.sum([np.asarray(p, float) for p in parts], axis=0)
    d = S["y_treat"].astype(float)
    rows = [psm_att(S, n_active, d, rng, "number of active periods (0 to 4)"),
            psm_att(S, (n_active >= 3).astype(float), d, rng, "active in at least 3 of 4 periods")]
    for r in rows:
        log.info("[active] %-36s N=%6d ATT=%+.4f [%+.4f, %+.4f]", r["Spec"], r["N"], r["ATT"], r["CI_lo"], r["CI_hi"])
    return pd.DataFrame(rows)


def question_anchor_outcome(D: dict, days: int = 14) -> np.ndarray:
    """At least one mainspace edit in (question time, question time + days], recomputed from the M3 edits."""
    uids = [int(u) for u in D["mentee_ids"]]
    e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "timestamp", "ns"], filter=ds.field("uid").isin(uids)).to_pandas()
    e = e[e.ns == 0]
    e["ts"] = pd.to_datetime(e.timestamp, utc=True)
    q = pd.DataFrame({"uid": uids, "q": pd.to_datetime(pd.Series(D["q_ts"]), utc=True)})
    m = e.merge(q, on="uid")
    hit = m[(m.ts > m.q) & (m.ts <= m.q + pd.Timedelta(days=days))].uid.unique()
    return np.isin(uids, hit).astype(float)


def question_anchor(D: dict, log) -> pd.DataFrame:
    """Analysis c: both groups anchored at the question time; only people with 14 days of follow-up after it."""
    rng = np.random.RandomState(config.SEED)
    q = pd.to_datetime(pd.Series(D["q_ts"]), utc=True)
    S = subset(D, (q + pd.Timedelta(days=14) <= pd.Timestamp(config.DATA_CUTOFF)).to_numpy())
    y = question_anchor_outcome(S)
    r = psm_att(S, y, S["y_treat"].astype(float), rng, "1+ mainspace edit within 14 days of the question")
    log.info("[q-anchor] N=%6d ATT=%+.4f [%+.4f, %+.4f]", r["N"], r["ATT"], r["CI_lo"], r["CI_hi"])
    return pd.DataFrame([r])


MONTHS = 12


def monthly_matrix(D: dict) -> tuple[np.ndarray, np.ndarray]:
    """n x MONTHS matrix (1 = a mainspace edit in month k after the anchor) and the anchors (M5)."""
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet", columns=["first_question_revid", "mentee_uid", "anchor"])
    s = s.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[D["cid_order"].astype(np.int64)]
    a = pd.to_datetime(s.anchor, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    uids = s.mentee_uid.astype(np.int64).to_numpy()
    e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "timestamp", "ns"], filter=(ds.field("uid").isin(list(uids))) & (ds.field("ns") == 0)).to_pandas()
    ts = pd.to_datetime(e.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    pos = {u: i for i, u in enumerate(uids)}
    i = e.uid.map(pos).to_numpy()
    days = (ts - a[i]) / np.timedelta64(1, "D")
    m = np.ceil(days / 30).astype(int)
    ok = (days > 0) & (m >= 1) & (m <= MONTHS)
    Y = np.zeros((len(uids), MONTHS))
    Y[i[ok], m[ok] - 1] = 1
    return Y, a


COHORT_DAYS = 365
COHORT_BOOT = 2000
#: (label, first day, last day) of the periods for analysis d; days 1 to 14 sit outside the correction
COHORT_PERIODS = [("days 1-14", 0, 14), ("days 15-28", 14, 28), ("days 29-60", 28, 60), ("days 61-180", 60, 180),
                  ("days 181-365", 180, 365)]


def period_edits(D: dict, a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Number of mainspace edits in days (lo, hi] after the anchor, per newcomer."""
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet", columns=["first_question_revid", "mentee_uid"])
    uids = s.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[
        D["cid_order"].astype(np.int64)].mentee_uid.astype(np.int64).to_numpy()
    e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "timestamp", "ns"], filter=(ds.field("uid").isin(list(uids))) & (ds.field("ns") == 0)).to_pandas()
    ts = pd.to_datetime(e.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    i = e.uid.map({u: k for k, u in enumerate(uids)}).to_numpy()
    days = (ts - a[i]) / np.timedelta64(1, "D")
    ok = (days > lo) & (days <= hi)
    return np.bincount(i[ok], minlength=len(uids)).astype(float)


def monthly_activity(D: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analysis d: the fixed one-year cohort, by month (figure) and by the four fixed periods (tests)."""
    Y, a = monthly_matrix(D)
    cutoff = np.datetime64(pd.Timestamp(config.DATA_CUTOFF).tz_localize(None), "s")
    full = a + np.timedelta64(COHORT_DAYS, "D") <= cutoff
    S, Y, a = subset(D, full), Y[full], a[full]
    X, d = covariates(S), S["y_treat"].astype(float)
    st = est.make_strata(est.fit_propensity(X[:, X.std(axis=0) > 0], d).e_hat, config.N_STRATA)
    cl = est.ClusterIndex(S["mentee_ids"])
    n1 = np.array([((st == k) & (d == 1)).sum() for k in range(st.max() + 1)])

    def ctrl_w(y):
        return sum(n1[j] * y[(st == j) & (d == 0)].mean() for j in range(len(n1))) / n1.sum()

    def fit(y):
        B = COHORT_BOOT if config.N_BOOT == config.N_BOOT_FINAL else config.N_BOOT
        draws = est.bootstrap_att(y, d, st, cl, np.random.RandomState(config.SEED), B)
        lo, hi = est.percentile_ci(draws)
        p = min(1.0, 2 * min(float((draws <= 0).mean()), float((draws >= 0).mean())))
        return est.att_stratified(y, d, st).att, draws, lo, hi, p

    rows, draws = [], []
    for k in range(MONTHS):
        y = Y[:, k]
        att, dr, lo, hi, p = fit(y)
        draws.append(dr)
        rows.append({"month": k + 1, "N": int(len(y)), "Treated": int(d.sum()), "Treat_share": y[d == 1].mean(),
                     "Ctrl_share_reweighted": ctrl_w(y), "Ctrl_share_raw": y[d == 0].mean(), "ATT": att,
                     "CI_lo": lo, "CI_hi": hi, "Sig": "*" if est.is_significant(lo, hi) else ""})
    m = pd.DataFrame(rows)
    Dr = np.vstack(draws)
    se = Dr.std(axis=1, ddof=1)
    crit = float(np.percentile(np.max(np.abs(Dr - m.ATT.to_numpy()[:, None]) / se[:, None], axis=0), 95))
    m["band_lo"], m["band_hi"] = m.ATT - crit * se, m.ATT + crit * se
    m["band_crit"] = crit
    m["band_excludes_0"] = np.where(m.band_lo > 0, "*", "")
    log.info("[cohort] N=%d (treated %d); sup-t critical value %.2f (pointwise 1.96)", len(d), int(d.sum()), crit)

    prow = []
    for label, lo_d, hi_d in COHORT_PERIODS:
        n = period_edits(S, a, lo_d, hi_d)
        for thr in (1, 5):
            y = (n >= thr).astype(float)
            att, _, lo, hi, p = fit(y)
            prow.append({"period": label, "threshold": f"{thr}+", "N": int(len(y)), "Treat_mean": y[d == 1].mean(),
                         "Ctrl_mean_reweighted": ctrl_w(y), "ATT": att, "CI_lo": lo, "CI_hi": hi, "p_boot": p,
                         "Sig": "*" if est.is_significant(lo, hi) else ""})
    P = pd.DataFrame(prow)
    later = P.period != "days 1-14"
    P["q_BH"] = np.nan
    P.loc[later, "q_BH"] = reg.bh_qvalues(P.loc[later, "p_boot"].to_numpy())
    for r in P.itertuples():
        log.info("[cohort] %-13s %s ATT=%+.4f [%+.4f, %+.4f] p=%.4f q=%s", r.period, r.threshold, r.ATT, r.CI_lo,
                 r.CI_hi, r.p_boot, "" if np.isnan(r.q_BH) else f"{r.q_BH:.3f}")
    return m, P


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("M8 follow-up analyses for the camera-ready (later periods, active periods, question anchor, monthly activity).")
    ap.add_argument("--boot", type=int, default=config.N_BOOT, help="bootstrap draws (final 500; 100 for a quick check)")
    args = ap.parse_args(argv)
    config.N_BOOT = args.boot
    debug = config.N_BOOT != config.N_BOOT_FINAL
    out_dir = config.RESULTS / ("m8" if not debug else f"m8_b{config.N_BOOT}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging("m8_followup")
    t0 = time.time()
    check_upstream("m8_followup", ["m7"])
    Dall = load()
    D = subset(Dall, Dall["in_sample"])
    res = {"followup_later_periods": later_periods(D, log), "followup_active_periods": active_periods(D, log),
           "followup_question_anchor": question_anchor(D, log)}
    res["followup_monthly"], res["followup_cohort_periods"] = monthly_activity(D, log)
    for k, v in res.items():
        v.to_csv(out_dir / f"{k}.csv", index=False)
    write_manifest("m8_followup" if not debug else f"m8_followup_b{config.N_BOOT}",
                   inputs=[config.ANALYTIC / "psm_dataset_legacy_layout.npz"],
                   outputs=[out_dir / f"{k}.csv" for k in res], params={"n_boot": config.N_BOOT, "seed": config.SEED},
                   counts={k: len(v) for k, v in res.items()}, started=t0)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
