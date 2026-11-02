"""
editor_kinds.py: what kind of editors newcomers become: edit-count levels reached and the main type of later editing

Module: M8 (camera-ready; moved from the exploration of 2026-11-02, editor_kinds.py in the research notes)

1. What it does
    For the one-year cohort of followup.monthly_activity (newcomers whose anchor is at least 365 days before the
    data cutoff; same propensity model and 10 strata), counting only edits after the anchor:
        Test 1, four levels reached within the year (all namespaces), Benjamini-Hochberg over the four:
            at least 10 edits; at least 5 edits in some rolling 30-day window (active); at least 100 edits in some
            30-day window (very active); at least 500 edits.
            Variants (no correction): edits in days 15-365 only; article edits only; 10 edits within days 1-30.
        Test 2, the main type of editing in days 15-365 (namespace group with at least half of the edits; mixed;
            no edits after day 14): stratified ATT for each of the six shares, and one joint Wald test on five of
            them (the six sum to one), with the covariance of the bootstrap draws.
    Every quantity uses the same newcomer bootstrap draws (B = 2000, RandomState(config.SEED)), as the cohort
    periods of followup.py.

2. Why it is designed this way
    The levels count only edits after the anchor, so they do not depend on how much a newcomer edited before the
    question. 10 and 500 are the edit counts of autoconfirmed and extended confirmed accounts; the module does not
    check account age or whether a right was granted. Active and very active follow Wikimedia's definitions, with a
    rolling 30-day window instead of calendar months, so the result does not depend on where months are cut.
    The specification was fixed on 2026-11-02 before the final run (secondary analysis, defined in the camera-ready
    revision). A carry-forward benchmark that was tried in the exploration is not included: it could not separate
    the two cases it was meant to separate.

3. Inputs
    data/analytic/psm_dataset_legacy_layout.npz (through load); data/interim/m5/sample.parquet (anchor, uid);
    data/interim/m3/mentee_edits (edit times and namespaces); results/m8/followup_cohort_periods.csv (check).

4. Outputs
    results/m8/editor_kinds_levels.csv   one row per outcome: treated share, reweighted control share, ATT, 95% CI,
                                         bootstrap p, relative effect, q (test 1 rows only)
    results/m8/editor_kinds_types.csv    one row per editing type, same columns
    results/m8/editor_kinds_wald.csv     the joint test of test 2

5. Key parameters
    DAYS = 365; levels 10, 5 and 100 per 30 days, 500; B = COHORT_BOOT (2000); main type threshold 0.5.

6. Possible errors and how they are handled
    The cohort's days 1-14 and days 181-365 ATTs must equal followup_cohort_periods.csv, otherwise the run stops
    (different cohort or strata). The rolling-window counts are checked by brute force on 200 newcomers.

7. How to run
    python -m pipeline.m8_analysis.editor_kinds             # B = 2000
    python -m pipeline.m8_analysis.editor_kinds --boot 50   # quick test (does not write results)

8. How to check
    The exploration script of 2026-11-02 gave the same numbers (research notes, editor_kinds_results/).
    An independent recomputation of the point estimates with pandas and est.att_stratified is in the research notes
    of 2026-11-02 (check_editor_kinds.py).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy import stats as sps

import config
from pipeline.common.manifest import write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis import regression as reg
from pipeline.m8_analysis.descriptive import ns_group
from pipeline.m8_analysis.followup import COHORT_BOOT
from pipeline.m8_analysis.run_analysis import covariates, load, subset

OUT = config.RESULTS / "m8"
DAYS = 365
LEVELS = [("10 edits", "n", 10), ("active: 5 edits in a 30-day window", "w", 5),
          ("very active: 100 edits in a 30-day window", "w", 100), ("500 edits", "n", 500)]
TYPES = ["no edits after day 14", "articles", "drafts", "user_and_talk", "other", "mixed"]


def cohort() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """The one-year cohort of followup.monthly_activity: subset of D, anchors, user ids, PS strata."""
    Dall = load()
    D = subset(Dall, Dall["in_sample"])
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet", columns=["first_question_revid", "mentee_uid", "anchor"])
    s = s.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[D["cid_order"].astype(np.int64)]
    a = pd.to_datetime(s.anchor, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    cutoff = np.datetime64(pd.Timestamp(config.DATA_CUTOFF).tz_localize(None), "s")
    full = a + np.timedelta64(DAYS, "D") <= cutoff
    S = subset(D, full)
    X = covariates(S)
    st = est.make_strata(est.fit_propensity(X[:, X.std(axis=0) > 0], S["y_treat"].astype(float)).e_hat, config.N_STRATA)
    return S, a[full], s.mentee_uid.astype(np.int64).to_numpy()[full], st


def max_in_30_days(i: np.ndarray, day: np.ndarray, n: int) -> np.ndarray:
    """Per newcomer, the largest number of the given edits in any window [x, x + 30) days (x = an edit's day)."""
    o = np.lexsort((day, i))
    i, day = i[o], day[o]
    key = i * 1000.0 + day                                      # days <= 365 < 1000: sorted by newcomer, then day
    inwin = np.searchsorted(key, key + 30, side="left") - np.arange(len(key))
    out = np.zeros(n)
    np.maximum.at(out, i, inwin)
    return out


def outcomes(a: np.ndarray, uids: np.ndarray) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """All outcome indicators; returns (outcomes, names of the four test-1 levels, names of the six types)."""
    n = len(uids)
    e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "timestamp", "ns"], filter=ds.field("uid").isin(list(uids))).to_pandas()
    ts = pd.to_datetime(e.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    i = e.uid.map({u: k for k, u in enumerate(uids)}).to_numpy()
    day = (ts - a[i]) / np.timedelta64(1, "D")
    art = e.ns.to_numpy() == 0
    post, late = (day > 0) & (day <= DAYS), (day > 14) & (day <= DAYS)

    def count(m):
        return np.bincount(i[m], minlength=n)

    def win(m):
        return max_in_30_days(i[m], day[m], n)

    # brute-force check of the rolling count on 200 newcomers
    w_all = win(post)
    for u in np.unique(i[post])[:200]:
        dv = day[post & (i == u)]
        assert w_all[u] == max(((dv >= x) & (dv < x + 30)).sum() for x in dv), u
    y: dict[str, np.ndarray] = {}
    main = []
    for label, kind, thr in LEVELS:
        y[label] = (count(post) if kind == "n" else w_all) >= thr
        main.append(label)
    for prefix, m in (("days 15-365 only: ", late), ("article edits only: ", post & art)):
        c, w = count(m), win(m)
        for label, kind, thr in LEVELS:
            y[prefix + label] = (c if kind == "n" else w) >= thr
    y["10 edits within days 1-30"] = count((day > 0) & (day <= 30)) >= 10
    y["article edit, days 1-14 (check)"] = count(art & (day > 0) & (day <= 14)) > 0
    y["article edit, days 181-365 (check)"] = count(art & (day > 180) & (day <= DAYS)) > 0
    g = ns_group(e.ns.to_numpy())
    sh = pd.crosstab(i[late], g[late]).reindex(range(n), fill_value=0)
    tot = sh.sum(1)
    frac = sh.div(tot.where(tot > 0, 1), axis=0)
    typ = frac.idxmax(1).where(frac.max(1) >= 0.5, "mixed").where(tot > 0, "no edits after day 14").to_numpy()
    types = []
    for c in TYPES:
        y[f"type: {c}"] = typ == c
        types.append(f"type: {c}")
    return {k: np.asarray(v, float) for k, v in y.items()}, main, types


def all_stats(y: dict[str, np.ndarray], d: np.ndarray, st: np.ndarray, w: np.ndarray) -> np.ndarray:
    """For every outcome: stratified ATT, treated share, control share reweighted to the treated strata."""
    K, t = int(st.max()) + 1, d == 1
    w1, w0 = w * t, w * ~t
    N1, N0 = np.bincount(st, w1, K), np.bincount(st, w0, K)
    ok = (N1 >= 2) & (N0 >= 2)
    wk = np.where(ok, N1, 0) / N1[ok].sum()
    cw = w0 * np.where(N0[st] > 0, N1[st] / np.where(N0[st] > 0, N0[st], 1), 0) * ok[st]
    out = []
    for v in y.values():
        m1 = np.bincount(st, w1 * v, K) / np.where(N1 > 0, N1, 1)
        m0 = np.bincount(st, w0 * v, K) / np.where(N0 > 0, N0, 1)
        out += [float((wk * (m1 - m0)).sum()), float((w1 * v).sum() / w1.sum()), float((cw * v).sum() / cw.sum())]
    return np.array(out)


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("What kind of editors newcomers become: edit-count levels and the main type of later editing.")
    ap.add_argument("--boot", type=int, default=COHORT_BOOT)
    args = ap.parse_args(argv)
    log = setup_logging("m8_editor_kinds")
    t0 = time.time()

    S, a, uids, st = cohort()
    d = S["y_treat"].astype(float)
    y, main_levels, types = outcomes(a, uids)
    names = list(y)
    pt = all_stats(y, d, st, np.ones(len(d))).reshape(-1, 3)
    ref = pd.read_csv(OUT / "followup_cohort_periods.csv").set_index(["period", "threshold"])
    for k, (p, th) in (("article edit, days 1-14 (check)", ("days 1-14", "1+")),
                       ("article edit, days 181-365 (check)", ("days 181-365", "1+"))):
        assert abs(pt[names.index(k), 0] - ref.at[(p, th), "ATT"]) < 1e-9, "cohort differs from followup.py"
    log.info("cohort N=%d treated=%d; matches followup_cohort_periods", len(d), int(d.sum()))

    cl = est.ClusterIndex(S["mentee_ids"])
    rng = np.random.RandomState(config.SEED)
    draws = np.array([all_stats(y, d, st, cl.draw_weights(rng)).reshape(-1, 3) for _ in range(args.boot)])
    rows = []
    for j, k in enumerate(names):
        att, ctl = pt[j, 0], pt[j, 2]
        b = draws[:, j, 0]
        lo, hi = np.percentile(b, [2.5, 97.5])
        rel = b / draws[:, j, 2]
        rows.append({"outcome": k, "N": len(d), "Treated": int(d.sum()), "treated_share": pt[j, 1],
                     "ctrl_share_reweighted": ctl, "ATT": att, "CI_lo": lo, "CI_hi": hi,
                     "p_boot": float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))),
                     "relative": att / ctl, "rel_CI_lo": np.nanpercentile(rel, 2.5),
                     "rel_CI_hi": np.nanpercentile(rel, 97.5), "B": args.boot})
    R = pd.DataFrame(rows)
    R["q_BH"] = np.nan
    m = R.outcome.isin(main_levels)
    R.loc[m, "q_BH"] = reg.bh_qvalues(R.loc[m, "p_boot"].to_numpy())

    free = [names.index(c) for c in types[:-1]]                # five of the six shares (they sum to one)
    tv, Db = pt[free, 0], draws[:, free, 0]
    W = float(tv @ np.linalg.solve(np.cov(Db, rowvar=False), tv))
    wald = pd.DataFrame([{"test": "main type of editing in days 15-365, five free shares", "W": W, "df": len(free),
                          "p": float(sps.chi2.sf(W, len(free))), "B": args.boot}])
    for r in R[m].itertuples():
        log.info("%-45s %.3f vs %.3f  ATT %+.4f [%+.4f, %+.4f] q=%.3f", r.outcome, r.treated_share,
                 r.ctrl_share_reweighted, r.ATT, r.CI_lo, r.CI_hi, r.q_BH)
    log.info("type test: W = %.2f, df = %d, p = %.3f", W, len(free), wald.p.iloc[0])
    if args.boot != COHORT_BOOT:
        log.warning("test run: B = %d, results not written", args.boot)
        return
    lv, ty, wf = OUT / "editor_kinds_levels.csv", OUT / "editor_kinds_types.csv", OUT / "editor_kinds_wald.csv"
    R[~R.outcome.isin(types)].to_csv(lv, index=False)
    R[R.outcome.isin(types)].to_csv(ty, index=False)
    wald.to_csv(wf, index=False)
    write_manifest("m8_editor_kinds", inputs=[config.ANALYTIC / "psm_dataset_legacy_layout.npz",
                                              config.INTERIM / "m5" / "sample.parquet", OUT / "followup_cohort_periods.csv"],
                   outputs=[lv, ty, wf], params={"n_boot": args.boot, "seed": config.SEED, "days": DAYS},
                   counts={"N": len(d), "treated": int(d.sum())}, started=t0)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
