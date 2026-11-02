"""
anchor_checks.py: checks of the start time of the control window (paper Appendix "Choosing the Start of the Control
Window", Tables lag_strata, landmark, and the within-stratum rows of Table anchor)

Module: M8 (camera-ready; moved from the exploration of 2026-11-02, missing_numbers.py in the research notes)

1. What it does
    All on the 34,073 newcomers in the sample, with propensity strata formed on all of them (main PS model, 10 strata).
    a. Reply lag by stratum: number of treated newcomers in the lag pool (0 <= lag < 365 days, as M5) and the 25th,
       50th and 75th percentile of their reply lag in hours, per stratum.
    b. Does reply timing respond to editing after the question? Among newcomers with no reply in the first 24 hours,
       the share who got a reply in hours 24 to 168, for those who made an article edit in the first 24 hours vs
       those who did not, raw and as a stratified difference (edited = "treatment") with a newcomer bootstrap.
    c. Landmark analysis: for L = 12, 24, 48, 72, 168 hours, treated = a first reply within L hours; controls = no
       reply within L hours (including later replies); both measured by the primary outcome over the 14 days after
       q + L; newcomers whose window ends after the cutoff are left out. The PS model is refitted on each sample
       (run_analysis.psm_att). At L = 48 hours also with controls who never got a reply.
    d. Control start drawn within the PS stratum: each control's lag is drawn from the treated lags of the same
       stratum, (i) one draw, (ii) the mean of the primary outcome over 20 draws; then the main ATT (psm_att).

2. Why it is designed this way
    The main design borrows one pool of treated reply lags for all controls; a and b check its two conditions, c and
    d relax it. Definitions follow the paper text. The random choices (fixed on 2026-11-02 before this module was
    run): every psm_att call uses a fresh RandomState(config.SEED); the within-stratum draws go in M5's order
    (q_ts, revid), one draw with RandomState(config.SEED) and each newcomer's own follow-up eligibility, 20 draws with
    RandomState(123) and the main eligibility flag, as M5 does for the main design.
    The paper's first version of c and d was computed with code that is not in this repository; point estimates and
    counts of c are reproduced exactly, intervals of c and the estimates of d differ by up to 0.002.

3. Inputs
    data/analytic/psm_dataset_legacy_layout.npz; data/interim/m5/sample.parquet; data/interim/m3/mentee_edits.

4. Outputs
    results/m8/anchor_lag_strata.csv, anchor_reply_timing.csv, anchor_landmark.csv, anchor_within_stratum.csv

5. Key parameters
    LANDMARKS_H = (12, 24, 48, 72, 168); window 14 days; config.N_BOOT (500); N_MULTI (20, from M5).

6. Possible errors and how they are handled
    The primary outcome is recomputed from the edits and the M5 anchors; if it differs from oc_primary for anyone,
    the run stops (the edit counting would not match M5).

7. How to run
    python -m pipeline.m8_analysis.anchor_checks

8. How to check
    The exploration script of 2026-11-02 (research notes) gave the same numbers; the paper number check
    (pipeline.m9_paper.check_camera_ready) compares every cell with the paper.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import config
from pipeline.common.manifest import write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m5_sample.build_sample import LAG_MAX_DAYS, N_MULTI
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis.run_analysis import add_followup_masks, covariates, load, psm_att, subset

OUT = config.RESULTS / "m8"
LANDMARKS_H = (12, 24, 48, 72, 168)
DAY = np.timedelta64(86400, "s")


class ArticleEdits:
    """Article edit times of the sample, for counting edits in (start, start + h] per newcomer."""

    def __init__(self, uids: np.ndarray):
        e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
            columns=["uid", "timestamp", "ns"], filter=(ds.field("uid").isin(list(set(uids)))) & (ds.field("ns") == 0)
        ).to_pandas()
        pos = pd.Series(np.arange(len(uids)), index=uids)
        assert pos.index.is_unique
        sec = pd.to_datetime(e.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]").astype(np.int64)
        self.n = len(uids)
        self.key = np.sort(e.uid.map(pos).to_numpy().astype(np.int64) * 10**10 + sec)   # newcomer, then time

    def count(self, start: np.ndarray, hours: float) -> np.ndarray:
        s0 = start.astype("datetime64[s]").astype(np.int64)
        base = np.arange(self.n, dtype=np.int64) * 10**10
        return (np.searchsorted(self.key, base + s0 + int(hours * 3600), side="right")
                - np.searchsorted(self.key, base + s0, side="right"))


def main(argv: list[str] | None = None) -> None:
    base_parser("Checks of the control window start: lag by stratum, reply timing, landmarks, within-stratum draws.").parse_args(argv)
    log = setup_logging("m8_anchor_checks")
    t0 = time.time()
    Dall = add_followup_masks(load())
    D = subset(Dall, Dall["in_sample"])
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet")
    S = s.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[D["cid_order"].astype(np.int64)]
    n = len(S)
    assert (S.treat.to_numpy() == D["y_treat"]).all()
    q = pd.to_datetime(S.q_ts, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    anchor = pd.to_datetime(S.anchor, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    lag = S.reply_lag_days.to_numpy(dtype=float)
    treat = S.treat.to_numpy() == 1
    cutoff = np.datetime64(pd.Timestamp(config.DATA_CUTOFF).tz_localize(None), "s")
    E = ArticleEdits(S.mentee_uid.astype(np.int64).to_numpy())
    prim = (E.count(anchor, 14 * 24) > 0).astype(float)
    assert (prim == D["oc_primary"].astype(float)).all(), "primary outcome not reproduced from edits and anchors"
    main_flag = D["keep__primary"].astype(bool)
    d = D["y_treat"].astype(float)
    X = covariates(D)
    st = est.make_strata(est.fit_propensity(X[:, X.std(axis=0) > 0], d).e_hat, config.N_STRATA)
    pool = treat & (lag >= 0) & (lag < LAG_MAX_DAYS)

    # a. reply lag by stratum
    a_rows = []
    for k in range(int(st.max()) + 1):
        h = lag[pool & (st == k)] * 24
        p25, p50, p75 = np.percentile(h, [25, 50, 75])
        a_rows.append({"stratum": k + 1, "treated": int(len(h)), "p25_hours": p25, "median_hours": p50, "p75_hours": p75})
    A = pd.DataFrame(a_rows)

    # b. reply timing vs editing in the first 24 hours
    no24 = ~(treat & (lag <= 1))
    ed = E.count(q, 24) > 0
    rep = treat & (lag > 1) & (lag <= 7)
    yy, dd = rep[no24].astype(float), ed[no24].astype(float)
    diff = est.att_stratified(yy, dd, st[no24]).att
    lo, hi = est.percentile_ci(est.bootstrap_att(yy, dd, st[no24], est.ClusterIndex(D["mentee_ids"][no24]),
                                                 np.random.RandomState(config.SEED), config.N_BOOT))
    B = pd.DataFrame([{"n_no_reply_24h": int(no24.sum()), "n_edited_24h": int((no24 & ed).sum()),
                       "reply_24_168h_edited": rep[no24 & ed].mean(), "reply_24_168h_not_edited": rep[no24 & ~ed].mean(),
                       "diff_within_strata": diff, "CI_lo": lo, "CI_hi": hi, "B": config.N_BOOT}])

    # c. landmarks
    c_rows = []
    for L in LANDMARKS_H:
        start = q + np.timedelta64(L * 3600, "s")
        elig = start + 14 * DAY <= cutoff
        y = (E.count(start, 14 * 24) > 0).astype(float)
        dL = (treat & (lag <= L / 24)).astype(float)
        variants = [("no reply within L", elig)]
        if L == 48:
            variants.append(("never replied", elig & ~(treat & (lag > L / 24))))
        for name, m in variants:
            r = psm_att(subset(D, m), y[m], dL[m], np.random.RandomState(config.SEED), f"{L} h, controls: {name}")
            c_rows.append({"L_hours": L, "controls": name, **r})
            log.info("[landmark] %3d h %-18s N=%d treated=%d ATT=%+.4f [%+.4f, %+.4f]", L, name, r["N"], r["Treated"],
                     r["ATT"], r["CI_lo"], r["CI_hi"])

    # d. control lag drawn within the PS stratum
    order = np.lexsort((S.index.to_numpy(), q))
    pools = {k: lag[pool & (st == k)] for k in range(int(st.max()) + 1)}

    def draw(rng) -> np.ndarray:
        out = np.zeros(n)
        for j in order:
            if not treat[j]:
                out[j] = rng.choice(pools[st[j]])
        return out

    rng = np.random.RandomState(config.SEED)
    s1 = np.where(treat, anchor, q + (draw(rng) * 86400).astype("timedelta64[s]"))
    y1 = (E.count(s1, 14 * 24) > 0).astype(float)
    el1 = np.where(treat, main_flag, s1 + 14 * DAY <= cutoff)
    rng = np.random.RandomState(123)
    ys = np.zeros(n)
    for _ in range(N_MULTI):
        sk = np.where(treat, anchor, q + (draw(rng) * 86400).astype("timedelta64[s]"))
        ys += E.count(sk, 14 * 24) > 0
    ym = np.where(treat, prim, ys / N_MULTI)
    d_rows = []
    for name, y, m in (("one draw within the PS stratum", y1, el1),
                       (f"mean of {N_MULTI} draws within the PS stratum", ym, main_flag)):
        r = psm_att(subset(D, m), y[m], d[m], np.random.RandomState(config.SEED), name)
        d_rows.append(r)
        log.info("[within stratum] %-40s N=%d ATT=%+.4f [%+.4f, %+.4f]", name, r["N"], r["ATT"], r["CI_lo"], r["CI_hi"])

    files = {"anchor_lag_strata.csv": A, "anchor_reply_timing.csv": B, "anchor_landmark.csv": pd.DataFrame(c_rows),
             "anchor_within_stratum.csv": pd.DataFrame(d_rows)}
    for f, df in files.items():
        df.to_csv(OUT / f, index=False)
    write_manifest("m8_anchor_checks", inputs=[config.ANALYTIC / "psm_dataset_legacy_layout.npz",
                                               config.INTERIM / "m5" / "sample.parquet"],
                   outputs=[OUT / f for f in files], params={"n_boot": config.N_BOOT, "seed": config.SEED,
                                                             "landmarks_h": list(LANDMARKS_H), "n_multi": N_MULTI},
                   counts={"N": n, "lag_pool": int(pool.sum())}, started=t0)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
