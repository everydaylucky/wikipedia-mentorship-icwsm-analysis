"""
quality.py: edit quality after a reply (camera-ready, paper paragraph "Edit quality" and Appendix "Edit Quality")

Module: M8 (analysis added for the ICWSM 2027 camera-ready)

1. What it does
    An article (namespace 0) edit counts as rejected if another editor reverted it within 48 hours (the window of
    Warncke-Wang et al. 2023). Every newcomer is placed in one of three groups by their article edits in the 14 days
    after the anchor (the reply, or the pseudo-reply time for controls):
        group 1  no article edit
        group 2  at least CUT of the edits kept (CUT = 80%, about the average rejected share of control editors)
        group 3  less than CUT kept
    and the ATT on the share in each group is estimated with the main design (propensity score with 5-fold
    cross-fitting, 10 strata, 500 bootstrap draws clustered by newcomer).
    Edits that the newcomer reverted themself within 48 hours (the reverting revision is one of their own edits) are
    left out together with that reverting edit; a newcomer with no other article edit in the window goes to group 1.

2. Why it is designed this way
    The rejected share exists only for newcomers who edited, and a reply changes who edits, so the groups cover every
    newcomer. The cut is tied to the control editors' average rejected share. Other cuts, and other treatments of
    self-reverts and of tagged edits without a matched reverting revision, are reported as checks.

3. Inputs
    data/analytic/psm_dataset_legacy_layout.npz (M7); data/interim/m5/sample.parquet; data/interim/m3/mentee_edits/;
    data/interim/m3/revert_times.parquet (M3, revert_times.py --phase outcomes, then --phase build).

4. Outputs (results/m8/, or results/m8_b<B>/ with --boot)
    quality_cuts.csv           group shares and ATT for each cut (main: 80%)
    quality_self_versions.csv  the 80% cut under other treatments of self-reverts and unmatched tagged edits
    quality_revert_stats.csv   counts used in the appendix text

5. Key parameters
    WINDOW = 14 days, REVERT_WINDOW = 48 hours, CUT = 0.8; config.N_BOOT; config.SEED.

6. Possible errors and how they are handled
    Only newcomers whose 14-day window ends at least 48 hours before the data cutoff are used.

7. How to run
    python -m pipeline.m8_analysis.quality

8. How to check
    checks/m8_check.ipynb (section 7.8).
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
from pipeline.m8_analysis.run_analysis import analysis_sample, covariates, subset

WINDOW = np.timedelta64(14, "D")
REVERT_WINDOW = np.timedelta64(48, "h")
CUT = 0.8
CUTS = [("at least one edit", None), ("50%", 0.5), ("60%", 0.6), ("70%", 0.7), ("80%", 0.8), ("90%", 0.9), ("all edits", 1.0)]


def window_edits(D: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Article edits in each newcomer's 14-day window, with revert information; and the anchors."""
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet", columns=["first_question_revid", "mentee_uid", "anchor"])
    s = s.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[D["cid_order"].astype(np.int64)]
    anchors = pd.to_datetime(s.anchor, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    uids = s.mentee_uid.astype(np.int64).to_numpy()
    e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "revid", "timestamp", "ns", "tags"],
        filter=(ds.field("uid").isin(list(uids))) & (ds.field("ns") == 0)).to_pandas()
    own = set(zip(e.uid.astype(np.int64), e.revid.astype(np.int64)))
    e["ts"] = pd.to_datetime(e.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    e["i"] = e.uid.map({u: k for k, u in enumerate(uids)}).to_numpy()
    e = e[(e.ts.to_numpy() > anchors[e.i]) & (e.ts.to_numpy() <= anchors[e.i] + WINDOW)].copy()
    e["tagged"] = e.tags.map(lambda t: "mw-reverted" in list(t))
    rv = pd.read_parquet(config.INTERIM / "m3" / "revert_times.parquet", columns=["revid", "revert_revid", "revert_ts"])
    rv = rv[rv.revert_ts.notna()].drop_duplicates("revid").set_index("revid")
    e = e.join(rv, on="revid")
    rts = pd.to_datetime(e.revert_ts, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    e["matched"] = ~np.isnat(rts)
    e["within48"] = e.matched & (rts - e.ts.to_numpy() <= REVERT_WINDOW)
    e["lag_h"] = (rts - e.ts.to_numpy()) / np.timedelta64(1, "h")
    e["self"] = e.within48 & np.array([(int(u), int(r)) in own if r == r else False
                                       for u, r in zip(e.uid, e.revert_revid)])
    return e, anchors


def per_newcomer(e: pd.DataFrame, n: int, self_reverts: str, unmatched: str) -> pd.DataFrame:
    """Kept and rejected article edits per newcomer.

    self_reverts: "leave out" (main: the self-reverted edit and the newcomer's reverting edit are dropped),
                  "rejected" or "kept". unmatched: tagged edits without a matched reverting revision, "rejected" or "kept".
    """
    rejected = e.tagged & (e.within48 | (~e.matched & (unmatched == "rejected")))
    drop = np.zeros(len(e), bool)
    if self_reverts == "kept":
        rejected &= ~e.self
    elif self_reverts == "leave out":
        reverting = set(e.revert_revid[e.self].astype(np.int64))
        drop = (e.self | e.revid.astype(np.int64).isin(reverting)).to_numpy()
    k = ~drop
    n_judged = np.bincount(e.i[k], minlength=n)
    n_rej = np.bincount(e.i[k & rejected.to_numpy()], minlength=n)
    return pd.DataFrame({"n": n_judged, "n_rejected": n_rej, "n_kept": n_judged - n_rej})


def groups(c: pd.DataFrame, cut: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indicators of groups 1, 2 and 3 for one cut (None = at least one kept edit)."""
    judged = c.n.to_numpy() > 0
    kept = c.n_kept.to_numpy()
    share = np.where(judged, kept / np.maximum(c.n.to_numpy(), 1), 0.0)
    g2 = (kept > 0) if cut is None else judged & (share >= cut - 1e-12)
    return ~judged, g2, judged & ~g2


def run(D: dict, log) -> dict[str, pd.DataFrame]:
    e, anchors = window_edits(D)
    cutoff = np.datetime64(pd.Timestamp(config.DATA_CUTOFF).tz_localize(None), "s")
    full = anchors + WINDOW + REVERT_WINDOW <= cutoff
    S = subset(D, full)
    e = e[full[e.i]].copy()
    e["i"] = np.cumsum(full)[e.i] - 1
    n = int(full.sum())
    X, d = covariates(S), S["y_treat"].astype(float)
    st = est.make_strata(est.fit_propensity(X[:, X.std(axis=0) > 0], d).e_hat, config.N_STRATA)
    cl = est.ClusterIndex(S["mentee_ids"])
    n1 = np.array([((st == k) & (d == 1)).sum() for k in range(st.max() + 1)])

    def fit(y):
        y = np.asarray(y, float)
        lo, hi = est.percentile_ci(est.bootstrap_att(y, d, st, cl, np.random.RandomState(config.SEED), config.N_BOOT))
        ctrl = sum(n1[k] * y[(st == k) & (d == 0)].mean() for k in range(len(n1))) / n1.sum()
        return {"Ctrl_reweighted": ctrl, "Treat": float(y[d == 1].mean()), "ATT": est.att_stratified(y, d, st).att,
                "CI_lo": lo, "CI_hi": hi, "Sig": "*" if est.is_significant(lo, hi) else ""}

    def rejected_share(c):
        j = (c.n.to_numpy() > 0) & (d == 0)
        return float((c.n_rejected.to_numpy()[j] / c.n.to_numpy()[j]).mean())

    main = per_newcomer(e, n, "leave out", "rejected")
    rows = []
    for label, cut in CUTS:
        g1, g2, g3 = groups(main, cut)
        rows.append({"cut": label, "N": n, **{f"g1_{k}": v for k, v in fit(g1).items()},
                     **{f"g2_{k}": v for k, v in fit(g2).items()}, **{f"g3_{k}": v for k, v in fit(g3).items()}})
        log.info("[quality] cut %-17s group 2 ATT=%+.4f [%+.4f, %+.4f]  group 3 ATT=%+.4f [%+.4f, %+.4f]", label,
                 rows[-1]["g2_ATT"], rows[-1]["g2_CI_lo"], rows[-1]["g2_CI_hi"], rows[-1]["g3_ATT"], rows[-1]["g3_CI_lo"],
                 rows[-1]["g3_CI_hi"])
    versions = []
    for sr, um in [("leave out", "rejected"), ("leave out", "kept"), ("rejected", "rejected"), ("kept", "rejected")]:
        c = main if (sr, um) == ("leave out", "rejected") else per_newcomer(e, n, sr, um)
        _, g2, g3 = groups(c, CUT)
        a2, a3 = fit(g2), fit(g3)
        versions.append({"self_reverts": sr, "unmatched": um, "ctrl_mean_rejected_share": rejected_share(c),
                         **{f"g2_{k}": v for k, v in a2.items()}, **{f"g3_{k}": v for k, v in a3.items()}})
    r48 = e[e.within48]
    stats = pd.DataFrame([{
        "N": n, "treated": int(d.sum()), "window_article_edits": len(e), "tagged": int(e.tagged.sum()),
        "tagged_matched": int((e.tagged & e.matched).sum()), "reverted_within_48h": len(r48),
        "reverted_by_newcomer": int(r48.self.sum()), "median_minutes_self": float(r48.lag_h[r48.self].median() * 60),
        "median_minutes_others": float(r48.lag_h[~r48.self].median() * 60),
        "share_self_within_10_min": float((r48.lag_h[r48.self] <= 1 / 6).mean()),
        "newcomers_only_self_reverted_pairs": int(((np.bincount(e.i, minlength=n) > 0) & (main.n.to_numpy() == 0)).sum()),
        "ctrl_mean_rejected_share": rejected_share(main)}])
    log.info("[quality] %s", stats.iloc[0].to_dict())
    return {"quality_cuts": pd.DataFrame(rows), "quality_self_versions": pd.DataFrame(versions), "quality_revert_stats": stats}


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("M8 edit quality (48-hour reverts, three groups).")
    ap.add_argument("--boot", type=int, default=config.N_BOOT, help="bootstrap draws (final 500; 100 for a quick check)")
    args = ap.parse_args(argv)
    config.N_BOOT = args.boot
    debug = config.N_BOOT != config.N_BOOT_FINAL
    out_dir = config.RESULTS / ("m8" if not debug else f"m8_b{config.N_BOOT}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging("m8_quality")
    t0 = time.time()
    check_upstream("m8_quality", ["m7"])
    res = run(analysis_sample(), log)
    for k, v in res.items():
        v.to_csv(out_dir / f"{k}.csv", index=False)
    write_manifest("m8_quality" if not debug else f"m8_quality_b{config.N_BOOT}",
                   inputs=[config.ANALYTIC / "psm_dataset_legacy_layout.npz", config.INTERIM / "m3" / "revert_times.parquet"],
                   outputs=[out_dir / f"{k}.csv" for k in res], params={"n_boot": config.N_BOOT, "seed": config.SEED, "cut": CUT},
                   counts={k: len(v) for k, v in res.items()}, started=t0)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
