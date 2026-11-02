"""
mentor_side.py: two checks for leaving mentor-side variables out of the propensity score model

Module: M8 (camera-ready check; moved from the exploratory script mentor_side_checks.py, checks 2 and 4, on 2026-10-31)

1. What it does
    The main PS model uses newcomer covariates only. Leaving out the mentor's reply habits is justified if (a) the
    mentor a newcomer got is unrelated to the newcomer, and (b) the mentor affects editing only through the reply.
    Mentor variable: the assigned mentor's past reply rate at the question (m_prior_reply_rate, M7), used when the
    mentor had at least 5 earlier questions. Within each question month, mentors are split into thirds by this rate.
        Balance (a): top vs bottom third, actual reply share of the two groups, and the month-adjusted standardized
            mean difference (SMD) of the 105 covariates (the 164 PS covariates without the 59 year-month
            indicators). The same SMDs among newcomers without a reply, the group the negative control uses.
        Negative control (b): among newcomers without a reply, the slope of the primary outcome on the mentor's past
            reply rate (per 10 points), adjusted for the 105 covariates and question-month indicators (OLS); 95%
            interval from a mentor-cluster bootstrap. If the mentor affected editing in other ways, non-replied
            newcomers of responsive mentors would edit more. Also reported: the exploratory version (within main-PS
            stratum x question month, no covariates).

2. Why it is designed this way
    The specification is the one of the exploratory script (same thirds, threshold, scale, strata, seed), fixed on
    2026-10-31 before this module was run, so the pipeline does not choose settings after seeing results.
    Within-month comparisons remove changes of the mentor pool and of the newcomer population over time.
    One change was made after the test run showed balance among non-replied newcomers, before the new estimate was
    computed: non-replied newcomers of responsive mentors asked fewer substantive questions (largest |SMD| 0.41),
    because responsive mentors leave mostly empty questions unanswered. PS strata balance replied vs not replied,
    not mentor groups among the non-replied, so the main negative control adjusts for the covariates directly.

3. Inputs
    data/analytic/psm_dataset_legacy_layout.npz (through analysis_sample); results/m8/B_main_results.csv (check that
    the strata are those of the main analysis).

4. Outputs
    results/m8/mentor_side_summary.csv   one row per quantity (name, value)
    results/m8/mentor_side_smd.csv       per covariate: SMD top vs bottom third, all askers and newcomers without a reply

5. Key parameters
    MIN_PRIOR = 5 earlier questions; thirds within question month; bootstrap B = config.N_BOOT (500); seed 4 for the
    adjusted negative control, 3 for the exploratory version.

6. Possible errors and how they are handled
    The PS is refitted as in the main analysis; if the primary ATT differs from B_main_results.csv the run stops,
    because the negative control must use the main analysis' strata.

7. How to run
    python -m pipeline.m8_analysis.mentor_side             # B = 500
    python -m pipeline.m8_analysis.mentor_side --boot 50   # quick test (does not write results)

8. How to check
    checks/m8_check.ipynb, section "Mentor-side checks": recomputes the reply shares, the largest SMD and the
    negative-control slope with separate code and compares them with mentor_side_summary.csv.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import config
from pipeline.common.manifest import write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis import legacy_parity as lp
from pipeline.m8_analysis.run_analysis import analysis_sample, covariates

OUT = config.RESULTS / "m8"
MIN_PRIOR = 5
SEED_NC = 3       # exploratory version (stratum x month)
SEED_ADJ = 4      # adjusted version (main)


def thirds(rate: np.ndarray, valid: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Third of the mentor's past reply rate within the question month: 0 bottom, 1 middle, 2 top, -1 not used."""
    out = np.full(len(rate), -1)
    pct = pd.Series(rate[valid]).groupby(month[valid]).rank(pct=True, method="average").to_numpy()
    out[np.where(valid)[0]] = np.where(pct <= 1 / 3, 0, np.where(pct <= 2 / 3, 1, 2))
    return out


def month_smd(X: np.ndarray, hi: np.ndarray, month: np.ndarray) -> np.ndarray:
    """SMD of each column between hi and not hi after removing month means; pooled sd of the two raw groups."""
    Xr = (pd.DataFrame(X) - pd.DataFrame(X).groupby(month).transform("mean")).to_numpy()
    sd = np.sqrt((X[hi].var(0) + X[~hi].var(0)) / 2)
    return np.where(sd > 0, (Xr[hi].mean(0) - Xr[~hi].mean(0)) / np.where(sd > 0, sd, 1), 0.0)


def nc_slope(y: np.ndarray, r: np.ndarray, cell: np.ndarray, w: np.ndarray) -> float:
    """Weighted slope of y on r after removing weighted cell means."""
    df = pd.DataFrame({"y": y, "r": r, "w": w, "g": cell})
    df = df[df.w > 0]
    sw = df.groupby("g").w.transform("sum")
    ym = (df.w * df.y).groupby(df.g).transform("sum") / sw
    rm = (df.w * df.r).groupby(df.g).transform("sum") / sw
    yy, rr = df.y - ym, df.r - rm
    return float((df.w * yy * rr).sum() / (df.w * rr * rr).sum())


def adj_slope(y: np.ndarray, r: np.ndarray, Z: np.ndarray, w: np.ndarray) -> float:
    """Weighted OLS coefficient of r in y ~ r + Z (Z holds a constant, covariates and month indicators)."""
    s = np.sqrt(w)[:, None]
    A = np.hstack([r[:, None], Z]) * s
    return float(np.linalg.lstsq(A, y * s.ravel(), rcond=None)[0][0])


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("Mentor-side checks: balance by the mentor's reply rate and a negative control.")
    ap.add_argument("--boot", type=int, default=config.N_BOOT)
    args = ap.parse_args(argv)
    log = setup_logging("m8_mentor_side")
    t0 = time.time()

    D = analysis_sample()
    X, d = covariates(D), np.asarray(D["y_treat"], float)
    y = np.asarray(D["oc_primary"], float)
    names = lp.covariate_names(D)
    month = np.asarray(D["q_year_month"]).astype(str)
    M = pd.DataFrame(D["X_M"], columns=list(D["M_cols"]))
    rate = M.m_prior_reply_rate.to_numpy()
    valid = (M.m_prior_mentee_count >= MIN_PRIOR).to_numpy()

    # strata of the main analysis (same fit as run_analysis and full_bootstrap)
    keep = X.std(axis=0) > 0
    st = est.make_strata(est.fit_propensity(X[:, keep], d).e_hat, config.N_STRATA)
    ref = pd.read_csv(OUT / "B_main_results.csv").set_index("Outcome")
    att = est.att_stratified(y, d, st).att
    assert abs(ref.ATT.iloc[0] - att) < 1e-12, "strata differ from the main analysis"

    # balance
    t = thirds(rate, valid, month)
    tc = set(D["temporal_cols"])
    cols = [i for i, n in enumerate(names) if not (n in tc and str(n).startswith("ym_"))]
    k = (t == 0) | (t == 2)
    smd_all = month_smd(X[k][:, cols], t[k] == 2, month[k])
    k0 = k & (d == 0)
    smd_ctl = month_smd(X[k0][:, cols], t[k0] == 2, month[k0])
    dm = d - pd.Series(d).groupby(month).transform("mean").to_numpy()

    # negative control
    m = (d == 0) & valid
    cell = (pd.Series(st[m]).astype(str) + "|" + pd.Series(month[m])).to_numpy()
    yc, rc = y[m], rate[m] * 10
    b0 = nc_slope(yc, rc, cell, np.ones(m.sum()))
    cl = est.ClusterIndex(np.asarray(D["mentor_ids"])[m])
    rng = np.random.RandomState(SEED_NC)
    draws = np.array([nc_slope(yc, rc, cell, cl.draw_weights(rng)) for _ in range(args.boot)])
    lo, hi = est.percentile_ci(draws)
    gap = float(rate[t == 2].mean() - rate[t == 0].mean()) * 10   # bottom-to-top gap in units of 10 points

    # main negative control: adjusted for the covariates and question month
    Xc = X[m][:, cols]
    Xc = Xc[:, Xc.std(axis=0) > 0]
    Z = np.hstack([np.ones((m.sum(), 1)), Xc, pd.get_dummies(month[m], drop_first=True).to_numpy(float)])
    a0 = adj_slope(yc, rc, Z, np.ones(m.sum()))
    rng = np.random.RandomState(SEED_ADJ)
    adraws = np.array([adj_slope(yc, rc, Z, cl.draw_weights(rng)) for _ in range(args.boot)])
    alo, ahi = est.percentile_ci(adraws)

    s = {
        "N": len(d), "N_mentor_rate_defined": int(valid.sum()),
        "n_bottom": int((t == 0).sum()), "n_top": int((t == 2).sum()),
        "past_rate_bottom": float(rate[t == 0].mean()), "past_rate_top": float(rate[t == 2].mean()),
        "reply_share_bottom": float(d[t == 0].mean()), "reply_share_top": float(d[t == 2].mean()),
        "reply_share_gap_within_month": float(dm[t == 2].mean() - dm[t == 0].mean()),
        "n_covariates": len(cols),
        "max_abs_smd_askers": float(np.abs(smd_all).max()), "n_abs_smd_gt_0.1_askers": int((np.abs(smd_all) > 0.1).sum()),
        "n_bottom_no_reply": int(((t == 0) & (d == 0)).sum()), "n_top_no_reply": int(((t == 2) & (d == 0)).sum()),
        "max_abs_smd_no_reply": float(np.abs(smd_ctl).max()), "n_abs_smd_gt_0.1_no_reply": int((np.abs(smd_ctl) > 0.1).sum()),
        "nc_n": int(m.sum()), "nc_outcome_mean": float(yc.mean()),
        "nc_B": args.boot, "nc_gap_in_10pt_units": gap,
        "nc_adj_slope_per_10pt": a0, "nc_adj_CI_lo": alo, "nc_adj_CI_hi": ahi,
        "nc_adj_implied_bottom_to_top": a0 * gap, "nc_adj_implied_CI_lo": alo * gap, "nc_adj_implied_CI_hi": ahi * gap,
        "nc_strata_slope_per_10pt": b0, "nc_strata_CI_lo": lo, "nc_strata_CI_hi": hi,
    }
    for key, v in s.items():
        log.info("%-34s %s", key, f"{v:.4f}" if isinstance(v, float) else v)
    if args.boot != config.N_BOOT:
        log.warning("test run: B = %d, results not written", args.boot)
        return
    summary = OUT / "mentor_side_summary.csv"
    smd = OUT / "mentor_side_smd.csv"
    pd.DataFrame({"name": list(s), "value": list(s.values())}).to_csv(summary, index=False)
    pd.DataFrame({"covariate": [names[i] for i in cols], "smd_askers": smd_all, "smd_no_reply": smd_ctl}).to_csv(smd, index=False)
    write_manifest("m8_mentor_side", inputs=[config.ANALYTIC / "psm_dataset_legacy_layout.npz", OUT / "B_main_results.csv"],
                   outputs=[summary, smd], params={"n_boot": args.boot, "min_prior": MIN_PRIOR, "seed_adj": SEED_ADJ, "seed_strata": SEED_NC},
                   counts={"N": len(d), "nc_n": int(m.sum())}, started=t0)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
