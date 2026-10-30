"""
full_bootstrap.py: robustness check of the confidence intervals of Table 4 with a full bootstrap

Module: M8 (camera-ready check)

1. What it does
    The main analysis (run_analysis.py) estimates the propensity score and the 10 strata once, and then resamples
    newcomers with those strata held fixed. That interval leaves out the uncertainty from estimating the propensity
    score. Here every bootstrap replicate repeats the whole procedure on the resampled newcomers:
        1. draw the analysis sample's newcomers with replacement (one row per newcomer, so this is the newcomer-level cluster bootstrap);
        2. fit the propensity score again with 5-fold cross-fitting; copies of the same newcomer are kept in the same
           fold (StratifiedGroupKFold), so a newcomer is never used to predict their own copy;
        3. cut the new scores into 10 strata and compute the stratified ATT for the six Table 4 outcomes.
    It reports the point estimate of the main analysis, the fixed-strata interval, and the full-bootstrap interval.
    As in the main analysis, each outcome uses only the people whose window ends before the cutoff (I-10).

2. Why it is designed this way
    A reviewer may ask whether the intervals account for the estimated propensity score. Refitting in each replicate
    is the plain, assumption-light answer. One cross-fitted fit takes about half a second, so B = 500 is cheap.

4. Outputs
    results/m8/full_bootstrap.csv

7. How to run
    python -m pipeline.m8_analysis.full_bootstrap            # B = 500 (config.N_BOOT)
    python -m pipeline.m8_analysis.full_bootstrap --boot 50  # quick test
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import config
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis.legacy_parity import DIMS_S13
from pipeline.m8_analysis.run_analysis import analysis_sample, covariates

OUT = config.RESULTS / "m8"


def _replicate(args) -> list[float]:
    """One full-bootstrap replicate: resample, refit the PS, restratify, ATT for every outcome."""
    seed, X, d, Y, K = args
    rng = np.random.RandomState(seed)
    n = len(d)
    idx = rng.randint(0, n, n)
    Xb, db, Yb, Kb = X[idx], d[idx], Y[idx], K[idx]
    keep = Xb.std(axis=0) > 0
    Xs = StandardScaler().fit_transform(Xb[:, keep])
    e = np.empty(n)
    cv = StratifiedGroupKFold(n_splits=config.PS_FOLDS, shuffle=True, random_state=seed % (2 ** 31))
    for tr, te in cv.split(Xs, db, groups=idx):
        lr = LogisticRegression(C=config.PS_C, max_iter=2000, solver="lbfgs", random_state=42)
        lr.fit(Xs[tr], db[tr])
        e[te] = lr.predict_proba(Xs[te])[:, 1]
    st = est.make_strata(e, config.N_STRATA)
    return [est.att_stratified(Yb[Kb[:, j], j], db[Kb[:, j]], st[Kb[:, j]]).att for j in range(Yb.shape[1])]


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("Full bootstrap (PS refitted in every replicate) for Table 4.")
    ap.add_argument("--boot", type=int, default=config.N_BOOT)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    args = ap.parse_args(argv)
    log = setup_logging("m8_full_bootstrap")
    t0 = time.time()
    D = analysis_sample()
    X, d = covariates(D), D["y_treat"].astype(float)
    Y = np.column_stack([D[f"oc_{k}"].astype(float) for _, k in DIMS_S13])
    K = np.column_stack([D[f"keep__{k}"] for _, k in DIMS_S13])
    ref = pd.read_csv(OUT / "B_main_results.csv").set_index("Outcome")   # the main analysis (fixed strata)

    # point estimate and fixed-strata interval, exactly as in the main analysis
    keep = X.std(axis=0) > 0
    st = est.make_strata(est.fit_propensity(X[:, keep], d).e_hat, config.N_STRATA)
    seeds = np.random.RandomState(config.SEED).randint(0, 2 ** 31 - 1, args.boot)
    tasks = [(int(s), X, d, Y, K) for s in seeds]
    with ProcessPoolExecutor(args.jobs) as ex:
        draws = np.array(list(ex.map(_replicate, tasks, chunksize=4)))
    rows = []
    for j, (name, key) in enumerate(DIMS_S13):
        k = K[:, j]
        att = est.att_stratified(Y[k, j], d[k], st[k]).att
        fixed = (float(ref.at[name, "CI_lo"]), float(ref.at[name, "CI_hi"]))
        assert abs(ref.at[name, "ATT"] - att) < 1e-12, "point estimate differs from the main analysis"
        b = draws[:, j][~np.isnan(draws[:, j])]
        full = est.percentile_ci(b)
        rows.append({"Outcome": name, "ATT": att, "fixed_CI_lo": fixed[0], "fixed_CI_hi": fixed[1],
                     "full_CI_lo": full[0], "full_CI_hi": full[1], "full_SE": float(b.std(ddof=1)),
                     "full_sig": "*" if est.is_significant(*full) else "", "B": len(b)})
        log.info("%-24s ATT=%+.4f fixed [%+.4f, %+.4f]  full [%+.4f, %+.4f] %s", name, att, *fixed, *full,
                 rows[-1]["full_sig"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "full_bootstrap.csv", index=False)
    log.info("done in %.0f s (B = %d, %d processes)", time.time() - t0, args.boot, args.jobs)


if __name__ == "__main__":
    main(sys.argv[1:])
