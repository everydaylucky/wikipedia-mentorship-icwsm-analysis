"""
ps_models.py: compare the propensity score model of the main analysis with other regularization strengths and models

Module: M8 (camera-ready check; moved from the exploratory comparison of 2026-11-02, run on the laptop copy)

1. What it does
    The main analysis fits the propensity score with an L2-regularized logistic regression at C = 1 (config.PS_C).
    Here the same sample, the same 164 covariates, the same 5-fold cross-fitting and the same 10 strata are used with
    seven propensity models:
        L2 logistic regression with C = 0.01, 0.1, 1 (main) and 10;
        random forest with at least 50 or at least 5 people per leaf;
        gradient boosting (histogram-based, sklearn defaults).
    For each model it reports prediction (AUC, Brier score, log loss), overlap (range of the scores, people outside
    common support, controls in the top four strata), balance after stratification (largest and mean |SMD| over the
    covariates, number above 0.1) and the primary ATT with its 95% interval.

2. Why it is designed this way
    The propensity model is judged by balance, not by prediction, so balance and the ATT are reported next to AUC.
    Everything except the model is held fixed, so differences between rows come from the model only.
    Every model uses a fresh RandomState(config.SEED) for its bootstrap, so all rows use the same resampled newcomers.
    Because of that the interval of the C = 1 row is not the one in B_main_results.csv (there the same random
    stream is first used by the feature ablation); the point estimate is the same, and the run stops if it is not.
    The model list (C grid, the two random forest leaf sizes, gradient boosting) is that of the exploratory
    comparison. That comparison's gradient boosting settings were not kept, so here it uses sklearn's defaults; its
    ATT (+2.0pp) differs from the exploratory one (+2.2pp), while all other rows agree to within 0.1pp.

3. Inputs
    data/analytic/psm_dataset_legacy_layout.npz (through analysis_sample); results/m8/B_main_results.csv (check that
    the C = 1 row reproduces the main ATT).

4. Outputs
    results/m8/ps_model_comparison.csv   one row per model
    results/m8/ps_model_balance.csv      per model and covariate: |SMD| before and after stratification (ATT weights)

5. Key parameters
    C grid (0.01, 0.1, 1, 10); random forest 500 trees, max_features = sqrt, min_samples_leaf 50 or 5;
    gradient boosting HistGradientBoostingClassifier defaults; random_state 42 for every model;
    config.PS_FOLDS, config.N_STRATA, config.SEED, B = config.N_BOOT (500).
    SMD after stratification uses ATT weights (w_k = n_1k / n_1, I-35); the legacy weights are also reported.

6. Possible errors and how they are handled
    If the C = 1 row does not give the primary ATT of B_main_results.csv, the run stops.
    BLAS threads change floating-point order and can move newcomers at stratum borders; run with one BLAS thread
    (OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1), as results/m8 was made.

7. How to run
    python -m pipeline.m8_analysis.ps_models             # B = 500
    python -m pipeline.m8_analysis.ps_models --boot 50   # quick test (does not write results)

8. How to check
    The C = 1 row must equal the main analysis: AUC 0.626, ATT of B_main_results.csv, largest stratified |SMD| of
    balance_att_weighting.csv. An independent recomputation with separate code (own cross-fitting loop, groupby
    strata, own SMD and ATT) of the C = 1 and gradient boosting rows matched AUC, ATT and |SMD| exactly on 2026-11-02
    (papers/icwsm2027_wikipedia_mentor/Date/2026-11/2026-11-02/check_ps_models.py in the research hub); a section in
    checks/m8_check.ipynb is still to be added.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import config
from pipeline.common.manifest import write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis import legacy_parity as lp
from pipeline.m8_analysis.run_analysis import analysis_sample, covariates

OUT = config.RESULTS / "m8"
MAIN = "L2 logit, C = 1 (main)"


def models(jobs: int) -> list[tuple[str, object]]:
    """The seven propensity models, in the order of the output table."""
    lr = lambda C: LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)  # as fit_propensity
    rf = lambda leaf: RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_features="sqrt",
                                             n_jobs=jobs, random_state=42)
    return [(MAIN, lr(1.0)), ("L2 logit, C = 0.01", lr(0.01)), ("L2 logit, C = 0.1", lr(0.1)),
            ("L2 logit, C = 10", lr(10.0)), ("Random forest, min leaf 50", rf(50)),
            ("Random forest, min leaf 5", rf(5)), ("Gradient boosting", HistGradientBoostingClassifier(random_state=42))]


def cross_fit(model, Xs: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Cross-fitted scores with the folds of est.fit_propensity (same split, same seed)."""
    cv = StratifiedKFold(n_splits=config.PS_FOLDS, shuffle=True, random_state=config.SEED)
    return cross_val_predict(model, Xs, d, cv=cv, method="predict_proba")[:, 1]


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("Propensity model comparison: regularization strength, random forest, gradient boosting.")
    ap.add_argument("--boot", type=int, default=config.N_BOOT)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    args = ap.parse_args(argv)
    log = setup_logging("m8_ps_models")
    t0 = time.time()

    D = analysis_sample()
    X, d = covariates(D), np.asarray(D["y_treat"], float)
    y = np.asarray(D["oc_primary"], float)
    keep = X.std(axis=0) > 0
    names = [n for n, k in zip(lp.covariate_names(D), keep) if k]
    Xs = StandardScaler().fit_transform(X[:, keep])     # standardized once on the full sample, as fit_propensity
    clusters = est.ClusterIndex(D["mentee_ids"])
    ref = pd.read_csv(OUT / "B_main_results.csv").set_index("Outcome")
    p = d.mean()

    rows, bal = [], []
    for label, model in models(args.jobs):
        t1 = time.time()
        e = cross_fit(model, Xs, d)
        st = est.make_strata(e, config.N_STRATA)
        att = est.att_stratified(y, d, st).att
        if label == MAIN:
            assert abs(ref.ATT.iloc[0] - att) < 1e-12, "C = 1 does not reproduce the main ATT"
        lo, hi = est.percentile_ci(est.bootstrap_att(y, d, st, clusters, np.random.RandomState(config.SEED), args.boot))
        smd = est.smd_table(Xs, d, st, names, weighting="att")
        smd_leg = est.smd_table(Xs, d, st, names, weighting="legacy")
        cs = est.common_support(e, d)
        ctl_top4 = [int(((st == k) & (d == 0)).sum()) for k in range(int(st.max()) - 3, int(st.max()) + 1)]
        rows.append({
            "Model": label, "N": len(d), "Treated": int(d.sum()), "Strata": int(st.max()) + 1,
            "AUC": float(roc_auc_score(d, e)), "Brier": float(brier_score_loss(d, e)),
            "Brier_constant": float(p * (1 - p)), "Log_loss": float(log_loss(d, e)),
            "PS_min": float(e.min()), "PS_max": float(e.max()),
            "n_outside_support": cs["n_outside"], "pct_outside_support": cs["pct_outside"],
            "Controls_top4_strata": " ".join(map(str, ctl_top4)),
            "Max_SMD_raw": float(smd.raw.max()), "Max_SMD_after": float(smd.strat.max()),
            "Mean_SMD_after": float(smd.strat.mean()), "n_SMD_after_gt_0.1": int((smd.strat > 0.1).sum()),
            "Max_SMD_after_legacy": float(smd_leg.strat.max()),
            "ATT": att, "CI_lo": lo, "CI_hi": hi, "Sig": "*" if est.is_significant(lo, hi) else "", "B": args.boot,
        })
        bal.append(smd.assign(Model=label)[["Model", "var", "raw", "strat"]])
        r = rows[-1]
        log.info("%-28s AUC=%.4f Brier=%.4f maxSMD=%.4f meanSMD=%.4f PS=[%.3f, %.3f] ATT=%+.4f [%+.4f, %+.4f] (%.0f s)",
                 label, r["AUC"], r["Brier"], r["Max_SMD_after"], r["Mean_SMD_after"], r["PS_min"], r["PS_max"],
                 att, lo, hi, time.time() - t1)
    if args.boot != config.N_BOOT:
        log.warning("test run: B = %d, results not written", args.boot)
        return
    comp, bfile = OUT / "ps_model_comparison.csv", OUT / "ps_model_balance.csv"
    pd.DataFrame(rows).to_csv(comp, index=False)
    pd.concat(bal).to_csv(bfile, index=False)
    write_manifest("m8_ps_models", inputs=[config.ANALYTIC / "psm_dataset_legacy_layout.npz", OUT / "B_main_results.csv"],
                   outputs=[comp, bfile], params={"n_boot": args.boot, "seed": config.SEED, "folds": config.PS_FOLDS,
                                                  "strata": config.N_STRATA},
                   counts={"N": len(d), "treated": int(d.sum()), "models": len(rows)}, started=t0)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
