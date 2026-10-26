"""
regression.py: regression of outcomes on reply features (mixed model), plus handling of collinearity

Module: M8 (PLAN.md M8.9; CODE_PLAN.md section 5, M8)
Issues fixed: I-26 (the paper says OLS, but the code fits a mixed model; the VIF step dropped 2 features)

1. What it does
    1. vif_screen(): computes the variance inflation factor (VIF) of each reply feature and drops
       features with VIF above the threshold.
    2. remove_collinear(): removes constant columns and linearly dependent columns from the controls.
    3. mixedlm_with_controls(): fits
           Y_i = α + Σ_j β_j z_ij + W_i'θ + u_m(i) + ε_i,  u_m ~ N(0, σ_u²)
       Here z_ij are the standardized reply features, W_i are the 164 pre-question covariates,
       and u_m is a random intercept for each mentor. It is estimated with REML. The function
       returns only the coefficients, standard errors, z and p for the reply features, plus the
       mentor-level ICC. If the mixed model fails, it falls back to ordinary least squares.
    4. bh_qvalues(): Benjamini-Hochberg correction (new in the corrected version; the paper version
       did not have it).

2. Why it is designed this way
    The code follows the paper version s15_2_reply_association.py line by line (same order of
    preprocessing, same statsmodels call arguments), so that the reply-feature table of the paper
    can be reproduced cell by cell.
    The paper text describes this model as "OLS with QR decomposition". The real code first fits a
    mixed model and falls back to OLS only when that fails (I-26). We keep both paths as they are,
    and the result records which one was used.

3. Inputs
    numpy arrays (outcome, reply features, controls, mentor ids).

4. Outputs
    MixedResult objects, arrays, and a VIF table (DataFrame).

5. Key parameters
    config.VIF_THRESHOLD (VIF cutoff). See each function's docstring for the rest.

6. Possible errors and how they are handled
    The mixed model does not converge or raises an error: fall back to OLS. The result then has
    method="ols_fallback" and ICC is NaN.

7. How to run
    Not run on its own. It is imported by run_analysis.py and legacy_parity.py.

8. How to check
    tests/test_m8.py (test_bh_qvalues_and_vif) checks the BH q values and the VIF screen;
    legacy_parity.py compares the reply-feature table with the paper version's output.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config


def vif_screen(Xs: np.ndarray, names: list[str], threshold: float = config.VIF_THRESHOLD
               ) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Screen features by variance inflation factor (same as the paper version: each VIF is
    computed on the design matrix with a constant column added).

    Args:
        Xs: standardized reply features.
        names: feature names.
        threshold: VIF cutoff. Features above it are dropped.

    Returns:
        (matrix of kept features, names of kept features, table with the VIF of every feature).
    """
    Xv = np.column_stack([np.ones(Xs.shape[0]), Xs])
    vifs = [float(variance_inflation_factor(Xv, j + 1)) for j in range(Xs.shape[1])]
    keep = np.array([v <= threshold for v in vifs])
    table = pd.DataFrame({"feature": names, "vif": vifs, "dropped": ~keep})
    return Xs[:, keep], [n for n, k in zip(names, keep) if k], table


def remove_collinear(X: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Drop columns with near-zero variance, then use a QR decomposition to drop linearly
    dependent columns (same as the paper version's function of the same name).

    Returns:
        (cleaned matrix, indexes of the kept columns in the original matrix).
    """
    keep = [j for j in range(X.shape[1]) if X[:, j].std() >= 1e-10]
    X = X[:, keep]
    _, R = np.linalg.qr(X, mode="reduced")
    diag = np.abs(np.diag(R))
    mask = diag > diag.max() * tol
    return X[:, mask], np.array(keep)[mask]


@dataclass
class MixedResult:
    """Result of the reply-feature regression (reply-feature part only)."""
    beta: np.ndarray
    se: np.ndarray
    z: np.ndarray
    p: np.ndarray
    icc: float
    method: str


def _ols(y: np.ndarray, Xr: np.ndarray, Xc: np.ndarray) -> MixedResult:
    """Ordinary least squares (solved with a QR decomposition). Used as the fallback when the
    mixed model fails (the paper version's _ols_fallback)."""
    X = np.hstack([np.ones((len(y), 1)), Xr, Xc])
    n, p = X.shape
    k = Xr.shape[1]
    Q, R = np.linalg.qr(X, mode="reduced")
    beta = np.linalg.solve(R, Q.T @ y)
    resid = y - X @ beta
    s2 = (resid ** 2).sum() / max(n - p, 1)
    Rinv = np.linalg.inv(R)
    se = np.sqrt(np.maximum(s2 * (Rinv @ Rinv.T).diagonal(), 0))
    b, s = beta[1:k + 1], se[1:k + 1]
    t = b / np.where(s > 0, s, np.nan)
    pv = 2 * (1 - stats.t.cdf(np.abs(t), df=max(n - p, 1)))
    return MixedResult(b, s, t, pv, float("nan"), "ols_fallback")


def mixedlm_with_controls(y: np.ndarray, Xr: np.ndarray, Xc: np.ndarray, groups: np.ndarray
                          ) -> MixedResult:
    """Linear mixed model with a random intercept for each mentor (PLAN M8.9; the paper
    version's mixed_effects_with_controls).

    Args:
        y: outcome (treated group).
        Xr: standardized reply features (already screened by VIF).
        Xc: standardized controls; the function runs remove_collinear on them first.
        groups: mentor ids (the groups for the random intercept).

    Returns:
        MixedResult. ICC = σ_u² / (σ_u² + σ_ε²).

    Notes:
        The statsmodels call uses the same arguments as the paper version:
        MixedLM(...).fit(reml=True, method="lbfgs", maxiter=200).
    """
    Xc_clean, _ = remove_collinear(Xc)
    k = Xr.shape[1]
    rf_cols = [f"rf_{i}" for i in range(k)]
    ct_cols = [f"ctrl_{i}" for i in range(Xc_clean.shape[1])]
    df = pd.concat([pd.DataFrame(Xr, columns=rf_cols), pd.DataFrame(Xc_clean, columns=ct_cols)], axis=1)
    exog = sm.add_constant(df[rf_cols + ct_cols])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MixedLM(pd.Series(y), exog, groups=pd.Series(groups)).fit(
                reml=True, method="lbfgs", maxiter=200)
        b = np.array([res.fe_params[c] for c in rf_cols])
        se = np.array([res.bse[c] for c in rf_cols])
        z = np.array([res.tvalues[c] for c in rf_cols])
        p = np.array([res.pvalues[c] for c in rf_cols])
        re_var = res.cov_re.iloc[0, 0] if hasattr(res.cov_re, "iloc") else float(res.cov_re)
        icc = re_var / (re_var + res.scale) if (re_var + res.scale) > 0 else 0.0
        return MixedResult(b, se, z, p, float(icc), "mixedlm")
    except Exception:
        return _ols(y, Xr, Xc_clean)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """q values after the Benjamini-Hochberg correction (new in the corrected version, PLAN M8.9)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def sig_stars(p: float) -> str:
    """Significance stars, same as the paper version: *** p<.001, ** p<.01, * p<.05."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
