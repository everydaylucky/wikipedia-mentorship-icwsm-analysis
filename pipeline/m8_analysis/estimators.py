"""
estimators.py: estimators for the propensity score, stratified ATT, bootstrap, balance, and sensitivity analysis

Module: M8 (PLAN.md M8.1 to M8.8; CODE_PLAN.md section 5, M8)
Issues fixed: I-19, I-36 (E-value formula), I-35 (weighting used in the balance check)
Decision (2026-10-07): the sensitivity analysis uses only the E-value. Rosenbaum bounds are no longer computed.

1. What it does
    Provides the estimators that all analyses share (propensity score, stratified ATT, bootstrap,
    balance, E-value, interaction terms). Each function's docstring states the formula it
    implements, and the formula numbers match the M8 section of PLAN.md. Where the paper version
    (git 23fea07) and the corrected version do things differently, a parameter switches between
    them explicitly (for example smd_table(weighting=...)), and a comment explains the difference.
    This way the same code can both reproduce the paper cell by cell and give the corrected results.

2. Why it is designed this way
    1. Match the paper version cell by cell: every detail of the propensity score model,
       stratification, ATT, and bootstrap (how variables are standardized, how cross-validation
       folds are split, how quantile strata are handled, skipping strata with too few people)
       is the same as in s13_psm_analysis.py. legacy_parity.py uses this to reproduce the paper.
    2. The bootstrap uses "number of times drawn as a weight" instead of copying rows.
       In each resample the paper version stacked the drawn rows into a new array. With
       500 draws times dozens of specifications, that is slow. Here we first draw how many
       times each cluster (newcomer or mentor) is picked, then use that count as a row weight
       to compute a weighted ATT. Mathematically this is exactly the same as copying rows
       (copying a row k times = weight k). It also uses random numbers in exactly the same way
       as the paper version (RandomState.choice(array, n) and randint(0, n, n) consume the same
       stream of random numbers; verified on 2026-10-07). So with the same seed and the same
       call order, the confidence intervals also match the paper version exactly.
    3. The estimators take only numpy arrays. They do not read files or depend on global state,
       which makes them easy to test.

3. Inputs / 4. Outputs
    numpy arrays in; dataclasses or arrays out.

5. Key parameters
    config.PS_FOLDS, config.SEED, config.PS_C (propensity score), config.N_BOOT (bootstrap draws).
    See each function's docstring for the rest.

6. Possible errors and how they are handled
    - A stratum has fewer than 2 treated or fewer than 2 control people: that stratum is skipped
      (same as the paper version) and listed in the result.
    - The bootstrap has 50 or fewer valid draws: the confidence interval is returned as NaN
      (same as the paper version).

7. How to run
    Not run on its own. It is imported by run_analysis.py and legacy_parity.py.

8. How to check
    tests/test_m8.py: tests each function on synthetic data with a known true effect.
    legacy_parity.py: compares with the paper version's output tables, cell by cell, on the paper data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import config

# ════════════════════════════════════════════════════════════════════════════
# M8.2 Propensity score
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class PSResult:
    """Result of the propensity score estimation.

    Attributes:
        e_hat: cross-fitted propensity score ê_i for each person.
        auc: AUC of ê for the treatment variable (a check of how well it separates the groups).
    """
    e_hat: np.ndarray
    auc: float


def fit_propensity(X: np.ndarray, d: np.ndarray, *, folds: int = config.PS_FOLDS,
                   seed: int = config.SEED, C: float = config.PS_C, max_iter: int = 2000,
                   lr_random_state: int | None = 42, clip: tuple[float, float] | None = None
                   ) -> PSResult:
    """Estimate the propensity score e(W) = P(D=1 | W) (PLAN M8.2).

    Formula:
        e(W_i) = 1 / (1 + exp(-β0 - W_i'β)), an L2-regularized logistic regression
        (C is the inverse of the regularization strength).
        We use K-fold cross-fitting: person i's ê_i comes from a model trained on the other
        folds, not the fold that contains i.

    Args:
        X: covariate matrix (not standardized). Inside the function it is standardized with the
           full-sample mean and standard deviation, the same as the paper version s13
           (X_main_s = StandardScaler().fit_transform(X_main)).
        d: treatment variable (0/1).
        folds: number of cross-fitting folds.
        seed: shuffle seed for StratifiedKFold.
        C: inverse of the regularization strength.
        max_iter: maximum number of optimizer iterations. The paper version uses 2000 in s13 and
                  s14, and 1000 in a2.
        lr_random_state: random_state of LogisticRegression. The paper version uses 42 in s13 and
                         s14, and does not set it in a2 (None). The lbfgs solver is not sensitive
                         to it; we keep it only to reproduce the paper cell by cell.
        clip: if (lo, hi) is given, ê is clipped to that range. The paper version a2 uses
              (0.01, 0.99); the main analysis does not clip.

    Returns:
        PSResult.
    """
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    lr = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", random_state=lr_random_state)
    e = cross_val_predict(lr, Xs, d, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(d, e))
    if clip is not None:
        e = np.clip(e, *clip)
    return PSResult(e, auc)


def make_strata(e_hat: np.ndarray, K: int) -> np.ndarray:
    """Split people into K strata by quantiles of the propensity score (PLAN M8.3).

    Notes:
        Same as the paper version, it uses pd.qcut(ê, K, labels=False, duplicates="drop").
        When quantile edges repeat, strata are merged, so the real number of strata can be
        smaller than K. Returns integer stratum numbers 0..K'-1.
    """
    return np.asarray(pd.qcut(e_hat, K, labels=False, duplicates="drop"), dtype=np.int64)


# ════════════════════════════════════════════════════════════════════════════
# M8.3 Stratified ATT
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ATTResult:
    """Result of the stratified ATT.

    Attributes:
        att: point estimate.
        skipped: numbers of the skipped strata (fewer than 2 treated or fewer than 2 control people).
        table: details for each stratum: stratum number, number treated, number of controls,
               the two group means, their difference, and the weight.
    """
    att: float
    skipped: list[int] = field(default_factory=list)
    table: pd.DataFrame | None = None


def _strata_sums(y: np.ndarray, d: np.ndarray, s: np.ndarray, w: np.ndarray, K: int):
    """Weighted head counts and weighted outcome sums for the treated and control groups in each
    stratum. Internal helper, vectorized."""
    t = d == 1
    n1 = np.bincount(s[t], weights=w[t], minlength=K)
    n0 = np.bincount(s[~t], weights=w[~t], minlength=K)
    y1 = np.bincount(s[t], weights=(w * y)[t], minlength=K)
    y0 = np.bincount(s[~t], weights=(w * y)[~t], minlength=K)
    return n1, n0, y1, y0


def att_stratified(y: np.ndarray, d: np.ndarray, strata: np.ndarray,
                   w: np.ndarray | None = None, *, with_table: bool = False) -> ATTResult:
    """Stratified ATT (PLAN M8.3).

    Formula:
        τ̂_ATT = Σ_k (n_1k / n_1) (Ȳ_1k − Ȳ_0k),
        where n_1k, Ȳ_1k, Ȳ_0k are the number treated, the treated mean, and the control mean in
        stratum k. Strata with n_1k < 2 or n_0k < 2 are left out, and the weights are rescaled
        over the remaining strata.

    Args:
        y: outcome.
        d: treatment variable (0/1).
        strata: stratum numbers (output of make_strata).
        w: row weights, all 1 by default. The bootstrap uses it for how many times each row was
           drawn. A weight of k is the same as the row appearing k times (exactly what the paper
           version does by copying rows).
        with_table: whether to return the per-stratum details.

    Returns:
        ATTResult. If every stratum is skipped, att is NaN.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d)
    s = np.asarray(strata, dtype=np.int64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    K = int(s.max()) + 1
    n1, n0, y1, y0 = _strata_sums(y, d, s, w, K)
    ok = (n1 >= 2) & (n0 >= 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = y1 / n1 - y0 / n0
    att = float(np.sum(n1[ok] * diff[ok]) / np.sum(n1[ok])) if ok.any() else float("nan")
    present = (n1 + n0) > 0
    skipped = [int(k) for k in np.where(present & ~ok)[0]]
    table = None
    if with_table:
        table = pd.DataFrame({"stratum": np.arange(K), "n_treated": n1, "n_control": n0,
                              "mean_treated": y1 / np.where(n1 > 0, n1, np.nan),
                              "mean_control": y0 / np.where(n0 > 0, n0, np.nan),
                              "diff": diff, "used": ok,
                              "weight": np.where(ok, n1, 0) / max(np.sum(n1[ok]), 1)})
    return ATTResult(att, skipped, table)


def naive_diff(y: np.ndarray, d: np.ndarray) -> float:
    """Unadjusted difference between groups, Ȳ_1 − Ȳ_0."""
    y = np.asarray(y, float)
    return float(y[d == 1].mean() - y[d == 0].mean())


def cohen_d(att: float, y: np.ndarray, d: np.ndarray) -> float:
    """Standardized effect size d = τ̂ / SD(Y | D=0). SD is the population standard deviation
    (ddof=0), same as the paper version."""
    sd = float(np.asarray(y, float)[d == 0].std())
    return att / sd if sd > 0 else 0.0


# ════════════════════════════════════════════════════════════════════════════
# M8.4 Confidence intervals: cluster bootstrap
# ════════════════════════════════════════════════════════════════════════════


class ClusterIndex:
    """Encode cluster ids (newcomer id or mentor id) as 0..G-1 for the bootstrap.

    Notes:
        Clusters are numbered in np.unique's sort order. This is exactly the order of
        unique_m = np.unique(mids) in the paper version's cluster_boot_ci, which is needed to
        reproduce the paper cell by cell.
    """

    def __init__(self, clusters: np.ndarray):
        self.unique, self.code = np.unique(np.asarray(clusters), return_inverse=True)
        self.n = len(self.unique)

    def draw_weights(self, rng: np.random.RandomState) -> np.ndarray:
        """Draw one bootstrap sample: pick G clusters with replacement and return a weight for each
        row (how many times its cluster was picked).

        It uses random numbers exactly like the paper version's
        rng.choice(unique_m, n_cl, replace=True).
        """
        idx = rng.randint(0, self.n, self.n)
        counts = np.bincount(idx, minlength=self.n).astype(float)
        return counts[self.code]


def bootstrap_att(y: np.ndarray, d: np.ndarray, strata: np.ndarray, clusters: ClusterIndex,
                  rng: np.random.RandomState, B: int | None = None) -> np.ndarray:
    """Sampling distribution of the ATT from the cluster bootstrap (PLAN M8.4).

    Args:
        clusters: ClusterIndex (by newcomer: one row per cluster; by mentor: many rows per cluster).
        rng: random number generator. To reproduce the paper, pass a RandomState that is the same
             as the paper version's and is used in the same order.
        B: number of resamples. Defaults to config.N_BOOT (read at call time, so the --boot debug
           setting takes effect).

    Returns:
        An array of length at most B (NaN values removed).
    """
    B = config.N_BOOT if B is None else B
    out = np.empty(B)
    for b in range(B):
        out[b] = att_stratified(y, d, strata, clusters.draw_weights(rng)).att
    return out[~np.isnan(out)]


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile confidence interval. Returns (NaN, NaN) when there are 50 or fewer valid draws,
    same as the paper version."""
    if len(draws) <= 50:
        return float("nan"), float("nan")
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def is_significant(lo: float, hi: float) -> bool:
    """True if the confidence interval does not contain 0."""
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


# ════════════════════════════════════════════════════════════════════════════
# Doubly robust estimate (the DR column of the paper version's Table B; not reported in the paper text)
# ════════════════════════════════════════════════════════════════════════════


def dr_stratified_legacy(y: np.ndarray, d: np.ndarray, strata: np.ndarray, X_std: np.ndarray,
                         n_pca: int = 10, seed: int = 42) -> float:
    """The paper version's "stratification + regression within strata" estimate (s13 strat_dr).
    Used only to reproduce the DR column of Table B.

    How it works:
        Take the first 10 principal components of the (already standardized) covariates. In each
        stratum, run least squares of y on [D, principal components, 1] and take the coefficient
        on D. Average these over strata, weighted by the total number of people in each stratum.

    Warning:
        Weighting by the total stratum size gives an ATE-type estimate, not an ATT-type one. The
        paper text does not report this number, and the corrected version no longer uses it.
    """
    Z = PCA(n_components=n_pca, random_state=seed).fit_transform(X_std)
    total, acc = 0.0, 0.0
    for k in np.unique(strata):
        m = strata == k
        dk = d[m].astype(float)
        if dk.sum() < 2 or (1 - dk).sum() < 2:
            continue
        A = np.column_stack([dk, Z[m], np.ones(m.sum())])
        beta = np.linalg.lstsq(A, y[m], rcond=None)[0]
        acc += m.sum() * beta[0]
        total += m.sum()
    return acc / total if total else float("nan")


# ════════════════════════════════════════════════════════════════════════════
# M8.5 Balance and overlap
# ════════════════════════════════════════════════════════════════════════════


def smd_table(X: np.ndarray, d: np.ndarray, strata: np.ndarray, names: list[str],
              weighting: str = "att") -> pd.DataFrame:
    """Standardized mean difference for each covariate (PLAN M8.5).

    Formula:
        SMD_raw   = |x̄_1 − x̄_0| / s_p,
        SMD_strat = |Σ_k w_k (x̄_1k − x̄_0k)| / s_p,
        s_p = sqrt((s_1² + s_0²) / 2), using the population variance (ddof=0), same as the paper version.

    Args:
        weighting:
            "att"     w_k = n_1k / n_1, which matches how the ATT is weighted (corrected version, I-35).
            "legacy"  w_k = n_k / n (total stratum size), what the paper version s13 A4 does;
                      used only for reproduction.
        Only strata with at least 1 treated and at least 1 control person are used (same as the
        paper version).

    Returns:
        DataFrame with columns var, raw, strat.
    """
    rows = []
    t = d == 1
    ks = np.unique(strata)
    for j, name in enumerate(names):
        v = X[:, j].astype(float)
        sp = np.sqrt((v[t].var() + v[~t].var()) / 2)
        raw = abs(v[t].mean() - v[~t].mean()) / sp if sp > 0 else 0.0
        num, den = 0.0, 0.0
        for k in ks:
            m = strata == k
            a, b = v[m & t], v[m & ~t]
            if len(a) == 0 or len(b) == 0:
                continue
            wk = len(a) if weighting == "att" else m.sum()
            num += wk * (a.mean() - b.mean())
            den += wk
        strat = abs(num / den) / sp if (sp > 0 and den > 0) else 0.0
        rows.append({"var": name, "raw": raw, "strat": strat})
    return pd.DataFrame(rows)


def common_support(e_hat: np.ndarray, d: np.ndarray) -> dict:
    """Common support range [max(min ê_1, min ê_0), min(max ê_1, max ê_0)] and the number of
    people outside it."""
    lo = max(e_hat[d == 1].min(), e_hat[d == 0].min())
    hi = min(e_hat[d == 1].max(), e_hat[d == 0].max())
    outside = int(((e_hat < lo) | (e_hat > hi)).sum())
    return {"lo": float(lo), "hi": float(hi), "n_outside": outside, "pct_outside": outside / len(e_hat)}


# ════════════════════════════════════════════════════════════════════════════
# M8.7 Sensitivity to unobserved confounding
# ════════════════════════════════════════════════════════════════════════════




def e_value_from_rr(rr: float) -> float:
    """E-value of VanderWeele and Ding (2017): E = RR + sqrt(RR (RR − 1)). If RR < 1, take its
    inverse first."""
    if not np.isfinite(rr):
        return float("nan")
    if rr < 1:
        rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def e_value_att(p1: float, att: float, att_lo: float | None = None) -> dict:
    """E-value on the ATT scale (PLAN M8.7, corrected version, I-19, I-36).

    Formula:
        Observed share in the treated group is p_1, the counterfactual share is p_1 − τ̂, and the
        risk ratio is RR = p_1 / (p_1 − τ̂).
        For the lower confidence bound, use τ_lo in place of τ̂. If τ_lo ≤ 0, the E-value is set to 1.

    Returns:
        {"rr": ..., "e_point": ..., "rr_lo": ..., "e_ci": ...}
    """
    rr = p1 / (p1 - att)
    out = {"rr": rr, "e_point": e_value_from_rr(rr)}
    if att_lo is not None:
        if att_lo <= 0:
            out.update({"rr_lo": 1.0, "e_ci": 1.0})
        else:
            rr_lo = p1 / (p1 - att_lo)
            out.update({"rr_lo": rr_lo, "e_ci": e_value_from_rr(rr_lo)})
    return out


def e_value_legacy_one_plus_att(att: float, att_lo: float) -> dict:
    """Where the paper's E-values came from: treating 1 + ATT as a risk ratio (a wrong method,
    used only to explain the paper's numbers).

    The paper's 1.146 (point estimate) and 1.080 (lower confidence bound) were computed this way
    (s13 in git 430f20b). The ATT is a difference in probabilities, so adding 1 does not give a
    risk ratio. This method understates the E-value.
    """
    return {"e_point": e_value_from_rr(1 + att), "e_ci": e_value_from_rr(1 + att_lo) if att_lo > 0 else 1.0}


def e_value_s13_current(p_treated: float, p_control: float, p0: float, att_lo: float) -> dict:
    """E-value of s13 in version 23fea07: the point estimate uses the unadjusted p1/p0, and the
    lower bound uses (p0+τ_lo)/p0 (I-36). Used only for reproduction."""
    rr = p_treated / p_control
    rr_ci = (p0 + att_lo) / p0
    return {"e_point": e_value_from_rr(rr), "e_ci": e_value_from_rr(rr_ci) if att_lo > 0 else 1.0}


# ════════════════════════════════════════════════════════════════════════════
# M8.8 Heterogeneity: interaction regression with stratum fixed effects
# ════════════════════════════════════════════════════════════════════════════


def interaction_ols(y: np.ndarray, d: np.ndarray, m: np.ndarray, strata: np.ndarray,
                    w: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) least squares coefficients of Y = α_s + β D + γ M + δ D·M + ε (PLAN M8.8, item 2).

    Args:
        m: moderator (0/1, or a standardized continuous variable).
        strata: stratum numbers. Stratum fixed effects are dummy variables with the first stratum
                left out (same as the paper version a2).
        w: row weights (in the bootstrap, the number of times each row was drawn).

    Returns:
        Coefficient vector [intercept, β_D, γ_M, δ_DxM, stratum dummies...]. δ is out[3].

    Notes:
        The paper version solves np.linalg.lstsq on the copied rows. Here we solve the weighted
        normal equations instead. Mathematically they are exactly the same (weight k = the row
        appears k times).
    """
    n = len(y)
    ks = np.unique(strata)
    S = (strata[:, None] == ks[None, 1:]).astype(float)
    A = np.column_stack([np.ones(n), d, m, d * m, S])
    if w is None:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    sw = np.sqrt(w)
    return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]


def bootstrap_interaction(y, d, m, strata, clusters: ClusterIndex, rng: np.random.RandomState,
                          B: int | None = None) -> np.ndarray:
    """Cluster bootstrap sampling distribution of the interaction term δ (uses random numbers the
    same way as the paper version a2). B defaults to config.N_BOOT."""
    B = config.N_BOOT if B is None else B
    out = []
    for _ in range(B):
        w = clusters.draw_weights(rng)
        keep = w > 0
        try:
            b = interaction_ols(y[keep], d[keep], m[keep], strata[keep], w[keep])[3]
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(b):
            out.append(b)
    return np.asarray(out)


def advance_rng(rng: np.random.RandomState, clusters: ClusterIndex, B: int) -> None:
    """Move rng forward by the random numbers that B calls of draw_weights would use (without
    computing δ). In parallel runs this sets the correct starting state for each task."""
    for _ in range(B):
        rng.randint(0, clusters.n, clusters.n)


def interaction_task(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate and bootstrap draws for one (moderator, outcome) pair. This is a
    multiprocessing worker, so it must be a top-level function in the module.

    args = (y, d, m, strata, clusters, rng_state, B). rng_state is the RandomState state at the
    moment this task would start in a serial run, so the parallel results are bit-for-bit the same
    as the serial ones.
    """
    y, d, m, strata, clusters, state, B = args
    rng = np.random.RandomState()
    rng.set_state(state)
    return interaction_ols(y, d, m, strata), bootstrap_interaction(y, d, m, strata, clusters, rng, B)
