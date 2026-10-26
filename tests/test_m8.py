"""
test_m8.py: correctness of the M8 estimation functions (CODE_PLAN.md section 7.1).

Each estimation function is checked on synthetic data:
- when the true effect is known, the estimate shows no clear bias;
- "use the draw count as a weight" gives exactly the same numbers as the paper version's "copy the rows";
- the bootstrap matches the paper version draw by draw;
- the E-value matches the example in VanderWeele & Ding (2017).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config

from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis import regression as reg


@pytest.fixture(scope="module")
def synth():
    """Synthetic data with confounding: X affects both treatment and outcome. The true treatment effect is τ = 0.05."""
    rng = np.random.RandomState(0)
    n = 6000
    X = rng.normal(size=(n, 4))
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * X[:, 0] - 0.5 * X[:, 1])))
    d = (rng.uniform(size=n) < p).astype(float)
    y = 0.2 + 0.05 * d + 0.1 * X[:, 0] + 0.05 * X[:, 1] + rng.normal(scale=0.3, size=n)
    return X, d, y


def legacy_strat_att(y_d, y_out, st):
    """An exact copy of the paper version's s13 strat_att, used for comparison."""
    us = np.unique(st); w_sum, att_sum = 0, 0
    for s in us:
        sm = st == s
        yt = y_out[sm & (y_d == 1)]; yc = y_out[sm & (y_d == 0)]
        if len(yt) < 2 or len(yc) < 2:
            continue
        w = len(yt); att_sum += w * (yt.mean() - yc.mean()); w_sum += w
    return att_sum / w_sum if w_sum else np.nan


def test_att_recovers_true_effect(synth):
    X, d, y = synth
    ps = est.fit_propensity(X, d)
    st = est.make_strata(ps.e_hat, 10)
    att = est.att_stratified(y, d, st).att
    naive = est.naive_diff(y, d)
    assert abs(att - 0.05) < 0.02             # after adjustment, close to the true effect
    assert abs(naive - 0.05) > abs(att - 0.05)  # while the unadjusted difference is clearly biased


def test_att_matches_legacy_implementation(synth):
    X, d, y = synth
    st = est.make_strata(est.fit_propensity(X, d).e_hat, 10)
    assert est.att_stratified(y, d, st).att == pytest.approx(legacy_strat_att(d, y, st), abs=1e-12)


def test_weights_equal_duplicated_rows(synth):
    X, d, y = synth
    st = est.make_strata(est.fit_propensity(X, d).e_hat, 10)
    w = np.random.RandomState(1).randint(0, 4, len(y)).astype(float)
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    a = est.att_stratified(y, d, st, w).att
    b = legacy_strat_att(d[rep], y[rep], st[rep])
    assert a == pytest.approx(b, abs=1e-12)


def test_bootstrap_matches_legacy_draw_by_draw(synth):
    """The new bootstrap gives the same result as the paper version's cluster_boot_ci on every draw (same seed and same order)."""
    X, d, y = synth
    st = est.make_strata(est.fit_propensity(X, d).e_hat, 10)
    clusters = np.array([f"m{i // 3:05d}" for i in range(len(y))])   # 3 rows per cluster, to mimic clustering by mentor
    new = est.bootstrap_att(y, d, st, est.ClusterIndex(clusters), np.random.RandomState(42), B=30)
    rng = np.random.RandomState(42)
    unique_m = np.unique(clusters)
    idx = {m: np.where(clusters == m)[0] for m in unique_m}
    old = []
    for _ in range(30):
        sampled = rng.choice(unique_m, len(unique_m), replace=True)
        bi = np.concatenate([idx[m] for m in sampled])
        old.append(legacy_strat_att(d[bi], y[bi], st[bi]))
    np.testing.assert_allclose(new, np.array(old), atol=1e-12)


def test_percentile_ci_requires_more_than_50_draws():
    assert np.isnan(est.percentile_ci(np.zeros(50))[0])
    lo, hi = est.percentile_ci(np.arange(101, dtype=float))
    assert (lo, hi) == (2.5, 97.5)




def test_e_values():
    # example from VanderWeele & Ding (2017): RR = 3.9 gives E = 7.26
    assert est.e_value_from_rr(3.9) == pytest.approx(7.26, abs=0.01)
    # main result in the paper version: p1 = 0.22308, ATT = 0.016505, lower bound 0.005529
    fixed = est.e_value_att(0.2230763376208235, 0.016505247812403104, 0.0055290197243339995)
    assert fixed["e_point"] == pytest.approx(1.374, abs=1e-3)
    assert fixed["e_ci"] == pytest.approx(1.187, abs=1e-3)
    # the 1.146 and 1.080 in the paper come from treating 1 + ATT as a risk ratio
    old = est.e_value_legacy_one_plus_att(0.016505247812403104, 0.0055290197243339995)
    assert old["e_point"] == pytest.approx(1.146, abs=1e-3)
    assert old["e_ci"] == pytest.approx(1.080, abs=1e-3)
    assert est.e_value_att(0.2, 0.01, -0.001)["e_ci"] == 1.0


def test_interaction_weights_equal_duplicated_rows(synth):
    X, d, y = synth
    st = est.make_strata(est.fit_propensity(X, d).e_hat, 10).astype(float)
    m = (X[:, 2] > 0).astype(float)
    w = np.random.RandomState(2).randint(0, 3, len(y)).astype(float)
    keep = w > 0
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    a = est.interaction_ols(y[keep], d[keep], m[keep], st[keep], w[keep])[:4]
    b = est.interaction_ols(y[rep], d[rep], m[rep], st[rep])[:4]
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_smd_weighting_options(synth):
    X, d, _ = synth
    st = est.make_strata(est.fit_propensity(X, d).e_hat, 10)
    att = est.smd_table(X, d, st, list("abcd"), weighting="att")
    leg = est.smd_table(X, d, st, list("abcd"), weighting="legacy")
    assert (att.raw.values == leg.raw.values).all()
    # only the real confounders (X0, X1) become clearly more balanced after stratification; X2 and X3 were balanced from the start, so we do not check them
    assert (att.strat.iloc[:2] < att.raw.iloc[:2]).all() and (att.raw.iloc[:2] > 0.1).all()


def test_bh_qvalues_and_vif():
    q = reg.bh_qvalues(np.array([0.01, 0.04, 0.03, 0.5]))
    # by hand: after sorting, p·n/rank = 0.04, 0.06, 0.0533, 0.5; taking the running minimum from the end gives 0.04, 0.0533, 0.0533, 0.5
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5], atol=1e-12)
    rng = np.random.RandomState(0)
    a = rng.normal(size=500)
    Xs = np.column_stack([a, a + rng.normal(scale=0.01, size=500), rng.normal(size=500)])
    _, kept, table = reg.vif_screen(Xs, ["a", "a_copy", "c"])
    assert "c" in kept and table.dropped.sum() == 2


def test_ns_group():
    from pipeline.m8_analysis.descriptive import ns_group
    assert list(ns_group(np.array([0, 118, 2, 3, 1, 4, 10]))) == ["articles", "drafts", "user_and_talk", "user_and_talk",
                                                                  "user_and_talk", "other", "other"]


@pytest.mark.skipif(not (config.ANALYTIC / "psm_dataset_legacy_layout.npz").exists(), reason="M7 has not been run")
def test_question_anchor_matches_zero_lag_for_controls():
    """For control newcomers, anchoring at the question equals the zero pseudo-lag version computed by M5."""
    import pandas as pd
    from pipeline.m8_analysis.followup import question_anchor_outcome
    from pipeline.m8_analysis.run_analysis import load, subset
    D = load()
    D = subset(D, D["in_sample"] & (D["y_treat"] == 0))
    q = pd.to_datetime(pd.Series(D["q_ts"]), utc=True)
    D = subset(D, (q + pd.Timedelta(days=14) <= pd.Timestamp(config.DATA_CUTOFF)).to_numpy())
    assert (question_anchor_outcome(D) == D["oc_primary__zero"].astype(float)).all()
