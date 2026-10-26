"""
build_m8_check.py: writes checks/m8_check.ipynb (checks of the M8 analysis results on the new data)

The notebook follows the fixed 8-section layout of PLAN.md section 12:
    1 module description   2 files and manifest   3 invariants of the result tables
    4 comparison with the paper version   5 independent reproduction of the main result, tracer and random people
    6 edge-case lists   7 distributions and the other result tables   8 summary (results/m8_report.json)

Section 5 recomputes the propensity score, the strata, the ATT, the E-value and the balance with plain
sklearn / numpy code written in the notebook (it does not call estimators.py or legacy_parity.py).

Run: python checks/build_m8_check.py [--run]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks._nbgen import code, md, run_as_script, write_notebook  # noqa: E402

CELLS = [
md("""
# M8 analysis on the new data: check notebook

## 1. What this module does, and how to use this notebook

**Module**: `pipeline/m8_analysis/run_analysis.py`, which reuses `legacy_parity.py` (code that reproduces the
paper version cell by cell) and `estimators.py`. Issues: I-10, I-11, I-19, I-24, I-26, I-35, I-36.

M8 answers the research question: does a reply to a newcomer's first question increase later editing?
It compares newcomers who got a reply (treated) with similar newcomers who did not (control):

1. **Propensity score**: for each person, the predicted chance of getting a reply given the 162 covariates
   from M7 (L2 logistic regression on standardized covariates, C = 1). The prediction for a person comes from
   a model fitted on the other 4 of 5 folds (cross-fitting).
2. **Strata**: people are cut into 10 groups of equal size by their propensity score. Inside one group,
   treated and control people had a similar chance of a reply.
3. **ATT** (average effect on the treated): in each stratum, treated mean minus control mean, averaged over
   strata with weights equal to the number of treated people in the stratum. Strata with fewer than 2 treated
   or 2 control people are skipped.
4. **Confidence interval**: 500 bootstrap draws that resample newcomers (and, as a second version, mentors).
5. **Checks of the method**: covariate balance after stratification, common support, E-value (how strong an
   unmeasured confounder would have to be), robustness settings, outcome windows, subgroups, interaction
   tests, and a mixed model of reply features within the treated group.

**Which paper numbers depend on it**: Table 4 (main effects), the balance and common-support numbers, the
E-value, the robustness and window results, Table 5 (subgroups), the interaction tests, and Tables 6 and 7
(reply features).

**How to use**: run from top to bottom. Every check prints `[OK ]` or `[FAIL]`. Section 5 rebuilds the main
result from the npz with a few lines of sklearn, so you can see each step. It also follows random people
into the estimate. To draw other random people, change `SEED` in the next cell.
"""),
code(r'''
SEED = 0        # change this number to draw a fresh random sample of people in section 5
N_RANDOM = 5    # how many random people to show in detail

import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.common.tracer import tracer_frame
from checks._show import Checks, compare, pick, oldid, diff, contribs, logid
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
R = config.RESULTS / "m8"
L = config.RESULTS / "m8_legacy"
def t(name, root=R):
    """Read one result table. Dashes in labels written by the pipeline are shown as plain hyphens."""
    x = pd.read_csv(root / f"{name}.csv")
    for c in x.columns[x.dtypes == object]:
        x[c] = x[c].astype(str).str.replace("\u2013", "-").str.replace("\u2014", "-").replace("nan", "")
    return x
NPZ = config.ANALYTIC / "psm_dataset_legacy_layout.npz"
with np.load(NPZ, allow_pickle=True) as z:
    ALL = {k: z[k] for k in z.files}
n_all = len(ALL["y_treat"])
D = {k: (v[ALL["in_sample"]] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n_all else v) for k, v in ALL.items()}
# outcome keys used for the 6 rows of Table B (label in the result table -> npz key)
OUTCOMES = [("1+_mainspace_edit_14d", "oc_primary"), ("n_mainspace_edits_14d", "oc_n_mainspace_edits_14d"),
            ("active_days_30d", "oc_active_days_30d"), ("unique_namespace_14d", "oc_unique_ns"),
            ("reverted_any_14d", "oc_reverted_any"), ("2+_active_days_14d", "oc_cross_day_any_14d")]
ck = Checks()
print(f"main sample: {len(D['y_treat'])} people, {int(D['y_treat'].sum())} treated")
'''),
md("""
## 2. Files and manifest

M8 must have run **after** the npz it read was written, and the npz must not have changed since (same sha256).
Otherwise the result tables are out of date.
"""),
code(r'''
man = json.loads((config.INTERIM / "m8" / "manifest.json").read_text())
man7 = json.loads((config.INTERIM / "m7" / "manifest.json").read_text())
print("M8 finished:", man["finished_at"], "| run time:", man["elapsed_sec"], "s | params:", man["params"])
print("M7 finished:", man7["finished_at"])
print("counts:", json.dumps(man["counts"]))
npz_sha = hashlib.sha256(NPZ.read_bytes()).hexdigest()
same_npz = npz_sha == man["inputs"][0]["sha256"]
print("npz unchanged since M8 read it:", same_npz)
if not same_npz:
    print("NOTE: the npz was rebuilt after M8 ran, so the tables in results/m8 may be out of date. Section 5 recomputes the main\n"
          "      result from the CURRENT npz; if it differs from Table B, rerun M8 (python -m pipeline.m8_analysis.run_analysis)\n"
          "      and then this notebook.")
files = sorted(R.glob("*.csv"))
display(pd.DataFrame([{"file": f.name, "rows": sum(1 for _ in f.open()) - 1,
                       "modified (UTC)": pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")} for f in files]))
'''),
md("""
## 3. Invariants of the result tables
"""),
code(r'''
B = t("B_main_results"); A3 = t("A3_strata_diagnostics"); A5 = t("A5_common_support"); BAL = t("balance_att_weighting")
EV = t("evalues"); ROB = t("robustness"); TXT = t("s13_text_numbers").iloc[0]
d = D["y_treat"].astype(float)
ck.add("M8 used the current npz (same sha256)", same_npz)
ck.add("M8 ran after the latest M7 run", pd.Timestamp(man["finished_at"]) > pd.Timestamp(man7["finished_at"]))
ck.add("bootstrap B = 500 (final run, not a debug run)", man["params"]["n_boot"] == 500)
ck.add("manifest N and treated equal the in_sample rows of the npz", man["counts"]["N"] == len(d) and man["counts"]["treated"] == int(d.sum()))
ck.add("Table B has the 6 outcomes in the expected order", list(B.Outcome) == [o for o, _ in OUTCOMES])
ck.add("every ATT lies inside its 95% CI", bool(((B.CI_lo <= B.ATT) & (B.ATT <= B.CI_hi)).all()))
ck.add("'*' is set exactly when the CI excludes 0", bool(((B.Sig == "*") == ((B.CI_lo > 0) | (B.CI_hi < 0))).all()))
ck.add("strata table: 10 strata, sizes add up to N", len(A3) == 10 and A3.N.sum() == len(d) and (A3.Treated + A3.Control == A3.N).all())
ck.add("strata are ordered by propensity score and do not overlap", bool((A3.PS_hi.shift().fillna(-1) < A3.PS_lo).all()))
ck.add("after stratification no covariate has |SMD| > 0.1 (ATT weighting)", int((BAL.strat.abs() > 0.1).sum()) == 0)
ck.add("fewer than 1% of people are outside common support", float(A5.pct_outside.iloc[0]) < 0.01)
ck.add("robustness 'Main' row has the same ATT as Table B", np.isclose(ROB.ATT.iloc[0], B.ATT.iloc[0]))
ck.add("corrected E-value >= 1", float(EV.e_point.iloc[2]) >= 1)
ck.report()
'''),
md("""
## 4. Comparison with the paper version (same code, paper-version data)

`results/m8_legacy/` holds the same tables computed on the paper's npz. Same code, different data: any
difference comes from the data fixes. Both versions use 500 bootstrap draws.
"""),
code(r'''
cols = ["Outcome", "Naive", "ATT", "CI_lo", "CI_hi", "Sig"]
cmp = B[cols].merge(t("B_main_results", L)[cols], on="Outcome", suffixes=("_new", "_paper"))
display(cmp.round(4))
print("\nMentor-clustered bootstrap (resample mentors instead of newcomers):")
display(t("B_mentor_boot").round(4))
'''),
md("""
## 5. Independent reproduction, random people, tracer people

### 5.1 Propensity score, strata and ATT, rebuilt with plain sklearn

Steps: stack the covariate blocks E, Qtext, Qpersp, Qtype, emb20, temporal (the M block is not used in the main
model); standardize; 5-fold cross-fitted logistic regression (C = 1, lbfgs, StratifiedKFold with shuffle and
random_state 42); cut into 10 quantile strata; ATT as defined in section 1. The naive difference is the plain
treated mean minus control mean, with no adjustment.
"""),
code(r'''
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

X = np.hstack([D["X_E"], D["X_Qtext"], D["X_Qpersp"], D["X_Qtype"], D["X_emb20"], D["X_temporal"]])
NAMES = (list(D["E_cols"]) + list(D["Qtext_cols"]) + list(D["Qpersp_cols"]) + list(D["Qtype_cols"])
         + [f"emb_pc{i + 1}" for i in range(20)] + [str(c) for c in D["temporal_cols"]])
print("covariate matrix:", X.shape)
Xs = StandardScaler().fit_transform(X)
lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
e_hat = cross_val_predict(lr, Xs, d, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
stratum = pd.qcut(e_hat, 10, labels=False, duplicates="drop").astype(int)
auc = roc_auc_score(d, e_hat)

def my_att(y, d, s):
    """Stratified ATT: sum over usable strata of n1k * (mean1k - mean0k) / sum of n1k."""
    num = den = 0.0
    for k in np.unique(s):
        a, b = y[(s == k) & (d == 1)], y[(s == k) & (d == 0)]
        if len(a) >= 2 and len(b) >= 2:
            num += len(a) * (a.mean() - b.mean()); den += len(a)
    return num / den

ok = []
for label, key in OUTCOMES:
    y = D[key].astype(float)
    row = B.set_index("Outcome").loc[label]
    ok.append(compare({"Naive": y[d == 1].mean() - y[d == 0].mean(), "ATT": my_att(y, d, stratum)},
                      {"Naive": row.Naive, "ATT": row.ATT}, title=f"\n{label}  (npz key {key})"))
print(f"\nAUC: recomputed {auc:.6f} | M8 {TXT.auc:.6f}")
ck.add("independent sklearn reproduction: naive difference and ATT of all 6 outcomes equal Table B", all(ok))
ck.add("independent reproduction: AUC equals M8", np.isclose(auc, TXT.auc))
y1 = D["oc_primary"].astype(float)
ATT = my_att(y1, d, stratum)
'''),
md("""
### 5.2 Per-stratum table for the main outcome (1+ mainspace edit in 14 days)

`weight` = share of all treated people who are in that stratum. ATT = sum of weight times (treated mean minus
control mean). The last column compares the stratum sizes with M8's table A3.
"""),
code(r'''
rows = []
for k in range(stratum.max() + 1):
    m = stratum == k
    a, b = y1[m & (d == 1)], y1[m & (d == 0)]
    rows.append({"stratum": k, "ps_from": e_hat[m].min(), "ps_to": e_hat[m].max(), "n_treated": len(a), "n_control": len(b),
                 "mean_treated": a.mean(), "mean_control": b.mean(), "difference": a.mean() - b.mean(),
                 "used": len(a) >= 2 and len(b) >= 2})
ST = pd.DataFrame(rows)
ST["weight"] = np.where(ST.used, ST.n_treated, 0) / ST.loc[ST.used, "n_treated"].sum()
ST["weight_x_difference"] = ST.weight * ST.difference
ST["same size as A3"] = (ST.n_treated.values == A3.Treated.values) & (ST.n_control.values == A3.Control.values)
display(ST.round(4))
print(f"sum of weight x difference = {ST.weight_x_difference.sum():.6f} = ATT {ATT:.6f}")
ck.add("per-stratum treated and control counts equal M8 table A3", bool(ST["same size as A3"].all()))
'''),
md("""
### 5.3 E-value, recomputed

p1 = share of treated people with the outcome. Without the reply, the treated group would have p1 minus ATT.
Risk ratio RR = p1 / (p1 - ATT), and E-value = RR + sqrt(RR x (RR - 1)). For the CI bound, use the lower CI
limit in place of the ATT. An E-value of 1.4 means an unmeasured confounder would need a risk ratio of about 1.4
with both the reply and the outcome to explain the effect away.
"""),
code(r'''
p1 = y1[d == 1].mean()
ev = lambda rr: rr + np.sqrt(rr * (rr - 1))
lo = B.CI_lo.iloc[0]
rr, rr_lo = p1 / (p1 - ATT), p1 / (p1 - lo)
fixed = EV.iloc[2]
okE = compare({"p1": p1, "rr": rr, "e_point": ev(rr), "rr_lo": rr_lo, "e_ci": ev(rr_lo) if lo > 0 else 1.0},
              {"p1": p1, "rr": fixed.rr, "e_point": fixed.e_point, "rr_lo": fixed.rr_lo, "e_ci": fixed.e_ci},
              title="corrected E-value (row 3 of results/m8/evalues.csv)")
ck.add("corrected E-value recomputed from p1 and ATT equals evalues.csv", okE)
EV_EN = EV.assign(method=["paper text (RR = 1 + ATT, wrong)", "23fea07 s13 code (unadjusted p1/p0)", "corrected (RR = p1 / (p1 - ATT))"])
display(EV_EN.round(4))
'''),
md("""
### 5.4 Balance: standardized mean difference (SMD) before and after stratification

SMD = |treated mean - control mean| / pooled SD (pooled SD = sqrt of the mean of the two group variances).
After stratification the mean difference is averaged over strata with ATT weights (number treated). A common
rule of thumb is that |SMD| below 0.1 means the groups are balanced on that covariate.
"""),
code(r'''
t1 = d == 1
sp = np.sqrt((X[t1].var(axis=0) + X[~t1].var(axis=0)) / 2)
raw = np.abs(X[t1].mean(axis=0) - X[~t1].mean(axis=0)) / np.where(sp > 0, sp, np.nan)
num, den = np.zeros(X.shape[1]), 0.0
for k in np.unique(stratum):
    a, b = X[(stratum == k) & t1], X[(stratum == k) & ~t1]
    if len(a) and len(b):
        num += len(a) * (a.mean(axis=0) - b.mean(axis=0)); den += len(a)
strat = np.abs(num / den) / np.where(sp > 0, sp, np.nan)
SMD = pd.DataFrame({"covariate": NAMES, "SMD_before": raw, "SMD_after": strat}).fillna(0)
print("covariates with |SMD| > 0.1: before", int((SMD.SMD_before > 0.1).sum()), "| after", int((SMD.SMD_after > 0.1).sum()), "of", len(SMD))
print("largest 10 before stratification, and what is left after:")
display(SMD.sort_values("SMD_before", ascending=False).head(10).round(4))
print("largest 10 after stratification:")
display(SMD.sort_values("SMD_after", ascending=False).head(10).round(4))
ck.add("our SMD (before and after) equals results/m8/balance_att_weighting.csv",
       np.allclose(SMD.SMD_before, BAL.raw, atol=1e-9) and np.allclose(SMD.SMD_after, BAL.strat, atol=1e-9))
'''),
md("""
### 5.5 Random people: how individuals enter the estimate

The ATT is a weighted sum of outcomes: a treated person gets weight 1 / (number of treated in usable strata);
a control person gets minus (n treated in their stratum) / (all treated x n control in their stratum). So a
control person in a stratum with many treated and few controls counts a lot. The column `weight_x_N1` shows the
weight times the number of treated people: 1 for every treated person, and minus (treated per control in the
stratum) for a control person. The check below confirms that the sum of weight x outcome over all people
equals the ATT.
"""),
code(r'''
n1k = ST.set_index("stratum").n_treated; n0k = ST.set_index("stratum").n_control; used = ST.set_index("stratum").used
N1 = ST.loc[ST.used, "n_treated"].sum()
w = np.where(used[stratum].to_numpy(), np.where(d == 1, 1 / N1, -n1k[stratum].to_numpy() / (N1 * n0k[stratum].to_numpy())), 0.0)
ck.add("sum of person weight x outcome equals the ATT", np.isclose((w * y1).sum(), ATT))
P = pd.DataFrame({"mentee_uid": D["mentee_ids"].astype(int), "first_question_revid": D["cid_order"], "q_ts": D["q_ts"],
                  "treated": d.astype(int), "e_hat": e_hat, "stratum": stratum, "oc_primary": y1, "weight_in_ATT": w,
                  "weight_x_N1": w * N1})
for c in ["n_edits", "account_age_hours", "ns0_mainspace"]:
    P[c] = D["X_E"][:, list(D["E_cols"]).index(c)]
P["q_body_word_count"] = D["X_Qtext"][:, list(D["Qtext_cols"]).index("q_body_word_count")]
P["q_substantive"] = D["X_Qtype"][:, 0]; P["q_persp_toxicity"] = D["X_Qpersp"][:, 0]
# about three quarters of people are treated; draw both groups so the reader sees how controls enter too
n_c = max(1, N_RANDOM // 2)
RND = pd.concat([pick(P[P.treated == 1], N_RANDOM - n_c, SEED), pick(P[P.treated == 0], n_c, SEED)])
display(RND.drop(columns="weight_in_ATT").round(4))
for r in RND.itertuples():
    k = r.stratum
    print(f"  uid {r.mentee_uid}: {'treated' if r.treated else 'control'}, e_hat {r.e_hat:.3f} -> stratum {k} "
          f"({n1k[k]} treated, {n0k[k]} control; treated mean {ST.mean_treated[k]:.3f}, control mean {ST.mean_control[k]:.3f}); "
          f"outcome {int(r.oc_primary)}; weight x N1 = {r.weight_x_N1:+.2f}; question {oldid(r.first_question_revid)}")
'''),
md("""
### 5.6 Tracer people in the main sample
"""),
code(r'''
tr = tracer_frame().query("kind == 'mentee'")
T = P[P.mentee_uid.isin(tr.uid.astype(int))].merge(tr[["label", "uid"]].assign(uid=lambda x: x.uid.astype(int)), left_on="mentee_uid", right_on="uid")
display(T[["label", "mentee_uid", "treated", "e_hat", "stratum", "oc_primary", "n_edits", "q_substantive", "weight_x_N1"]].round(4))
print("tracer mentees not in the main sample:", sorted(set(tr.uid.astype(int)) - set(P.mentee_uid)))
'''),
md("""
## 6. Edge-case lists
"""),
code(r'''
print("6.1 strata skipped in the ATT (fewer than 2 treated or 2 control):", list(ST.stratum[~ST.used]))
cs_lo, cs_hi = max(e_hat[t1].min(), e_hat[~t1].min()), min(e_hat[t1].max(), e_hat[~t1].max())
out = P[(P.e_hat < cs_lo) | (P.e_hat > cs_hi)]
print(f"\n6.2 common support [{cs_lo:.4f}, {cs_hi:.4f}] (M8: [{A5.lo.iloc[0]:.4f}, {A5.hi.iloc[0]:.4f}]); people outside: {len(out)}")
ck.add("common support range and count equal A5", np.isclose(cs_lo, A5.lo.iloc[0]) and np.isclose(cs_hi, A5.hi.iloc[0]) and len(out) == A5.n_outside.iloc[0])
display(out[["mentee_uid", "treated", "e_hat", "stratum", "n_edits", "account_age_hours"]].round(4))
print("\n6.3 lowest and highest propensity scores:")
display(pd.concat([P.nsmallest(3, "e_hat"), P.nlargest(3, "e_hat")])[["mentee_uid", "treated", "e_hat", "n_edits", "account_age_hours", "q_body_word_count"]].round(4))
print("\n6.4 control people with the largest weight (they stand in for many treated people):")
display(P.nsmallest(3, "weight_in_ATT")[["mentee_uid", "e_hat", "stratum", "oc_primary", "weight_x_N1"]].round(3))
print("\n6.5 covariates that are constant in the main sample (no information):", [str(n) for n, s in zip(NAMES, X.std(axis=0)) if s == 0])
'''),
md("""
## 7. Distributions and the other result tables

### 7.1 Propensity scores by group (common support)

If the two histograms overlap over the whole range, every treated person has comparable control people.
"""),
code(r'''
import matplotlib.pyplot as plt
bins = np.linspace(0, 1, 21)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(e_hat[~t1], bins=bins, density=True, histtype="step", linewidth=2, color="#2a6fdb", label=f"control (n={int((~t1).sum())})")
ax.hist(e_hat[t1], bins=bins, density=True, histtype="step", linewidth=2, color="#e07b00", label=f"treated (n={int(t1.sum())})")
ax.set_xlabel("propensity score (predicted chance of a reply)"); ax.set_ylabel("density"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.show()
h = pd.DataFrame({"bin": [f"{a:.2f}-{b:.2f}" for a, b in zip(bins[:-1], bins[1:])],
                  "control": np.histogram(e_hat[~t1], bins)[0], "treated": np.histogram(e_hat[t1], bins)[0]})
display(h[(h.control + h.treated) > 0])
'''),
md("""
### 7.2 Ablation: which covariate blocks matter (table A7)
Each row fits the propensity score with only some blocks. If the ATT barely moves, the result does not hinge on
one block.
"""),
code(r'''
display(t("A7_feature_ablation").round(4))
'''),
md("""
### 7.3 Robustness settings
Each row re-runs the whole estimate (new propensity score, new strata, 500 bootstrap draws) on a different
sample or treatment definition: without the D10 restriction, treatment = reply from the assigned mentor only,
without late-fetched people, only questions at least 30 days before the cutoff, only people with full 14-day
follow-up, only people with no imputed text values, only people also in the paper sample.
Then: number of strata (C1), trimming extreme propensity scores (C3), and other pseudo start times for the
control group (C7).
"""),
code(r'''
display(t("robustness").round(4))
display(t("C1_strata_sensitivity").round(4).head(12))
display(t("C3_ps_trimming").round(4))
display(t("C7_pseudo_time").round(4))
'''),
md("""
### 7.4 Outcome windows
Main outcome measured over 7 to 180 days. `ATT` uses only people whose full window lies before the data cutoff;
`ATT_all_obs` uses everyone (as the paper did).
"""),
code(r'''
w = t("window_followup_complete")[["window", "N", "ATT", "CI_lo", "CI_hi", "Sig"]].merge(
    t("C4_window_sensitivity")[["window", "ATT"]], on="window", suffixes=("", "_all_obs"))
display(w.round(4))
'''),
md("""
### 7.5 Subgroups (Table 5) and interaction tests (a2)
Subgroup ATTs use 5 strata inside each subgroup. A subgroup effect that looks larger is only meaningful if the
interaction test (difference between groups, with a bootstrap CI) excludes 0.
"""),
code(r'''
sg = t("subgroup_att")
display(sg[sg.Outcome.str.contains("1\\+|primary", regex=True)].round(4))
display(t("subgroup_diagnostics").round(4))
a2 = t("a2_interaction_results")
display(a2[a2.DV == "1+_mainspace_edit_14d"].round(4))
'''),
md("""
### 7.6 Reply features (Tables 6 and 7)
Within treated people only: a mixed model of the main outcome on standardized reply features (random intercept
per mentor), controlling for all covariates. `q_BH` is the Benjamini-Hochberg adjusted p value. These are
associations, not causal effects.
"""),
code(r'''
display(t("reply_association").sort_values("p").round(4))
display(t("reply_vif").round(3))
'''),
md("""
## 8. Summary (written to results/m8_report.json)
"""),
code(r'''
ck.report()
b0 = B.iloc[0]
rep = {"N": int(len(d)), "treated": int(d.sum()), "AUC": float(auc), "ATT_primary": float(b0.ATT), "CI": [float(b0.CI_lo), float(b0.CI_hi)],
       "ATT_primary_reproduced": float(ATT), "naive_primary": float(b0.Naive), "e_value_corrected": float(fixed.e_point),
       "e_value_ci": float(fixed.e_ci), "n_smd_after_gt_0.1": int((SMD.SMD_after > 0.1).sum()),
       "checks_passed": sum(ok for _, ok in ck.items), "checks_total": len(ck.items),
       "failed_checks": [n for n, ok in ck.items if not ok], "seed": SEED, "random_people": [int(u) for u in RND.mentee_uid]}
(config.RESULTS / "m8_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
print(f"\nMain result: a reply raises the chance of at least one article edit in the next 14 days by {b0.ATT * 100:.2f} percentage points "
      f"(95% CI {b0.CI_lo * 100:.2f} to {b0.CI_hi * 100:.2f}); the unadjusted difference is {b0.Naive * 100:.2f} points.")
'''),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "m8_check.ipynb"
    write_notebook(CELLS, out)
    print("wrote", out)
    if "--run" in sys.argv:
        run_as_script(out, cwd=out.parent.parent)
        print("\nnotebook ran end to end")


if __name__ == "__main__":
    main()
