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
"""),
md("""
## Plain-language guide

**What this module is for.** The study asks one question: when a newcomer on English Wikipedia asks their
assigned mentor a question and gets a reply, are they more likely to keep editing? M8 is the step that answers
it. All earlier modules (M1 to M7) collect and clean the data; M8 turns that data into the effect sizes, the
confidence intervals and the extra checks reported in the paper.

**What goes in.** One file built by M7: `data/analytic/psm_dataset_legacy_layout.npz`. It has one row per
newcomer. The main analysis sample is the 33,518 newcomers in `in_sample` whose 14-day window ends before the
data cutoff (25,760 of them got a reply). For each newcomer it holds: whether they got a
reply (the "treatment"), 164 background facts measured before the reply (edit history, account age, length and
tone of the question, question type, text embedding summaries, month), and the outcomes (for example, did they
make at least one edit to an article in the 14 days after the reply).

**What it does, step by step.**
1. **Propensity score.** Newcomers who get replies are not a random group: for example, they ask clearer,
   more substantive questions. So M8 first predicts, for each newcomer, the chance of getting a reply from their
   164 background facts. This predicted chance is the propensity score. Each person's score comes from a model
   that did not see that person (5-fold cross-fitting: split people into 5 parts, predict each part with a model
   trained on the other 4).
2. **Strata.** Sort everyone by that score and cut them into 10 equal-sized groups (strata). Inside one group,
   everyone had about the same chance of a reply. So we compare each replied newcomer with similar newcomers in
   the same group who did not get a reply.
3. **ATT** (average effect on the treated). In each group, take the share of replied newcomers who kept editing
   minus the share of non-replied newcomers who kept editing. Average these 10 differences, giving more weight to
   groups with more replied newcomers. The result answers: "for the newcomers who did get a reply, how much did
   the reply raise their chance of editing?"
4. **Bootstrap** (for the uncertainty range). Redraw the sample of newcomers at random with replacement 500
   times, recompute the ATT each time, and take the middle 95% of these values as the 95% confidence interval
   (CI). The main version keeps the 10 groups fixed; a second version resamples mentors instead of newcomers; a
   separate "full bootstrap" (file `full_bootstrap.csv`) also refits the propensity score in every redraw.
5. **Balance.** Check that, after grouping, replied and non-replied newcomers look alike on every background
   fact. The measure is the standardized mean difference (SMD): the gap between group averages divided by the
   spread. Below 0.1 is the usual "balanced" line. A love plot is the chart of these SMDs.
6. **Extra checks.** E-value (how strong a hidden factor would have to be to explain the effect away), other
   sample and treatment definitions (robustness), other time windows, subgroups by question type, tests of
   whether the effect differs between subgroups, and a model of which reply features go with more editing.

**What comes out.** All in `results/m8/`, one CSV per table (`summary.md` gives a short overview):
- `B_main_results.csv` (Table 4): one row per outcome (6 rows). Columns: `Naive` (raw difference, replied minus
  not replied), `ATT` (adjusted effect), `CI_lo` and `CI_hi` (95% CI), `Sig` (`*` if the CI excludes 0).
- `B_mentor_boot.csv`: same 6 outcomes, with CIs from resampling newcomers and from resampling mentors.
- `full_bootstrap.csv`: same 6 outcomes, with the fixed-strata CI and the full-bootstrap CI side by side.
- `A3_strata_diagnostics.csv`: one row per stratum (10 rows): size, number replied and not replied, score range.
- `balance_att_weighting.csv` and `A4_covariate_balance.csv`: one row per background fact (164 rows), with the
  SMD before (`raw`) and after (`strat`) grouping.
- `A5_common_support.csv`: one row: the score range shared by both groups and how many people fall outside it.
- `evalues.csv`: three rows: the E-value computed three ways (the paper's old formula, an older code version, and
  the corrected one).
- `robustness.csv`, `C1_*`, `C3_*`, `C4_*`, `C7_*`, `window_followup_complete.csv`: one row per alternative
  setting, each with its ATT and CI.
- `subgroup_att.csv`, `subgroup_diagnostics.csv` (Table 5) and `a2_interaction_results.csv`: one row per
  subgroup (or moderator) and outcome.
- `reply_association.csv` and `reply_vif.csv` (Tables 6 and 7): one row per reply feature, with its coefficient
  and p value (association within replied newcomers only).
- `A7_feature_ablation.csv`: one row per choice of background facts used in the propensity score.

**How to read this notebook.** Each check prints `[OK ]` (passed) or `[FAIL]` (did not pass). The notebook was
already run, and all outputs below are saved from the run on the final data (cutoff 2026-04-20T14:00Z), so nothing needs to be run.
Short notes marked **What this checks.** and **What the output shows.** explain each step in plain words.
In the saved run all 20 checks pass.
"""),
md("""
## 1. What this module does, and how to use this notebook (technical summary)

**Module**: `pipeline/m8_analysis/run_analysis.py`, which reuses `legacy_parity.py` (code that reproduces the
paper version cell by cell) and `estimators.py`. Issues: I-10, I-11, I-19, I-24, I-26, I-35, I-36.

M8 answers the research question: does a reply to a newcomer's first question increase later editing?
It compares newcomers who got a reply (treated) with similar newcomers who did not (control):

1. **Propensity score**: for each person, the predicted chance of getting a reply given the 164 covariates
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

**How to use**: the saved outputs can be read as they are. To rerun, run from top to bottom. Every check prints
`[OK ]` or `[FAIL]`. Section 5 rebuilds the main result from the npz with a few lines of sklearn, so you can see
each step. It also follows random people into the estimate. To draw other random people, change `SEED` in the
next cell.

**What this checks.** The next cell loads the M7 dataset and keeps only the main sample.
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
# main analysis sample: in_sample and a full 14-day follow-up (the 14-day window ends before the cutoff, I-10)
MAIN = ALL["in_sample"] & (ALL["eligible_14d"] == 1)
D = {k: (v[MAIN] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n_all else v) for k, v in ALL.items()}
print(f"in_sample: {int(ALL['in_sample'].sum())} people, of whom {int(MAIN.sum())} have a full 14-day follow-up")
# outcome keys used for the 6 rows of Table B (label in the result table -> npz key)
OUTCOMES = [("1+_mainspace_edit_14d", "oc_primary"), ("n_mainspace_edits_14d", "oc_n_mainspace_edits_14d"),
            ("active_days_30d", "oc_active_days_30d"), ("unique_namespace_14d", "oc_unique_ns"),
            ("reverted_any_14d", "oc_reverted_any"), ("2+_active_days_14d", "oc_cross_day_any_14d")]
ck = Checks()
print(f"main sample: {len(D['y_treat'])} people, {int(D['y_treat'].sum())} treated")
'''),
md("""
**What the output shows.** `in_sample` has 34,073 newcomers. 555 of them asked so close to the cutoff that their
14-day window is not fully observed, so the main analysis keeps 33,518, of whom 25,760 (77%) got a reply. So most
askers do get a reply; the comparison group (no reply) is 7,758 people. Each outcome also uses only people whose
own window is observed: the 30-day outcome uses the 32,883 people with a full 30-day follow-up.
"""),
md("""
## 2. Files and manifest

M8 must have run **after** the npz it read was written, and the npz must not have changed since (same sha256).
Otherwise the result tables are out of date.

**What this checks.** That the result tables were made from the current dataset, so the paper numbers are not
stale.
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
**What the output shows.** M8 finished after M7, used 500 bootstrap draws, and read the same npz that is on disk
now (`True`). The manifest already shows the main result: ATT 0.0184, CI 0.0062 to 0.0299. The `followup_*`
files are older: they come from `followup.py`, which already used only people with a full follow-up, so the
change to the main sample did not require rerunning it. `descriptive_*` and `full_bootstrap.csv` are newer; they
are written by separate scripts after `run_analysis.py`.
"""),
md("""
## 3. Invariants of the result tables

**What this checks.** Basic sanity rules every correct result table must follow: right inputs, B = 500, sample
size matches, every estimate inside its own CI, the `*` mark consistent with the CI, 10 non-overlapping strata,
good balance, almost everyone inside common support (the range of scores where both groups have people), and an
E-value of at least 1.
"""),
code(r'''
B = t("B_main_results"); A3 = t("A3_strata_diagnostics"); A5 = t("A5_common_support"); BAL = t("balance_att_weighting")
EV = t("evalues"); ROB = t("robustness"); TXT = t("s13_text_numbers").iloc[0]
d = D["y_treat"].astype(float)
ck.add("M8 used the current npz (same sha256)", same_npz)
ck.add("M8 ran after the latest M7 run", pd.Timestamp(man["finished_at"]) > pd.Timestamp(man7["finished_at"]))
ck.add("bootstrap B = 500 (final run, not a debug run)", man["params"]["n_boot"] == 500)
ck.add("manifest N and treated equal the main-sample rows of the npz (in_sample, full 14-day follow-up)", man["counts"]["N"] == len(d) and man["counts"]["treated"] == int(d.sum()))
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
**What the output shows.** All 13 rules pass. In particular, after grouping no background fact has
|SMD| > 0.1, and fewer than 1% of people (in fact 2 of 33,518) fall outside common support.
"""),
md("""
## 4. Comparison with the paper version (same code, paper-version data)

`results/m8_legacy/` holds the same tables computed on the paper's npz. Same code, different data: any
difference comes from the data fixes. Both versions use 500 bootstrap draws.

**What this checks.** How much the corrected data changed the main table, and whether the CIs hold up when the
resampling is done by mentor (because one mentor answers many newcomers, their newcomers are not fully
independent).
"""),
code(r'''
cols = ["Outcome", "Naive", "ATT", "CI_lo", "CI_hi", "Sig"]
cmp = B[cols].merge(t("B_main_results", L)[cols], on="Outcome", suffixes=("_new", "_paper"))
display(cmp.round(4))
print("\nMentor-clustered bootstrap (resample mentors instead of newcomers):")
display(t("B_mentor_boot").round(4))
'''),
md("""
**What the output shows.** On the final data the main effect is 0.0184 (1.84 percentage points, CI 0.62 to
2.99), up from 0.0165 on the paper-version data. 5 of 6 outcomes are significant, against 3 of 6 before; the
raw differences also grew (4.62 vs 3.60 points), so the change comes from the data fixes, not the method.
The number of article edits (+0.78, CI -0.06 to 1.53) is borderline: not significant here, but significant when
mentors are resampled (0.03 to 1.54), with 5 strata, and in the full bootstrap. Counts like this have a long
tail (a few newcomers make hundreds of edits), so their CIs are wide and move easily. Resampling mentors changes
the main CI very little (0.0066 to 0.0301). One point a reviewer might notice: "reverted at least once" also
rises with a reply (+0.85 points; not significant when mentors are resampled). This is plausible, since people
who edit more also get reverted more; it is not a sign of harm by itself.
"""),
md("""
## 5. Independent reproduction, random people, tracer people

### 5.1 Propensity score, strata and ATT, rebuilt with plain sklearn

Steps: stack the covariate blocks E, Qtext, Qpersp, Qtype, emb20, temporal (the M block is not used in the main
model); standardize; 5-fold cross-fitted logistic regression (C = 1, lbfgs, StratifiedKFold with shuffle and
random_state 42); cut into 10 quantile strata; ATT as defined in section 1. The naive difference is the plain
treated mean minus control mean, with no adjustment.

**What this checks.** That the main numbers can be rebuilt from scratch with a few lines of standard code,
independently of the pipeline. If they match, the pipeline does what it claims.
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
    full = D["eligible_30d" if key == "oc_active_days_30d" else "eligible_14d"] == 1   # the outcome's own window ends before the cutoff
    y, dk = D[key].astype(float)[full], d[full]
    row = B.set_index("Outcome").loc[label]
    ok.append(compare({"N": int(full.sum()), "Naive": y[dk == 1].mean() - y[dk == 0].mean(), "ATT": my_att(y, dk, stratum[full])},
                      {"N": int(row.N), "Naive": row.Naive, "ATT": row.ATT}, title=f"\n{label}  (npz key {key})"))
print(f"\nAUC: recomputed {auc:.6f} | M8 {TXT.auc:.6f}")
ck.add("independent sklearn reproduction: N, naive difference and ATT of all 6 outcomes equal Table B", all(ok))
ck.add("independent reproduction: AUC equals M8", np.isclose(auc, TXT.auc))
y1 = D["oc_primary"].astype(float)
ATT = my_att(y1, d, stratum)
'''),
md("""
**What the output shows.** For all 6 outcomes the raw difference and the ATT match Table B exactly, and so
does the AUC (0.6256). The AUC measures how well the background facts predict who gets a reply (0.5 is a coin
flip, 1.0 is perfect). 0.62 means replies are only weakly predictable, which is good for the comparison: replied
and non-replied people overlap a lot. For the main outcome, the raw gap is 4.62 points but the adjusted effect
is 1.84: more than half of the raw gap comes from replied newcomers already being more likely to edit (for
example, they ask more substantive questions).
"""),
md("""
### 5.2 Per-stratum table for the main outcome (1+ mainspace edit in 14 days)

`weight` = share of all treated people who are in that stratum. ATT = sum of weight times (treated mean minus
control mean). The last column compares the stratum sizes with M8's table A3.

**What this checks.** The ATT broken into its 10 pieces, so a reader can see where the effect comes from.
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
**What the output shows.** The share who edit rises steadily with the score (non-replied: 9% in stratum 0,
33% in stratum 9). That is why the raw comparison is too large: people likely to get a reply are also likely to
edit anyway. Inside the strata, the replied group edits more in all 10 strata (from +0.2 points in stratum 6 to
+3.4 points in strata 4 and 8). The weighted sum is exactly the ATT 0.0184, all strata are used, and the
counts match table A3. Note that the trailing `True` is just the return value of the last line, not an extra
check.
"""),
md("""
### 5.3 E-value, recomputed

p1 = share of treated people with the outcome. Without the reply, the treated group would have p1 minus ATT.
Risk ratio RR = p1 / (p1 - ATT), and E-value = RR + sqrt(RR x (RR - 1)). For the CI bound, use the lower CI
limit in place of the ATT. An E-value of 1.4 means an unmeasured confounder would need a risk ratio of about 1.4
with both the reply and the outcome to explain the effect away.

**What this checks.** That the E-value in the paper uses the right formula, and shows the two older formulas
for comparison.
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
**What the output shows.** 22.2% of replied newcomers edit; without the reply the estimate is 22.2 minus 2.1,
so the reply multiplies the chance by 1.09. The corrected E-value is 1.41 (1.20 for the CI bound): a hidden
factor would need to raise both the chance of a reply and the chance of editing by about 1.2 to 1.4 times, beyond
everything already controlled, to erase the effect. The first row (1.16, the formula in an earlier paper text)
is wrong because it treats the 2-point difference as a ratio; the second row (1.84) used the unadjusted numbers.
Only the third row should be cited.
"""),
md("""
### 5.4 Balance: standardized mean difference (SMD) before and after stratification

SMD = |treated mean - control mean| / pooled SD (pooled SD = sqrt of the mean of the two group variances).
After stratification the mean difference is averaged over strata with ATT weights (number treated). A common
rule of thumb is that |SMD| below 0.1 means the groups are balanced on that covariate.

**What this checks.** Whether grouping by the score really made replied and non-replied newcomers alike on
every background fact (this is the table behind the love plot).
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
**What the output shows.** Before grouping, 9 of 164 facts were unbalanced; the largest is
`q_substantive` (whether the question is a real editing question) at 0.40, followed by the first text-embedding
summary (0.34). After grouping, none is above 0.1; the largest is 0.03. So on everything we measured, the two
groups are now comparable. The numbers equal the pipeline's file exactly.
"""),
md("""
### 5.5 Random people: how individuals enter the estimate

The ATT is a weighted sum of outcomes: a treated person gets weight 1 / (number of treated in usable strata);
a control person gets minus (n treated in their stratum) / (all treated x n control in their stratum). So a
control person in a stratum with many treated and few controls counts a lot. The column `weight_x_N1` shows the
weight times the number of treated people: 1 for every treated person, and minus (treated per control in the
stratum) for a control person. The check below confirms that the sum of weight x outcome over all people
equals the ATT.

**What this checks.** How single newcomers enter the estimate (5 random people: 3 replied, 2 not).
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
**What the output shows.** The weights add up to the ATT exactly. Each replied newcomer counts +1; each
non-replied newcomer counts between about -1 and -6, because in those strata there are 1 to 6 replied people
for every non-replied one, so each non-replied person stands in for several. Each link opens the person's first question
on Wikipedia, so a reader can look at real cases.
"""),
md("""
### 5.6 Tracer people in the main sample

**What this checks.** Where the fixed "tracer" accounts (followed through every module) end up in the analysis.
"""),
code(r'''
tr = tracer_frame().query("kind == 'mentee'")
T = P[P.mentee_uid.isin(tr.uid.astype(int))].merge(tr[["label", "uid"]].assign(uid=lambda x: x.uid.astype(int)), left_on="mentee_uid", right_on="uid")
display(T[["label", "mentee_uid", "treated", "e_hat", "stratum", "oc_primary", "n_edits", "q_substantive", "weight_x_N1"]].round(4))
print("tracer mentees not in the main sample:", sorted(set(tr.uid.astype(int)) - set(P.mentee_uid)))
'''),
md("""
**What the output shows.** 10 tracer newcomers are in the main sample (7 replied, 3 not), each with a score,
a stratum and a weight that follow the same rules as above. Two tracers are not in the main sample; they were
left out by the sample rules of earlier modules (see the M5 check), not by M8.
"""),
md("""
## 6. Edge-case lists

**What this checks.** Unusual cases that could distort the estimate: skipped strata, people outside common
support, extreme scores, the most heavily weighted controls, and background facts with no variation.
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
**What the output shows.** No stratum is skipped. Only 2 people lie outside common support (one replied
newcomer with a score of 0.090 and one non-replied with 1.000); both are unusual accounts with over 2,000 edits.
Two people out of 33,518 cannot move the result. The heaviest controls count -5.94 (stratum 9, where replied
people outnumber the others almost 6 to 1); that is a moderate weight, not an extreme one. One fact,
`q_has_unsigned`, is the same for everyone in the main sample, so it carries no information and does no harm.
"""),
md("""
## 7. Distributions and the other result tables

### 7.1 Propensity scores by group (common support)

If the two histograms overlap over the whole range, every treated person has comparable control people.

**What this checks.** Visually, whether replied and non-replied newcomers share the same range of scores.
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
**What the output shows.** Replied newcomers cluster at higher scores (most between 0.75 and 0.90), and
non-replied ones are spread a bit lower, but both groups have people in every bin from 0.25 to 1.00. There are
hundreds of non-replied newcomers even in the busiest bins, so every replied person has comparable people to be
compared with.
"""),
md("""
### 7.2 Ablation: which covariate blocks matter (table A7)
Each row fits the propensity score with only some blocks. If the ATT barely moves, the result does not hinge on
one block.

**What this checks.** Whether the result depends on one particular kind of background fact.
"""),
code(r'''
display(t("A7_feature_ablation").round(4))
'''),
md("""
**What the output shows.** Dropping any one block keeps the ATT between 0.018 and 0.021, all significant.
Using fewer facts (for example edit history only, 0.031) gives larger effects, which is what one expects when
less of the head start of replied newcomers is removed. Two rows to note: using the full 1,024-number embedding
leaves 15 facts unbalanced, so the 20-summary version is the better choice; adding mentor features predicts
replies much better (AUC 0.79, since some mentors reply far more than others) and gives 0.015, just short of
significance (CI -0.0001 to 0.0323). The CI of the "Full model" row (0.0070 to 0.0303) differs slightly from
Table B (0.0062 to 0.0299)
only because it uses a different set of random bootstrap draws; the point estimate is identical.
"""),
md("""
### 7.3 Robustness settings
Each row re-runs the whole estimate (new propensity score, new strata, 500 bootstrap draws) on a different
sample or treatment definition: without the D10 restriction, treatment = a reply from the assigned mentor
compared with no reply at all (newcomers answered only by others are left out, and the window starts at the
mentor's first reply; `mentor_reply_versions.csv` shows the simpler versions), without late-fetched people, only
questions at least 30 days before the cutoff, also people without a full 14-day follow-up (the main sample before
that rule), only people with no imputed text values, only people also in the paper sample.
Then: number of strata (C1), trimming extreme propensity scores (C3), and other pseudo start times for the
control group (C7).

**What this checks.** Whether the main result survives reasonable changes to the sample, the treatment
definition and the method. The pseudo start time: replied newcomers' 14 days are counted from the reply;
non-replied newcomers have no reply, so their 14 days start at the question time plus a delay drawn at random
from the real reply delays.
"""),
code(r'''
display(t("robustness").round(4))
display(t("C1_strata_sensitivity").round(4).head(12))
display(t("C3_ps_trimming").round(4))
display(t("C7_pseudo_time").round(4))
'''),
md("""
**What the output shows.** All 8 sample and treatment settings give an ATT between 0.016 and 0.021, all
significant; the row that also keeps the 555 people without a full 14-day follow-up gives 0.0208, the main value
before this rule. Using 5, 10 or 20 strata gives nearly the same result, and so does trimming extreme scores
(0.017 to 0.021). The pseudo start time matters more, and a reviewer should see this. With the main random draw,
the median delay or the average of many draws, the ATT is 0.017 to 0.018. With a short fixed delay (25th
percentile) it is slightly negative (-0.012, CI -0.025 to -0.001), with a long one (75th percentile) 0.037, and
with zero delay it turns clearly negative (-0.081). This is because non-replied newcomers are most active right after asking, so starting their
clock earlier counts more of their edits. The random draw gives both groups the same spread of start delays,
but the fixed-delay rows show that the result depends on the clock; this is the main open question for the
design (a risk-set matching analysis is planned).
"""),
md("""
### 7.4 Outcome windows
Main outcome measured over 7 to 180 days. `ATT` uses only people in `in_sample` whose full window lies before the
data cutoff, with the propensity score refitted on them. `ATT_main_strata` uses the people of the main analysis
sample whose window is observed, with the main strata kept fixed (table C4).

**What this checks.** Whether the effect holds over shorter and longer periods, and whether people whose window
is cut off by the data end (2026-04-20) change the result.
"""),
code(r'''
w = t("window_followup_complete")[["window", "N", "ATT", "CI_lo", "CI_hi", "Sig"]].merge(
    t("C4_window_sensitivity")[["window", "ATT"]], on="window", suffixes=("", "_main_strata"))
display(w.round(4))
'''),
md("""
**What the output shows.** The effect stays positive and significant from 7 to 180 days (0.020 at 7 days, 0.016
at 180 days), and the two columns agree to within 0.001. Every number here uses only people whose window is fully
observed. Before this rule the paper counted everyone, including people whose window was cut off by the data end;
that gave somewhat larger values (14 days: 0.0208; 180 days: 0.0228).
"""),
md("""
### 7.5 Subgroups (Table 5) and interaction tests (a2)
Subgroup ATTs use 5 strata inside each subgroup. A subgroup effect that looks larger is only meaningful if the
interaction test (difference between groups, with a bootstrap CI) excludes 0.

**What this checks.** For which kinds of questions a reply seems to help, and whether those differences are
real.
"""),
code(r'''
sg = t("subgroup_att")
display(sg[sg.Outcome.str.contains("1\\+|primary", regex=True)].round(4))
display(t("subgroup_diagnostics").round(4))
a2 = t("a2_interaction_results")
display(a2[a2.DV == "1+_mainspace_edit_14d"].round(4))
'''),
md("""
**What the output shows.** The "Full sample" row here is 0.0228, not 0.0184, because subgroup models use 5
strata and leave out the question-type facts (which define the subgroups); that is expected. The effect is
significant for technical (0.024), normative (0.034), own-work (0.034) and no-own-work (0.020) questions, and
not significant for referent, appraisal and non-substantive questions. But the interaction tests show that only
one difference is reliable: for referent questions the effect is about 4 points smaller (CI -0.069 to -0.012).
Two cautions: the appraisal subgroup is small (1,864 people) and still has 8 unbalanced facts, so its estimate is
rough; and with 8 tests, one significant result could partly be chance.
"""),
md("""
### 7.6 Reply features (Tables 6 and 7)
Within treated people only: a mixed model of the main outcome on standardized reply features (random intercept
per mentor), controlling for all covariates. `q_BH` is the Benjamini-Hochberg adjusted p value. These are
associations, not causal effects.

**What this checks.** Among replied newcomers, which features of the reply go along with more editing. `VIF`
measures how much a feature duplicates others; very high values make coefficients unstable.
"""),
code(r'''
display(t("reply_association").sort_values("p").round(4))
display(t("reply_vif").round(3))
'''),
md("""
**What the output shows.** The clearest pattern is speed: slower replies go with less editing (coefficient
-0.036 per standard deviation, the strongest by far). Replies that are closer in meaning to the question go with
more editing (+0.014). After correcting for many tests (q below 0.05), these two remain, plus three small terms
just under the line (q = 0.048): a negative "we" rate, and positive sentiment and subjectivity. Two technical notes: `Resource links` and `Wikilinks` were dropped because they nearly
duplicate each other (VIF about 52); and `ICC` is 0.0, meaning the model found no extra mentor-to-mentor
variation, so the mixed model gives the same answer as an ordinary regression here.
"""),
md("""
### 7.7 Long-term effect on one cohort (paper Table "later") and the monthly figure
`followup.py` analysis d. One cohort: newcomers whose start time (the reply, or the pseudo-reply time for
controls) is at least 365 days before the cutoff, so every period is observed for the same people. For each of the
periods days 1-14, 15-28, 29-60, 61-180 and 181-365 the outcome is at least one (or at least five) article edits
in that period. `q_BH` is the Benjamini-Hochberg q value over the eight later estimates (days 1-14 are only a
reference). The cell below rebuilds the cohort, the edit counts per period, the propensity score and the ATT with
plain code, independently of the pipeline, and recomputes the q values from the pipeline's bootstrap p values.

**What this checks.** That the long-term table in the paper is computed on the people and periods it says, and
that the multiple-testing correction covers exactly the eight later estimates.
"""),
code(r'''
import pyarrow.dataset as pads
LT = t("followup_cohort_periods")
s5 = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet", columns=["first_question_revid", "mentee_uid", "anchor"])
s5 = s5.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[D["cid_order"].astype(np.int64)]
anc = pd.to_datetime(s5.anchor, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
cutoff = np.datetime64(pd.Timestamp(config.DATA_CUTOFF).tz_localize(None), "s")
coh = anc + np.timedelta64(365, "D") <= cutoff
uid_c = s5.mentee_uid.astype(np.int64).to_numpy()[coh]
dc = d[coh]
print(f"cohort: {int(coh.sum())} people ({int(dc.sum())} treated); latest start time {pd.Timestamp(anc[coh].max())}")
ed = pads.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(columns=["uid", "timestamp", "ns"],
        filter=(pads.field("uid").isin(list(uid_c))) & (pads.field("ns") == 0)).to_pandas()
pos = {u: i for i, u in enumerate(uid_c)}
ei = ed.uid.map(pos).to_numpy()
edays = (pd.to_datetime(ed.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]") - anc[coh][ei]) / np.timedelta64(1, "D")
Xc = StandardScaler().fit_transform(X[coh][:, X[coh].std(axis=0) > 0])
e_c = cross_val_predict(LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42), Xc, dc,
                        cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
s_c = pd.qcut(e_c, 10, labels=False, duplicates="drop").astype(int)
rows = []
for period, lo, hi in [("days 1-14", 0, 14), ("days 15-28", 14, 28), ("days 29-60", 28, 60), ("days 61-180", 60, 180), ("days 181-365", 180, 365)]:
    n = np.bincount(ei[(edays > lo) & (edays <= hi)], minlength=len(uid_c))
    for thr in (1, 5):
        y = (n >= thr).astype(float)
        rows.append({"period": period, "threshold": f"{thr}+", "N_mine": len(y), "treated_mean_mine": y[dc == 1].mean(),
                     "ATT_mine": my_att(y, dc, s_c)})
MINE = pd.DataFrame(rows).merge(LT, on=["period", "threshold"])
later = MINE.period != "days 1-14"
from statsmodels.stats.multitest import multipletests
MINE["q_mine"] = np.nan
MINE.loc[later, "q_mine"] = multipletests(MINE.loc[later, "p_boot"], method="fdr_bh")[1]
display(MINE[["period", "threshold", "N", "N_mine", "Treat_mean", "treated_mean_mine", "ATT", "ATT_mine", "CI_lo", "CI_hi",
              "p_boot", "q_BH", "q_mine"]].round(4))
ck.add("long-term table: cohort size, treated means and ATTs equal an independent recomputation",
       bool((MINE.N == MINE.N_mine).all() and np.allclose(MINE.Treat_mean, MINE.treated_mean_mine)
            and np.allclose(MINE.ATT, MINE.ATT_mine, atol=1e-10)))
ck.add("long-term table: q values are BH over exactly the eight later estimates",
       bool(np.allclose(MINE.loc[later, "q_BH"], MINE.loc[later, "q_mine"]) and MINE.loc[~later, "q_BH"].isna().all()))
MO = t("followup_monthly")
mi = np.ceil(edays / 30).astype(int)
okm = (edays > 0) & (mi >= 1) & (mi <= 12)
YM = np.zeros((len(uid_c), 12)); YM[ei[okm], mi[okm] - 1] = 1
ck.add("monthly figure: treated share per month equals an independent recomputation",
       bool(np.allclose(MO.Treat_share, YM[dc == 1].mean(axis=0))))
display(MO[["month", "Treat_share", "Ctrl_share_reweighted"]].round(3).T)
'''),
md("""
**What the output shows.** The cohort has 19,512 people (14,858 treated); the latest start time is
2025-04-20, one year before the cutoff. The recomputed cohort, treated means and ATTs equal the pipeline's for all
ten rows, and the q values equal a Benjamini-Hochberg correction over the eight later rows. One later estimate has
q below 0.05: at least one article edit in days 181-365, +1.46 points (9.2% of treated against 7.7% of reweighted
controls). In days 1-14 this cohort's effect is +2.49 points, larger than the 1.84 of the full sample. The monthly
shares fall from 22.5% of treated people in month 1 to 9.5% in month 2 and 3.5% in month 12.
"""),
md("""
### 7.8 Edit quality (paper paragraph "Edit quality", Appendix Tables "quality" and "quality_self")
`quality.py`. An article edit counts as rejected if another editor reverted it within 48 hours. Every newcomer is
in one of three groups by their article edits in the 14 days after the start time: (1) no article edit, (2) at
least 80% of the edits kept, (3) less than 80% kept. Edits that the newcomer reverted themself are left out with
the reverting edit. Only newcomers whose 14-day window ends at least 48 hours before the cutoff are used.

**What this checks.** That the three groups cover everyone exactly once (their shares add up to 1 and their ATTs
add up to 0), that group 1 differs from "no article edit" in the main outcome only by the newcomers whose edits were
all reverted by themselves, and the counts quoted in the appendix.
"""),
code(r'''
QC, QS, QR = t("quality_cuts"), t("quality_self_versions"), t("quality_revert_stats")
display(QC[["cut", "g1_Ctrl_reweighted", "g1_ATT", "g2_Ctrl_reweighted", "g2_ATT", "g2_CI_lo", "g2_CI_hi",
            "g3_Ctrl_reweighted", "g3_ATT", "g3_CI_lo", "g3_CI_hi"]].round(4))
display(QS.round(4)); display(QR.T)
ck.add("edit quality: group shares add up to 1 and group ATTs add up to 0 for every cut",
       bool(np.allclose(QC.g1_Ctrl_reweighted + QC.g2_Ctrl_reweighted + QC.g3_Ctrl_reweighted, 1)
            and np.allclose(QC.g1_ATT + QC.g2_ATT + QC.g3_ATT, 0, atol=1e-10)))
s5q = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet", columns=["first_question_revid", "anchor"])
s5q = s5q.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[D["cid_order"].astype(np.int64)]
aq = pd.to_datetime(s5q.anchor, utc=True).dt.tz_localize(None)
fq = (aq + pd.Timedelta(days=16) <= pd.Timestamp(config.DATA_CUTOFF).tz_localize(None)).to_numpy()
no_edit_treated = 1 - D["oc_primary"][fq][d[fq] == 1].mean()
gap = QC.g1_Treat.iloc[0] - no_edit_treated
print(f"N with 48 hours after the window: {int(fq.sum())} (quality.py {int(QR.N[0])}); treated share in group 1 {QC.g1_Treat.iloc[0]:.5f}, "
      f"treated share without an article edit {no_edit_treated:.5f}")
ck.add("edit quality: same sample as the 48-hour rule needs, and group 1 = no article edit up to the self-revert-only newcomers",
       bool(int(fq.sum()) == int(QR.N[0]) and 0 <= gap * d[fq].sum() <= QR.newcomers_only_self_reverted_pairs[0] + 1e-9))
'''),
md("""
**What the output shows.** 33,452 newcomers have 48 hours of observation after their 14-day window. In every row the
three group shares add up to 1 and the three ATTs add up to 0. Group 1 holds 77.89% of treated newcomers, against
77.83% without any article edit; the difference is the newcomers whose only edits were reverted by themselves (21 in
both groups together). With the 80% cut, group 2 rises by 1.9 points and group 3 does not change (+0.0, CI -0.5 to
+0.7). Of 16,381 tagged article edits in the windows, 98.6% were matched to a reverting revision; of the 12,666
reverted within 48 hours, 1,375 were reverted by the newcomer, a median of 1.5 minutes after the edit.
"""),
md("""
### 7.9 Mentor pool over time (paper Appendix "Mentor Pool over Time", Figure "mentor_pool")
`descriptive.py` part e. Month-end counts of mentors in the automatic assignment pool and of mentors who take manual
claims only (from the M1 roster history), the number of main-sample newcomers who asked in each month, and that
number divided by the automatic pool.

**What this checks.** The month-end counts are recomputed here from the M1 status periods with a different method
(a month-by-period matrix instead of a loop), and the askers per month from the M7 dataset, not from the pipeline's
sample function. The counts quoted in the paper are printed, and the two published pool sizes (Phabricator T323048:
83 in November 2022, 187 in February 2025) are compared with ours.
"""),
code(r'''
MP, MS = t("descriptive_mentor_pool_by_month"), t("descriptive_mentor_pool_summary")
per = pd.read_parquet(config.INTERIM / "m1" / "mentor_periods.parquet")
ps = pd.to_datetime(per.start, utc=True).to_numpy()
pe = pd.to_datetime(per.end, utc=True).fillna(pd.Timestamp("2100-01-01", tz="UTC")).to_numpy()
ends = (pd.to_datetime(MP.month + "-01", utc=True) + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)).to_numpy()
on = (ps[None, :] <= ends[:, None]) & (pe[None, :] > ends[:, None])
auto_ = (on & (per.in_auto_pool == True).to_numpy()[None, :]).sum(1)   # noqa: E712
man_ = (on & (per.in_auto_pool == False).to_numpy()[None, :]).sum(1)   # noqa: E712
qm = pd.to_datetime(pd.Series(D["q_ts"]), utc=True).dt.strftime("%Y-%m")
ask = qm.value_counts().reindex(MP.month).to_numpy()
print("months:", len(MP), "| automatic pool equal:", bool((auto_ == MP.auto_pool).all()),
      "| manual only equal:", bool((man_ == MP.manual_only).all()), "| askers equal:", bool((ask == MP.askers).all()))
ck.add("mentor pool: month-end counts and askers per month recomputed independently",
       bool((auto_ == MP.auto_pool).all() and (man_ == MP.manual_only).all() and (ask == MP.askers).all()
            and np.allclose(ask / auto_, MP.askers_per_pool_mentor)))
ml = pd.read_parquet(config.INTERIM / "m1" / "mentor_list.parquet")
nm = pd.Series(D["mentor_ids"]).astype(str).nunique()
print(f"mentors ever {ml.mentor_uid.nunique()} (automatic pool {int(ml.ever_in_auto_pool.sum())}); at the cutoff {int(ml.is_current.sum())}; "
      f"main-sample mentors {nm}")
pub = MP.set_index("month").loc[["2022-11", "2025-02"], "auto_pool"]
print("published 83 and 187 against ours:", pub.to_dict())
ck.add("mentor pool: summary counts match, and ours are within 2 of the published pool sizes",
       bool(ml.mentor_uid.nunique() == MS.mentors_ever[0] and nm == MS.sample_mentors[0]
            and abs(pub["2022-11"] - 83) <= 2 and abs(pub["2025-02"] - 187) <= 2))
display(MP[MP.month.isin(["2021-05", "2022-03", "2022-11", "2023-03", "2023-04", "2023-07", "2023-10", "2025-02", "2025-03", "2026-03"])].round(2))
'''),
md("""
**What the output shows.** For all 59 months the recomputed automatic-pool and manual-only counts and the askers per
month equal the pipeline's. 450 mentors were on the list at some time (443 of them in the automatic pool) and 291
at the cutoff; the main-sample newcomers were assigned to 400 of them. The automatic pool had 85 mentors in November
2022 and 188 in February 2025 (published: 83 and 187), and rose from 91 to 157 in April 2023. Askers per pool mentor
rose from 1.2 to 1.9 in July 2023, from 2.6 to 5.0 in October 2023 and from 4.3 to 6.7 between January and March
2025, the months in which the visible share rose.
"""),
md("""
## 8. Summary (written to results/m8_report.json)

**What this checks.** All checks in one list, plus the main numbers saved to a small file for later modules.
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
md("""
**What the output shows.** All 27 checks pass. In the run on the final data (cutoff 2026-04-20T14:00Z), a reply from the mentor raises the
chance of at least one article edit in the next 14 days by 1.84 percentage points (95% CI 0.62 to 2.99), from a
base of 17.5% among non-replied newcomers (a relative increase of about 10%). The raw difference is 4.62 points;
the rest is explained by replied newcomers already being more active. The full bootstrap in
`results/m8/full_bootstrap.csv` (not shown above), which also re-estimates the scores in each draw, gives a CI
of 1.17 to 3.20 points, so the conclusion holds there too.
"""),
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
