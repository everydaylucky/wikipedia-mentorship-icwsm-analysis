"""
build_m7_check.py: writes checks/m7_check.ipynb (checks of the M7 PSM dataset and data dictionary)

The notebook follows the fixed 8-section layout of PLAN.md section 12:
    1 module description   2 files, manifest and data dictionary   3 invariants (every row, every block)
    4 comparison with the paper version   5 tracer and random samples (raw data next to processed values)
    6 edge-case lists   7 distributions   8 summary (results/m7_report.json)

Every E-block variable is recomputed in the notebook with its own code (it does not call
block_edit_history), for every person and, in detail, for random people.

Run: python checks/build_m7_check.py [--run]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks._nbgen import code, md, run_as_script, write_notebook  # noqa: E402

CELLS = [
md("""
# M7 PSM dataset: check notebook

## 1. What this module does, and how to use this notebook

**Module**: `pipeline/m7_dataset/build_dataset.py`. Issues: I-01 to I-05, I-08, I-29, I-33, I-37. Rules: R7.2, R7.8.

M7 builds the analysis dataset: **one row per newcomer** who asked a first question to a mentor. Each row holds
the treatment (did the question get a reply), the outcomes (later editing, copied from M5) and the
**covariates**, which describe the person and the question using only information from **before** the question
time `q_ts`. The covariates come in blocks:

| block | what it is | where it comes from |
|---|---|---|
| E (32) | edit history before the question: number of edits, edits in the last 7 days / 24 hours, namespaces, size changes, reverts, tags, logs, abuse filter hits, time of day | M3 raw edits, logs, abuse filter, profile |
| Qtext (37) | simple text features of the question (length, links, sentiment, politeness) | M6 |
| Qpersp (8) | Perspective API scores (toxicity and so on) | M6 |
| Qtype (5) | LLM labels of the question type | M6 |
| emb20 (20) | first 20 principal components (PCA) of the question embedding | M6 embeddings, PCA done here |
| temporal | year-month dummies (reference month 2021-05) and "was editing a page when asking" | M2/M5 |
| M (6) | mentor-side history before the question (used only in one ablation row) | M2 questions + M1 mentor periods |

Missing values (Perspective, embedding) are filled with the median and get a 0/1 missing indicator.
M7 also writes `psm_dataset_legacy_layout.npz`, the same data in the layout of the paper version, which M8 reads.

**Which paper numbers depend on it**: every number of the propensity score analysis. Main effect (Table 4),
covariate balance and common support, ablation, robustness checks, window and subgroup results (Table 5),
interaction tests, and the reply-feature models (Tables 6 and 7) all use the covariates and outcomes built here.

**How to use**: run the cells from top to bottom. Every check prints `[OK ]` or `[FAIL]`. Comparison tables
have a `match` column. In section 5 you see random people: their raw edits, logs and abuse filter hits are
printed next to the values in the dataset, and you can open their contributions on Wikipedia with the links.
To draw a fresh random sample, change `SEED` in the next cell and run again.
"""),
code(r'''
SEED = 0        # change this number to draw a fresh random sample of people in section 5
N_RANDOM = 5    # how many random people to show in detail

import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd, yaml
import pyarrow.dataset as ds
ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.common.io import iter_jsonl
from pipeline.common.tracer import tracer_frame
from checks._show import Checks, compare, pick, oldid, diff, contribs, logid
pd.set_option("display.max_colwidth", 70); pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)

df = pd.read_parquet(config.ANALYTIC / "psm_dataset.parquet")
dd = yaml.safe_load((config.ANALYTIC / "data_dictionary.yaml").read_text())
Z = np.load(config.ANALYTIC / "psm_dataset_legacy_layout.npz", allow_pickle=True)
Z = {k: Z[k] for k in Z.files}
man = json.loads((config.INTERIM / "m7" / "manifest.json").read_text())
B = dd["blocks"]
E_COLS = B["E"]
COV = [c for b, cols in B.items() if b != "M" for c in cols]      # covariates of the propensity score model
df["Q"] = pd.to_datetime(df.q_ts, utc=True)
ck = Checks()
print(f"rows {len(df)} | main sample (in_sample) {int(df.in_sample.sum())} | covariates {len(COV)} + M block {len(B['M'])}")
'''),
md("""
## 2. Files, manifest and data dictionary

### 2.1 Files and manifest
The manifest records when M7 ran and a fingerprint (sha256) of each input. If an input file changed after M7 ran,
the dataset is out of date and M7 should be run again.
"""),
code(r'''
for p in [config.ANALYTIC / "psm_dataset.parquet", config.ANALYTIC / "data_dictionary.yaml",
          config.ANALYTIC / "psm_dataset_legacy_layout.npz"]:
    print(f"  {p.name:36s} {p.stat().st_size / 1e6:8.1f} MB")
print("\nM7 finished at:", man["finished_at"], "| run time:", man["elapsed_sec"], "s | data cutoff:", man["data_cutoff"])
print("counts:", {k: v for k, v in man["counts"].items() if k != "missing_indicators"})
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inp = pd.DataFrame([{"input": Path(i["path"]).relative_to(ROOT).as_posix(), "unchanged_since_M7": sha(i["path"]) == i["sha256"]}
                    for i in man["inputs"]])
display(inp)
print("\nnpz arrays:", {k: Z[k].shape for k in ["X_E", "X_Qtext", "X_Qpersp", "X_Qtype", "X_emb20", "X_temporal", "X_M", "y_treat"]})
'''),
md("""
### 2.2 Data dictionary: every covariate, its meaning, block, missing rate, and where it is verified

`missing_rate` is the share of rows that were missing before imputation (from data_dictionary.yaml).
`verified in` says which cell of this notebook checks the variable against the raw data.
"""),
code(r'''
MEAN = {
 "n_edits": "edits before the question", "n_edits_7d": "edits in the 168 hours before the question",
 "n_edits_1d": "edits in the 24 hours before the question", "ns0_mainspace": "edits to articles (namespace 0)",
 "ns2_userpage": "edits to user pages (ns 2)", "ns3_usertalk": "edits to user talk pages (ns 3)",
 "ns4_wp": "edits to Wikipedia: pages (ns 4)", "ns118_draft": "edits to drafts (ns 118)",
 "n_unique_ns": "number of different namespaces edited", "mainspace_ratio": "ns0_mainspace / n_edits",
 "draft_ratio": "ns118_draft / n_edits", "avg_sizediff": "mean bytes added per edit",
 "std_sizediff": "population SD of bytes added", "max_sizediff": "largest absolute byte change",
 "neg_sizediff_ratio": "share of edits that removed bytes", "hours_since_last_edit": "hours from last edit to question (-1 if none)",
 "active_span_hours": "hours from first to last edit", "account_age_hours": "hours from registration to question (-1 if unknown)",
 "n_reverts": "edits already reverted before the question", "revert_rate_pre": "n_reverts / n_edits",
 "tag_visualeditor": "edits tagged visualeditor", "tag_mobile": "edits tagged mobile edit",
 "tag_newcomer_task": "edits tagged newcomer task", "tag_editcheck_newref": "edits tagged editcheck-newreference",
 "tag_discussion": "edits tagged discussiontools-added-comment", "log_create": "page creations (log create/create)",
 "log_thanks": "thanks log entries (thanks/thank)", "n_abuse": "abuse filter hits", "n_abuse_warn": "abuse filter hits with a warn result",
 "q_weekday": "weekday of the question (0 = Monday)", "q_hour_utc": "hour of the question (UTC)", "q_is_weekend": "question on Saturday or Sunday",
 "q_has_article_context": "the person was editing a page when asking", "m_prior_mentee_count": "questions this mentor got before",
 "m_prior_reply_rate": "share of those already answered before Q", "m_prior_avg_response_hours": "mean reply delay (hours) of those",
 "m_recent30d_mentee_count": "questions to this mentor in the 30 days before", "m_recent30d_reply_rate": "share of those answered before Q",
 "m_days_active": "days since the mentor first joined the roster",
 "q_has_sig": "body has a signature timestamp", "q_has_unsigned": "uses an {{unsigned}} template",
 "q_link_count": "internal links (not to user pages)", "q_template_count": "templates in the body",
 "q_has_url": "has an external link or bare URL", "q_mentions_wp": "links to Wikipedia: pages (incl. WP:)",
 "q_mentions_help": "links to Help: pages", "q_is_indented": "body starts with an indent mark (: or *)",
 "q_body_char_len": "characters in the plain text", "q_body_word_count": "words in the plain text",
 "q_vader_neg": "VADER negative share", "q_vader_neu": "VADER neutral share", "q_vader_pos": "VADER positive share",
 "q_vader_compound": "VADER overall sentiment (-1 to 1)", "q_tb_polarity": "TextBlob polarity (-1 to 1)",
 "q_tb_subjectivity": "TextBlob subjectivity (0 to 1)"}
def meaning(c, b):
    if c in MEAN: return MEAN[c]
    if b == "Qtext": return "question text feature: " + c.replace("q_poly_feature_politeness_", "ConvoKit politeness strategy ").replace("q_", "")
    if b == "Qpersp": return "Perspective score: " + c.replace("q_persp_", "")
    if b == "Qtype": return "LLM label (0/1): " + c.replace("q_", "")
    if b == "emb20": return "principal component of the question embedding"
    if c.startswith("ym_"): return f"question asked in {c[3:]} (0/1)"
    return ""
WHERE = {"E": "3 ranges; 5.1 all rows; 5.2 random people", "Qtext": "3; 5.4 all rows vs M6 + random people",
         "Qpersp": "3 range; 5.4 all rows vs M6; 5.5 imputation", "Qtype": "3 (0/1); 5.4 all rows vs M6",
         "emb20": "5.5 PCA redone, all rows", "temporal": "3 all rows; 5.6 random people", "M": "5.7 random people"}
rows = [{"variable": c, "block": b, "meaning": meaning(c, b), "type": dd["columns"][c]["type"],
         "missing_rate": dd["columns"][c]["missing_rate"], "verified in": WHERE[b]} for b, cols in B.items() for c in cols]
DD = pd.DataFrame(rows)
with pd.option_context("display.max_rows", 500):
    display(DD)
print("\nmissing indicators added by imputation:", dd["missing_indicators"])
'''),
md("""
## 3. Invariants

Each line is a rule that must hold for every row. A `[FAIL]` means the data break the rule.
"""),
code(r'''
s5 = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet")
s5 = s5[s5.in_sample | s5.in_sample_unrestricted].set_index("mentee_uid")
print("Keys and coverage")
ck.add("mentee_uid is unique", df.mentee_uid.is_unique)
ck.add("first_question_revid is unique", df.first_question_revid.is_unique)
ck.add("rows = people in the M5 sample (in_sample or in_sample_unrestricted)", set(df.mentee_uid) == set(s5.index))
ck.add("in_sample is a subset of in_sample_unrestricted", bool((~df.in_sample | df.in_sample_unrestricted).all()))
ck.add("data dictionary lists every covariate column, and every one is in the parquet",
       set(COV + B["M"]) <= set(dd["columns"]) and set(COV + B["M"]) <= set(df.columns))
ck.add("treat, outcomes and flags equal the M5 sample (all rows)",
       all((df.set_index("mentee_uid")[c] == s5.loc[df.mentee_uid, c]).all() for c in ["treat", "treat_48h", "treat_mentor", "in_sample"])
       and all(np.allclose(df[c].to_numpy(float), s5.loc[df.mentee_uid, c].to_numpy(float), equal_nan=True)
               for c in df.columns if c.startswith("oc_") and c in s5.columns))

print("\nNo missing values left in the propensity score covariates")
ck.add("E block has no missing values", bool(df[E_COLS].notna().all().all()))
ck.add("Qtext and Qtype have no missing values", bool(df[B["Qtext"] + B["Qtype"]].notna().all().all()))
ck.add("after imputation no covariate is missing", bool(df[COV].notna().all().all()))

e = df[E_COLS]
print("\nE block: logical ranges")
cnt = ["n_edits", "n_edits_7d", "n_edits_1d", "ns0_mainspace", "ns2_userpage", "ns3_usertalk", "ns4_wp", "ns118_draft", "n_unique_ns",
       "n_reverts", "tag_visualeditor", "tag_mobile", "tag_newcomer_task", "tag_editcheck_newref", "tag_discussion",
       "log_create", "log_thanks", "n_abuse", "n_abuse_warn"]
ck.add("all counts are whole numbers >= 0", bool(((e[cnt] >= 0) & (e[cnt] % 1 == 0)).all().all()))
ck.add("n_edits_1d <= n_edits_7d <= n_edits", bool(((e.n_edits_1d <= e.n_edits_7d) & (e.n_edits_7d <= e.n_edits)).all()))
ck.add("namespace counts add up to at most n_edits", bool((e[["ns0_mainspace", "ns2_userpage", "ns3_usertalk", "ns4_wp", "ns118_draft"]].sum(axis=1) <= e.n_edits).all()))
ck.add("each tag count, n_reverts <= n_edits", bool((e[[c for c in cnt if c.startswith("tag_")] + ["n_reverts"]].le(e.n_edits, axis=0)).all().all()))
ck.add("n_abuse_warn <= n_abuse", bool((e.n_abuse_warn <= e.n_abuse).all()))
ck.add("ratios are in [0, 1]", bool(e[["mainspace_ratio", "draft_ratio", "neg_sizediff_ratio", "revert_rate_pre"]].stack().between(0, 1).all()))
nz = e.n_edits > 0
ck.add("mainspace_ratio = ns0_mainspace / n_edits (people with edits)", np.allclose(e.mainspace_ratio[nz], (e.ns0_mainspace / e.n_edits)[nz]))
ck.add("revert_rate_pre = n_reverts / n_edits", np.allclose(e.revert_rate_pre[nz], (e.n_reverts / e.n_edits)[nz]))
ck.add("hours_since_last_edit = -1 exactly when n_edits = 0", bool(((e.hours_since_last_edit == -1) == ~nz).all()))
ck.add("leakage guard: hours_since_last_edit > 0 for everyone with edits (last edit strictly before Q)", bool((e.hours_since_last_edit[nz] > 0).all()))
ck.add("std_sizediff >= 0 and max_sizediff >= |avg_sizediff|", bool(((e.std_sizediff >= 0) & (e.max_sizediff + 1e-9 >= e.avg_sizediff.abs())).all()))
known = e.account_age_hours >= 0
ck.add("registration is before the first edit (account_age >= active_span + hours_since_last_edit)",
       bool((e.account_age_hours[known & nz] + 1e-6 >= (e.active_span_hours + e.hours_since_last_edit)[known & nz]).all()))
ck.add("q_weekday, q_hour_utc, q_is_weekend agree with q_ts (all rows)",
       bool(((e.q_weekday == df.Q.dt.weekday) & (e.q_hour_utc == df.Q.dt.hour) & (e.q_is_weekend == (df.Q.dt.weekday >= 5).astype(int))).all()))

print("\nQuestion blocks")
ck.add("Qtype labels are 0/1", bool(df[B["Qtype"]].isin([0, 1]).all().all()))
ck.add("Perspective scores are in [0, 1]", bool(df[B["Qpersp"]].stack().between(0, 1).all()))
qf6 = pd.read_parquet(config.INTERIM / "m6" / "question_features.parquet", columns=["first_question_revid", "q_persp_toxicity"]).set_index("first_question_revid")
p_nan = df.first_question_revid.map(qf6.q_persp_toxicity).isna()
ck.add("Perspective missing indicator = 1 exactly when the M6 score is missing", bool((df.q_persp_toxicity_missing == p_nan.astype(float)).all()))
bad_flag = df[p_nan & df.has_persp.astype(bool)]
ck.add("M6 flag has_persp is False whenever the M6 score is missing", bad_flag.empty)
if len(bad_flag):
    print(f"    DATA PROBLEM: {len(bad_flag)} people have has_persp = True but no Perspective score in M6 (they were imputed correctly;"
          " but the npz flag text_complete counts them as complete):")
    display(bad_flag[["mentee_uid", "first_question_revid", "in_sample", "question_text"]].assign(question_text=lambda x: x.question_text.str[:60]))
ck.add("embedding missing indicator = 1 exactly when has_emb_q is False", bool((df.emb_pc1_missing == (~df.has_emb_q.astype(bool)).astype(float)).all()))
ck.add("all indicators of one block are identical (so the npz keeps one per block)",
       bool((df[[c for c in dd["missing_indicators"] if c.startswith("q_persp")]].nunique(axis=1) == 1).all()
            and (df[[c for c in dd["missing_indicators"] if c.startswith("emb")]].nunique(axis=1) == 1).all()))

print("\nTemporal block")
ym = [c for c in B["temporal"] if c.startswith("ym_")]
month = df.Q.dt.strftime("%Y-%m")
hot = df[ym].to_numpy()
ck.add("each row has at most one year-month dummy equal to 1", bool((hot.sum(axis=1) <= 1).all()))
ck.add("the dummy that is 1 is the month of q_ts; all-zero rows are the reference month 2021-05",
       bool((np.where(hot.sum(axis=1) == 1, np.array([c[3:] for c in ym])[hot.argmax(axis=1)], "2021-05") == month).all()))
ck.add("q_has_article_context = editing_page", bool((df.q_has_article_context == df.editing_page.fillna(False).astype(float)).all()))

print("\nnpz in the paper layout (used by M8) equals the parquet, row by row")
ck.add("npz row order = parquet row order (first_question_revid)", bool((Z["cid_order"] == df.first_question_revid.to_numpy()).all()))
for k, cols in [("X_E", E_COLS), ("X_Qtext", B["Qtext"]), ("X_Qpersp", B["Qpersp"]), ("X_Qtype", B["Qtype"]),
                ("X_emb20", B["emb20"]), ("X_M", B["M"]), ("X_temporal", [str(c) for c in Z["temporal_cols"]])]:
    ck.add(f"{k} equals parquet columns ({len(cols)} columns)", np.array_equal(Z[k], df[cols].to_numpy(float)))
ck.add("temporal_cols = ym dummies + q_has_article_context + one missing indicator per block",
       [str(c) for c in Z["temporal_cols"]] == B["temporal"] + ["q_persp_toxicity_missing", "emb_pc1_missing"])
ck.add("y_treat, in_sample, mentee_ids equal the parquet", np.array_equal(Z["y_treat"], df.treat.to_numpy(float))
       and np.array_equal(Z["in_sample"], df.in_sample.to_numpy(bool)) and np.array_equal(Z["mentee_ids"], df.mentee_uid.astype(str).to_numpy()))
ck.add("every oc_* array equals the parquet column", all(np.allclose(Z[k], df[k].to_numpy(float), equal_nan=True) for k in Z if k.startswith("oc_")))
const = [c for c in COV if df.loc[df.in_sample, c].nunique() <= 1]
print("\nconstant columns in the main sample (they carry no information; StandardScaler turns them into 0):", const)
ck.report()
'''),
md("""
## 4. Comparison with the paper version

For people who are in both datasets (matched on the revid of the first question), how often each E-block
variable has exactly the same value. Differences are expected where the definition was fixed:
n_edits_1d / 7d now use hours (I-08), log_create counts only page creations (I-03), n_reverts counts only edits
already reverted before the question (R7.8), and edits that were fetched later (I-53).
"""),
code(r'''
z = np.load(config.EXTERNAL["paper_npz"], allow_pickle=True)
s7 = {int(r["conversation_id"]): r["revid"] for r in iter_jsonl(config.EXTERNAL["paper_s7"])}
P = pd.DataFrame(z["X_E"], columns=[str(c) for c in z["E_cols"]])
P["first_question_revid"] = [int(s7[int(c)]) if s7.get(int(c)) else -1 for c in z["cid_order"]]
m = df.merge(P, on="first_question_revid", suffixes=("", "_paper"))
print(f"paper rows {len(P)} | new rows {len(df)} | in both {len(m)}")
rows = []
for c in E_COLS:
    if f"{c}_paper" in m:
        a, b = m[c].astype(float), m[f"{c}_paper"].astype(float)
        rows.append({"variable": c, "same_value_share": round(float(np.isclose(a, b).mean()), 4),
                     "corr": round(float(np.corrcoef(a, b)[0, 1]), 4) if a.std() > 0 and b.std() > 0 else None,
                     "mean_new": round(a.mean(), 3), "mean_paper": round(b.mean(), 3)})
display(pd.DataFrame(rows))
'''),
md("""
## 5. Tracer samples and random samples: raw data next to processed values

### 5.1 Our own recomputation of all 32 E-block variables, for every person

The function `my_E` below is written from the definitions in section 2.2. It does **not** call the pipeline.
It reads the raw edits (`data/interim/m3/mentee_edits`), logs, abuse filter hits, profile (registration time)
and revert times, keeps only records with timestamp **strictly before** `q_ts`, and computes each variable.

Revert times: `data/raw/m3/revert_times*.jsonl` may still be growing (a collection job adds records). To
reproduce what M7 saw, we use only records fetched before M7 started. An edit counts as reverted before Q if it
carries the `mw-reverted` tag and either the revert time is unknown or the revert happened before Q.
"""),
code(r'''
m7_start = pd.Timestamp(man["finished_at"]) - pd.Timedelta(seconds=man["elapsed_sec"])
REV, late_rev = {}, 0
for name in ("revert_times_sample.jsonl", "revert_times.jsonl"):
    for r in iter_jsonl(config.RAW / "m3" / name):
        if pd.Timestamp(r["fetched_at"]) > m7_start:
            late_rev += 1; continue
        if r.get("revert_ts"):
            REV[int(r["revid"])] = pd.Timestamp(r["revert_ts"])
print(f"revert times known: {len(REV)} | records fetched after M7 started (not used here): {late_rev}")

uids = df.mentee_uid.tolist()
ED = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
    columns=["uid", "revid", "timestamp", "ns", "title", "sizediff", "tags", "comment"], filter=ds.field("uid").isin(uids)).to_pandas()
ED["ts"] = pd.to_datetime(ED.timestamp, utc=True)
LG = pd.read_parquet(config.INTERIM / "m3" / "mentee_logs.parquet"); LG = LG[LG.uid.isin(uids)].assign(ts=lambda d: pd.to_datetime(d.timestamp, utc=True))
AB = pd.read_parquet(config.INTERIM / "m3" / "mentee_abuse.parquet"); AB = AB[AB.uid.isin(uids)].assign(ts=lambda d: pd.to_datetime(d.timestamp, utc=True))
PROF = pd.read_parquet(config.INTERIM / "m3" / "mentee_profile.parquet").set_index("mentee_uid")
QT = df.set_index("mentee_uid").Q
TAGS = {"tag_visualeditor": "visualeditor", "tag_mobile": "mobile edit", "tag_newcomer_task": "newcomer task",
        "tag_editcheck_newref": "editcheck-newreference", "tag_discussion": "discussiontools-added-comment"}

def mark(ed):
    """Add Q, 'used' (strictly before Q) and 'reverted before Q' to a table of edits."""
    ed = ed.assign(Q=ed.uid.map(QT))
    ed["used"] = ed.ts < ed.Q
    tagged = ed.tags.map(lambda t: "mw-reverted" in list(t))
    rts = pd.to_datetime(ed.revid.map(REV), utc=True)
    ed["revert_ts"] = rts
    ed["reverted_before_Q"] = tagged & (rts.isna() | (rts < ed.Q))
    return ed

def my_E(people, strict=True):
    """All 32 E-block variables for the given mentee_uids. strict=False also uses edits AT q_ts (to show leakage)."""
    Qs = QT.loc[people]
    ed = mark(ED[ED.uid.isin(people)])
    pre = ed[ed.ts < ed.Q] if strict else ed[ed.ts <= ed.Q]
    pre = pre.assign(sd=pre.sizediff.fillna(0).astype(float), hrs=(pre.Q - pre.ts).dt.total_seconds() / 3600)
    g = pre.groupby("uid")
    out = pd.DataFrame(index=pd.Index(people, name="mentee_uid"))
    out["n_edits"] = g.size()
    out["n_edits_7d"] = g.hrs.apply(lambda h: (h <= 168).sum())
    out["n_edits_1d"] = g.hrs.apply(lambda h: (h <= 24).sum())
    for col, ns in [("ns0_mainspace", 0), ("ns2_userpage", 2), ("ns3_usertalk", 3), ("ns4_wp", 4), ("ns118_draft", 118)]:
        out[col] = g.ns.apply(lambda x, ns=ns: (x == ns).sum())
    out["n_unique_ns"] = g.ns.nunique()
    out = out.fillna(0)
    n = out.n_edits
    out["mainspace_ratio"] = (out.ns0_mainspace / n).where(n > 0, 0.0)
    out["draft_ratio"] = (out.ns118_draft / n).where(n > 0, 0.0)
    out["avg_sizediff"] = g.sd.mean().reindex(out.index).fillna(0.0)
    out["std_sizediff"] = g.sd.std(ddof=0).reindex(out.index).where(n > 1, 0.0).fillna(0.0)
    out["max_sizediff"] = g.sd.apply(lambda x: x.abs().max()).reindex(out.index).fillna(0.0)
    out["neg_sizediff_ratio"] = (g.sd.apply(lambda x: (x < 0).sum()).reindex(out.index).fillna(0) / n).where(n > 0, 0.0)
    out["hours_since_last_edit"] = g.hrs.min().reindex(out.index).fillna(-1.0)
    out["active_span_hours"] = ((g.ts.max() - g.ts.min()).dt.total_seconds() / 3600).reindex(out.index).where(n > 1, 0.0).fillna(0.0)
    reg = pd.to_datetime(PROF.registration.reindex(people), utc=True, errors="coerce")
    out["account_age_hours"] = ((Qs - reg).dt.total_seconds() / 3600).fillna(-1.0).to_numpy()
    out["n_reverts"] = g.reverted_before_Q.sum().reindex(out.index).fillna(0)
    out["revert_rate_pre"] = (out.n_reverts / n).where(n > 0, 0.0)
    for col, tag in TAGS.items():
        out[col] = g.tags.apply(lambda s, tag=tag: sum(tag in list(t) for t in s)).reindex(out.index).fillna(0)
    lg = LG[LG.uid.isin(people)]; lg = lg[lg.ts < lg.uid.map(QT)]
    out["log_create"] = lg[(lg.type == "create") & (lg.action == "create")].groupby("uid").size().reindex(out.index).fillna(0)
    out["log_thanks"] = lg[(lg.type == "thanks") & (lg.action == "thank")].groupby("uid").size().reindex(out.index).fillna(0)
    ab = AB[AB.uid.isin(people)]; ab = ab[ab.ts < ab.uid.map(QT)]
    out["n_abuse"] = ab.groupby("uid").size().reindex(out.index).fillna(0)
    out["n_abuse_warn"] = ab[ab.result.fillna("").str.lower().str.contains("warn")].groupby("uid").size().reindex(out.index).fillna(0)
    out["q_weekday"] = Qs.dt.weekday.to_numpy(); out["q_hour_utc"] = Qs.dt.hour.to_numpy()
    out["q_is_weekend"] = (Qs.dt.weekday >= 5).astype(int).to_numpy()
    return out[E_COLS].astype(float)

MINE = my_E(uids)
STORED = df.set_index("mentee_uid")[E_COLS].astype(float)
agree = pd.DataFrame({"variable": E_COLS,
                      "rows_that_agree": [int(np.isclose(MINE[c], STORED[c], atol=1e-6).sum()) for c in E_COLS],
                      "max_abs_diff": [float((MINE[c] - STORED[c]).abs().max()) for c in E_COLS]})
agree["share"] = (agree.rows_that_agree / len(df)).round(6)
with pd.option_context("display.max_rows", 50):
    display(agree)
E_BAD = [c for c in E_COLS if agree.set_index("variable").share[c] < 1]
ck.add("every E-block variable recomputed independently agrees with the dataset for every person", not E_BAD)
if E_BAD:
    bad = MINE.index[~np.isclose(MINE[E_BAD], STORED[E_BAD], atol=1e-6).all(axis=1)]
    print("MISMATCH in", E_BAD, "| people:", len(bad), "| first ids:", list(bad[:10]))
    display(pd.concat({"recomputed": MINE.loc[bad[:10], E_BAD], "in data": STORED.loc[bad[:10], E_BAD]}, axis=1))
'''),
md("""
### 5.2 Random people: raw edits, logs, abuse filter hits, and all 32 E-block values

For each random person we print the raw records. `used` = the record is strictly before the question and enters
the variables. Records at or after `q_ts` are shown too, marked `used = False`, so you can see they are left out.
Usually the first record at `q_ts` is the question edit itself (on the mentor's talk page).

The second table recomputes the 32 variables from the printed records and puts them next to the dataset.
The leakage check recomputes `n_edits` including edits at or after Q: if that number differs from the stored
value, the stored value clearly did not use those edits.
"""),
code(r'''
main = df[df.in_sample]
RAND = list(pick(main, N_RANDOM, SEED).mentee_uid) + list(pick(main[main.n_edits >= 10], 1, SEED).mentee_uid)
print("random people (the last one is drawn among people with at least 10 edits before Q):", RAND)

def show_person(uid, max_rows=12):
    r = df.set_index("mentee_uid").loc[uid]
    print("=" * 110)
    print(f"mentee_uid {uid} | name {r.mentee_name} | q_ts {r.q_ts} | treat {r.treat} | first question {oldid(r.first_question_revid)}")
    print("contributions:", contribs(r.mentee_name), "| registration:", PROF.registration.get(uid))
    ed = mark(ED[ED.uid == uid]).sort_values("ts")
    ed["tags"] = ed.tags.map(lambda t: ",".join(t)[:40])
    show = pd.concat([ed[ed.used].tail(max_rows), ed[~ed.used].head(3)])
    show = show.assign(link=show.revid.map(diff))[["revid", "timestamp", "used", "ns", "title", "sizediff", "tags", "reverted_before_Q", "link"]]
    print(f"\nedits: {int(ed.used.sum())} before Q (showing the last {min(max_rows, int(ed.used.sum()))}), {int((~ed.used).sum())} at or after Q (showing up to 3)")
    if len(show): display(show)
    lg = LG[LG.uid == uid].sort_values("ts").assign(used=lambda d: d.ts < r.Q)
    print(f"logs: {int(lg.used.sum())} before Q")
    if lg.used.any(): display(lg[lg.used][["logid", "timestamp", "type", "action", "title"]].tail(8))
    ab = AB[AB.uid == uid].sort_values("ts").assign(used=lambda d: d.ts < r.Q)
    print(f"abuse filter hits: {int(ab.used.sum())} before Q")
    if ab.used.any(): display(ab[ab.used][["abuse_id", "timestamp", "filter", "action", "result"]].tail(8))
    hand = my_E([uid]).iloc[0].to_dict()
    ok = compare(hand, STORED.loc[uid].to_dict(), title="\nE block: recomputed from the records above vs. the dataset")
    loose = my_E([uid], strict=False).iloc[0]
    n_after = int((~ed.used).sum())
    print(f"leakage check: edits at or after Q = {n_after}; n_edits if edits AT q_ts were used = {int(loose.n_edits)}; "
          f"stored n_edits = {int(r.n_edits)}; strict recomputation = {int(hand['n_edits'])}")
    leak_ok = int(r.n_edits) == int(ed.used.sum())
    return ok and leak_ok

res = {u: show_person(u) for u in RAND}
ck.add(f"random people ({len(RAND)}): all 32 E variables match and no edit at or after Q was used", all(res.values()))
'''),
md("""
### 5.3 Tracer samples (the fixed newcomers followed through every module)

Same recomputation for the fixed tracer newcomers (see `pipeline/common/tracer.py`). A tracer that is not in
the dataset was removed by an earlier sample filter (M4/M5), which those notebooks explain.
"""),
code(r'''
tr = tracer_frame().query("kind == 'mentee'")
rows = []
for t in tr.itertuples():
    uid = int(t.uid)
    if uid not in STORED.index:
        rows.append({"label": t.label, "uid": uid, "in dataset": False}); continue
    h = my_E([uid]).iloc[0]
    bad = [c for c in E_COLS if not np.isclose(h[c], STORED.loc[uid, c], atol=1e-6)]
    rows.append({"label": t.label, "uid": uid, "in dataset": True, "in_sample": bool(df.set_index("mentee_uid").in_sample[uid]),
                 "n_edits": int(h.n_edits), "n_reverts": int(h.n_reverts), "account_age_hours": round(h.account_age_hours, 1),
                 "all 32 match": not bad, "mismatched": ",".join(bad)})
TR = pd.DataFrame(rows)
display(TR)
ck.add("tracer newcomers in the dataset: all 32 E variables match", bool(TR.loc[TR["in dataset"], "all 32 match"].all()))
'''),
md("""
### 5.4 Question blocks: dataset vs. the M6 file, for random people

Qtext, Qpersp and Qtype are copied from `data/interim/m6/question_features.parquet` (joined on the revid of the
first question). The question text is printed so you can see what the features describe. For people whose M6
value is missing, the dataset holds the median instead (checked in 5.5).
"""),
code(r'''
QF = pd.read_parquet(config.INTERIM / "m6" / "question_features.parquet").set_index("first_question_revid")
QCOLS = B["Qtext"] + B["Qpersp"] + B["Qtype"]
DF = df.set_index("mentee_uid")
for uid in RAND:
    r = DF.loc[uid]; q = QF.loc[r.first_question_revid]
    print("=" * 110)
    print(f"mentee_uid {uid} | question {oldid(r.first_question_revid)}\n  text: {str(r.question_text)[:300]!r}")
    show = ["q_body_word_count", "q_link_count", "q_mentions_wp", "q_vader_compound", "q_poly_feature_politeness_==Gratitude==",
            "q_persp_toxicity"] + B["Qtype"]
    compare({c: q[c] for c in show}, {c: r[c] for c in show}, title="selected features: M6 file vs. dataset")
# all 50 columns, all rows that were not imputed
J = df[["first_question_revid", "has_persp"] + QCOLS].join(QF[QCOLS], on="first_question_revid", rsuffix="_m6")
diffs = {}
for c in QCOLS:
    okrow = J[f"{c}_m6"].notna()          # rows where M6 has a value (the others were imputed, see 5.5)
    diffs[c] = int((~np.isclose(J.loc[okrow, c].astype(float), J.loc[okrow, f"{c}_m6"].astype(float))).sum())
ck.add("all Qtext / Qpersp / Qtype values equal the M6 file (every row where M6 has a value)", sum(diffs.values()) == 0)
if sum(diffs.values()): print("MISMATCH counts:", {k: v for k, v in diffs.items() if v})
'''),
md("""
### 5.5 Imputation and the embedding PCA

* For a person with a missing Perspective score or embedding, the dataset value must be the **median** of that
  column over all rows, and the missing indicator must be 1.
* The 20 embedding components are recomputed here: PCA with 20 components (random_state 42) on the M6 question
  embeddings of everyone who has one. A principal component is only defined up to its sign, so we compare after
  aligning the sign.
"""),
code(r'''
raw = df[["first_question_revid"]].join(QF[B["Qpersp"]], on="first_question_revid")
miss_p = df.index[raw.q_persp_toxicity.isna()]
print(f"people with missing Perspective scores: {len(miss_p)}")
med = raw[B["Qpersp"]].median()
ck.add("missing Perspective scores were replaced by the column median",
       all(np.allclose(df.loc[miss_p, c], med[c]) for c in B["Qpersp"]) if len(miss_p) else True)
if len(miss_p):
    i = miss_p[0]
    compare({"q_persp_toxicity": med["q_persp_toxicity"], "q_persp_toxicity_missing": 1.0},
            {"q_persp_toxicity": df.loc[i, "q_persp_toxicity"], "q_persp_toxicity_missing": df.loc[i, "q_persp_toxicity_missing"]},
            title=f"example: mentee_uid {df.loc[i, 'mentee_uid']} (M6 value is missing)")

from sklearn.decomposition import PCA
EMB = np.load(config.INTERIM / "m6" / "emb_q_sample.npy")
pos = {rv: i for i, rv in enumerate(QF.index)}
full = EMB[[pos[rv] for rv in df.first_question_revid]]
has = ~np.isnan(full).any(axis=1)
ck.add("has_emb_q agrees with whether the M6 embedding row has values", bool((has == df.has_emb_q.astype(bool).to_numpy()).all()))
pca = PCA(n_components=20, random_state=config.SEED).fit(full[has])
mine = pca.transform(full[has])
stored = df.loc[has, B["emb20"]].to_numpy()
sign = np.sign((mine * stored).sum(axis=0))
maxdiff = float(np.abs(mine * sign - stored).max())
print(f"PCA redone: explained variance {pca.explained_variance_ratio_.sum():.4f} (manifest {man['counts']['pca_var']}); "
      f"largest difference to the dataset {maxdiff:.2e}")
ck.add("emb20 equals our own PCA of the M6 embeddings (up to sign)", maxdiff < 1e-4)
med_e = df.loc[has, B["emb20"]].median()
ck.add("people without an embedding hold the median component values", bool(np.allclose(df.loc[~has, B["emb20"]], med_e)) if (~has).any() else True)
ck.add("npz X_emb_full equals the M6 embedding (missing set to 0)", np.allclose(Z["X_emb_full"], np.nan_to_num(full, nan=0.0)))
'''),
md("""
### 5.6 Temporal block: dummies vs. q_ts, for random people
"""),
code(r'''
rows = []
for uid in RAND:
    r = DF.loc[uid]
    ones = [c for c in ym if r[c] == 1]
    rows.append({"mentee_uid": uid, "q_ts": r.q_ts, "month of q_ts": r.Q.strftime("%Y-%m"), "dummy equal to 1": ones[0] if ones else "(none: reference 2021-05)",
                 "editing_page": r.editing_page, "q_has_article_context": r.q_has_article_context,
                 "page edited": r.article_context if isinstance(r.article_context, str) else ""})
display(pd.DataFrame(rows))
'''),
md("""
### 5.7 Mentor block (M): recomputed from M2 questions and M1 mentor periods, for random people

For the owner of the talk page where the question was asked (`page_mentor_uid`), take all questions on that
mentor's page before `q_ts` (M2 `questions.parquet`). A question counts as answered only if the mentor's first
reply came before `q_ts`. Days active = `q_ts` minus the mentor's first roster start (M1 `mentor_periods.parquet`).
"""),
code(r'''
MQ = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["question_revid", "page_mentor_uid", "q_ts", "mentor_first_reply_ts"])
MQ = MQ.dropna(subset=["page_mentor_uid"]).assign(t=lambda d: pd.to_datetime(d.q_ts, utc=True), rt=lambda d: pd.to_datetime(d.mentor_first_reply_ts, utc=True))
PER = pd.read_parquet(config.INTERIM / "m1" / "mentor_periods.parquet")
def my_M(uid):
    r = DF.loc[uid]; Q = r.Q
    g = MQ[MQ.page_mentor_uid == r.page_mentor_uid]
    prior = g[g.t < Q]
    ans = prior.rt.notna() & (prior.rt < Q)
    rec = prior[prior.t >= Q - pd.Timedelta(days=30)]
    ans30 = rec.rt.notna() & (rec.rt < Q)
    lag = ((prior.rt - prior.t).dt.total_seconds() / 3600)[ans]
    starts = pd.to_datetime(PER[PER.mentor_uid == r.page_mentor_uid].start, utc=True)
    out = {"m_prior_mentee_count": len(prior), "m_prior_reply_rate": ans.mean() if len(prior) else 0.0,
           "m_prior_avg_response_hours": lag.mean() if len(lag) else 0.0, "m_recent30d_mentee_count": len(rec),
           "m_recent30d_reply_rate": ans30.mean() if len(rec) else 0.0,
           "m_days_active": (Q - starts.min()).total_seconds() / 86400 if len(starts) else 0.0}
    return out, prior
okM = []
for uid in RAND:
    r = DF.loc[uid]
    hand, prior = my_M(uid)
    print("=" * 110)
    print(f"mentee_uid {uid} | page_mentor_uid {r.page_mentor_uid} | q_ts {r.q_ts} | earlier questions on this page: {len(prior)}")
    if len(prior):
        display(prior.tail(5).assign(answered_before_Q=lambda d: d.rt.notna() & (d.rt < r.Q))[["question_revid", "q_ts", "mentor_first_reply_ts", "answered_before_Q"]])
    okM.append(compare(hand, {c: r[c] for c in B["M"]}, title="M block: recomputed vs. dataset"))
ck.add("random people: all 6 mentor-block variables match", all(okM))
'''),
md("""
### 5.8 npz vs. parquet for random rows

M8 reads the npz, not the parquet. For random rows we look up the same person in both files and compare a few
values from every block, the treatment and the main outcome.
"""),
code(r'''
okZ = []
for uid in RAND:
    i = int(np.where(Z["mentee_ids"] == str(uid))[0][0])
    r = DF.loc[uid]
    hand = {"first_question_revid": int(Z["cid_order"][i]), "y_treat": Z["y_treat"][i], "oc_primary": Z["oc_primary"][i],
            "n_edits": Z["X_E"][i, list(Z["E_cols"]).index("n_edits")], "q_body_word_count": Z["X_Qtext"][i, list(Z["Qtext_cols"]).index("q_body_word_count")],
            "q_persp_toxicity": Z["X_Qpersp"][i, 0], "q_substantive": Z["X_Qtype"][i, 0], "emb_pc1": Z["X_emb20"][i, 0],
            "q_has_article_context": Z["X_temporal"][i, list(Z["temporal_cols"]).index("q_has_article_context")],
            "m_prior_reply_rate": Z["X_M"][i, list(Z["M_cols"]).index("m_prior_reply_rate")], "q_year_month": Z["q_year_month"][i]}
    data = {k: r[k] for k in hand if k not in ("y_treat", "q_year_month")}
    data.update({"first_question_revid": r.first_question_revid, "y_treat": r.treat, "q_year_month": r.Q.strftime("%Y-%m")})
    okZ.append(compare(hand, data, title=f"mentee_uid {uid}, npz row {i}: npz vs. parquet"))
ck.add("random rows: npz values equal the parquet", all(okZ))
'''),
md("""
## 6. Edge-case lists

Cases where a rule makes a choice. Open a few with the links to see that the choice is right.
"""),
code(r'''
ed_all = mark(ED)
print("6.1 people with no edit before the question (all E counts are 0, hours_since_last_edit = -1):", int((df.n_edits == 0).sum()))
for r in df[df.n_edits == 0].head(3).itertuples(): print("   ", r.mentee_uid, contribs(r.mentee_name))
print("\n6.2 people with unknown registration time (account_age_hours = -1):", int((df.account_age_hours == -1).sum()))
for r in df[df.account_age_hours == -1].head(3).itertuples(): print("   ", r.mentee_uid, contribs(r.mentee_name))
tagged = ed_all.tags.map(lambda t: "mw-reverted" in list(t)) & ed_all.used
print("\n6.3 edits before Q tagged mw-reverted:", int(tagged.sum()))
print("    revert time unknown, counted as reverted before Q:", int((tagged & ed_all.revert_ts.isna()).sum()))
late = ed_all[tagged & (ed_all.revert_ts >= ed_all.Q)]
print("    reverted only AFTER Q, so NOT counted (R7.8):", len(late))
for r in late.head(3).itertuples(): print(f"     uid {r.uid} edit {r.revid} at {r.timestamp}, Q {r.Q}, reverted {r.revert_ts}  {diff(r.revid)}")
atq = ed_all[ed_all.ts == ed_all.Q]
print("\n6.4 edits at exactly q_ts (left out; normally the question itself):", len(atq), "| people:", atq.uid.nunique(),
      "| share of these on a user talk page (ns 3):", round(float((atq.ns == 3).mean()), 4))
print("\n6.5 largest values over all rows (check that they are plausible; experienced editors appear only in the sample without the D10 restriction):")
for c in ["n_edits", "max_sizediff", "n_abuse", "log_create", "account_age_hours"]:
    r = df.loc[df[c].idxmax()]
    print(f"    max {c:18s} = {r[c]:>12.1f}  mentee_uid {r.mentee_uid}  in main sample: {r.in_sample}  {contribs(r.mentee_name)}")
print("\n6.6 people whose page_mentor_uid has no earlier question (M block counts are 0):", int((df.m_prior_mentee_count == 0).sum()))
print("\n6.7 missing rate before imputation, by block:")
display(DD.groupby("block").agg(columns=("variable", "size"), max_missing_rate=("missing_rate", "max")))
print("6.8 constant columns in the main sample:", const)
'''),
md("""
## 7. Distributions

### 7.1 Treated vs. control (main sample): covariates with the largest standardized difference before adjustment

These are the differences the propensity score has to remove. M8 shows the balance after stratification.
"""),
code(r'''
d = df[df.in_sample]
g = d.groupby("treat")[COV].mean().T
g.columns = ["control_mean", "treated_mean"]
g["std_diff"] = (g.treated_mean - g.control_mean) / d[COV].std().replace(0, np.nan)
display(g.reindex(g.std_diff.abs().sort_values(ascending=False).index).head(20).round(4))
'''),
md("### 7.2 E block: summary over the main sample"),
code(r'''
display(d[E_COLS].describe(percentiles=[.5, .9, .99]).T.round(2))
'''),
md("### 7.3 Questions per year (main sample) and share treated"),
code(r'''
display(d.assign(year=d.Q.dt.year).groupby("year").agg(n=("treat", "size"), treated_share=("treat", "mean")).round(3))
'''),
md("""
## 8. Summary (written to results/m7_report.json)
"""),
code(r'''
ck.report()
rep = {"rows": len(df), "main": int(df.in_sample.sum()), "unrestricted": int(df.in_sample_unrestricted.sum()),
       "n_covariates": len(COV), "constant_in_main": const,
       "checks_passed": sum(ok for _, ok in ck.items), "checks_total": len(ck.items),
       "failed_checks": [n for n, ok in ck.items if not ok], "seed": SEED, "random_people": [int(u) for u in RAND]}
(config.RESULTS / "m7_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
'''),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "m7_check.ipynb"
    write_notebook(CELLS, out)
    print("wrote", out)
    if "--run" in sys.argv:
        run_as_script(out, cwd=out.parent.parent)
        print("\nnotebook ran end to end")


if __name__ == "__main__":
    main()
