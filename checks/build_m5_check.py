"""
build_m5_check.py: writes checks/m5_check.ipynb (checks of the M5 sample, treatment variables, anchors, and outcomes)

The notebook follows the fixed 8-section layout of PLAN.md section 12:
    1 module description  2 files, manifest, data dictionary  3 invariants (Checks)
    4 comparison with the paper version (sample overlap, treatment agreement, I-34, I-50)
    5 tracer samples and random samples: raw thread turns and raw edits next to the processed values, with every
      oc_* outcome recomputed by independent code; sample funnel recomputed from the flags; language decisions
      recomputed with fastText
    6 edge-case lists  7 distributions  8 summary (results/m5_report.json)

Run:  python checks/build_m5_check.py          # write the notebook
      python checks/build_m5_check.py --run    # write it and run it from top to bottom
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks._nbgen import code, md, run_as_script, write_notebook  # noqa: E402

CELLS = [
md("""
# M5 analysis sample, treatment variables, anchors, and outcomes: checks

## 1. Module description

**What M5 does.** `pipeline/m5_sample/build_sample.py` and `outcomes.py` turn the M2 questions, the M3 edit
histories and the M4 eligibility flags into one row per asker (`data/interim/m5/sample.parquet`):

1. **First question.** Each asker's earliest tagged question (chosen by M4, rule R6.4).
2. **Sample funnel.** Askers are removed step by step (M4 reasons, thread not found, reply earlier than the question,
   question removed in the next edit, question text hidden, empty text, not English). The counts go to
   `results/tables/sample_funnel.csv`.
3. **Language.** fastText lid.176 plus a few simple rules (decision D5).
4. **Treatment.** `treat` = anyone other than the newcomer replied before the cutoff; `treat_mentor` = the assigned
   mentor replied; `treat_48h` = a reply within 48 hours (rule R5.3).
5. **Anchor (clock start).** Treated: time of the first reply. Control: question time plus a *pseudo lag* drawn at
   random from the reply lags of the treated group.
6. **Outcomes `oc_*`.** Edits strictly after the anchor, in windows of 7 to 180 days measured in hours (R7.2),
   reverted edits counted only if the revert happened before the cutoff (R7.8), and follow-up flags
   `oc_eligible_{w}d` (R7.4). Robustness versions of every outcome use other pseudo lags for the control group
   (`__median`, `__p25`, `__p75`, `__zero`, `__multi`).

**Paper numbers that depend on M5.** The sample construction table (Appendix Table A), the treatment and outcome
definitions (Section 3.2), the pseudo anchor for controls (Section 3.3), and every effect estimate on Y1 to Y6 and
the secondary outcomes (all of them use these columns through M7 and M8).

**How to use this notebook.** Run it from top to bottom. Each check prints `[OK ]` or `[FAIL]`. Each comparison of
a hand recomputation with the stored data prints a table with a `match` column and ends with
`all fields match` or `MISMATCH in: [...]`. Random rows are drawn with `SEED` in the next cell: change `SEED` to any
other integer and run again to look at fresh rows. Links open the raw record on Wikipedia.
"""),
code(r'''
import json, sys, re
from pathlib import Path
from datetime import timedelta
import numpy as np, pandas as pd
import pyarrow.dataset as ds
ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.common.io import iter_jsonl
from pipeline.common.tracer import tracer_frame
from checks._show import Checks, compare, pick, oldid, diff, contribs, logid

SEED = 0        # change SEED to any other integer to draw a fresh random sample in every section below
N_RANDOM = 5    # number of random rows shown in each random check
N_BULK = 1000   # number of random people for the bulk recomputation of all outcomes (section 5c)

pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 230); pd.set_option("display.max_rows", 200)
CUTOFF = pd.Timestamp(config.DATA_CUTOFF)
M5 = config.INTERIM / "m5"
s_all = pd.read_parquet(M5 / "sample.parquet")                       # every asker
s = s_all[s_all.in_sample]                                           # main analysis sample
s_any = s_all[s_all.in_sample | s_all.in_sample_unrestricted]        # people who get anchors and outcomes
fun = pd.read_csv(config.RESULTS / "tables" / "sample_funnel.csv")
man = json.loads((M5 / "manifest.json").read_text())
info = json.loads((M5 / "lag_info.json").read_text())
ts = lambda x: pd.to_datetime(x, utc=True, format="ISO8601")         # noqa: E731
print(f"askers {len(s_all)} | main sample {len(s)} | unrestricted sample {int(s_all.in_sample_unrestricted.sum())}"
      f" | rows with outcomes {len(s_any)}")

# raw thread turns from M2 (DiscussionTools output), for every asker's first question
need = set(s_all.first_question_revid)
TURNS = {r["question_revid"]: r["turns"] for r in iter_jsonl(config.INTERIM / "m2" / "turns.jsonl") if r["question_revid"] in need}
WIKITEXT = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["question_revid", "question_wikitext"]
                           ).set_index("question_revid").question_wikitext

# raw edits from M3 for everyone who has outcomes
E = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
    columns=["uid", "revid", "timestamp", "ns", "tags"], filter=ds.field("uid").isin(list(s_any.mentee_uid))).to_pandas()
E["ts"] = ts(E.timestamp)
E["has_reverted_tag"] = E.tags.map(lambda t: "mw-reverted" in list(t if t is not None else []))

# revert times (R7.8). The revert-time crawler is still appending to revert_times.jsonl, so we use only the rows
# that were already fetched when M5 started; this is the file as M5 saw it.
m5_start = pd.Timestamp(man["finished_at"]) - pd.Timedelta(seconds=man["elapsed_sec"])
REV, n_late = {}, 0
for name in ("revert_times_sample.jsonl", "revert_times.jsonl"):
    p = config.RAW / "m3" / name
    for r in (iter_jsonl(p) if p.exists() else []):
        if not r.get("revert_ts"):
            continue
        if pd.Timestamp(r["fetched_at"]) > m5_start:
            n_late += 1; continue
        REV[int(r["revid"])] = pd.Timestamp(r["revert_ts"])
E["revert_ts"] = E.revid.map(REV)
# reverted before the cutoff: has the mw-reverted tag, and if we know when it was reverted, that time is before the cutoff
E["reverted"] = E.has_reverted_tag & (E.revert_ts.isna() | (E.revert_ts < CUTOFF))
BY_UID = {u: g.sort_values("ts") for u, g in E.groupby("uid")}
EMPTY = E.iloc[0:0]
print(f"edits loaded {len(E)} | revert times known {len(REV)} (ignored {n_late} rows fetched after M5 started at {m5_start})")
'''),
md("""
### Helper: my own recomputation of every outcome

`hand_outcomes(edits, anchor)` is written here from the definitions in the paper and in `outcomes.py` section 1,
without calling the pipeline. Notation: a window (lo, hi] means edit time > anchor + lo days and <= anchor + hi
days; "main" = namespace 0 (articles); "even" = even namespaces (everything except talk pages); "days" = number
of distinct UTC dates; "kept" = not reverted before the cutoff. The anchor is cut to whole seconds, as in the
pipeline (edit times are whole seconds, so this changes nothing except exact ties).
"""),
code(r'''
WINDOWS = (7, 14, 21, 28, 30, 60, 180)

def hand_outcomes(ed, anchor):
    a = pd.Timestamp(anchor).floor("s")
    def win(lo, hi, even=False):
        w = ed[(ed.ts > a + pd.Timedelta(days=lo)) & (ed.ts <= a + pd.Timedelta(days=hi))]
        return w[w.ns % 2 == 0] if even else w[w.ns == 0]
    days = lambda w: int(w.ts.dt.date.nunique())                       # noqa: E731
    kept = lambda w: w[~w.reverted]                                    # noqa: E731
    m14, e14, e30 = win(0, 14), win(0, 14, True), win(0, 30, True)
    o = {"primary": int(len(m14) >= 1),                                # Y1
         "n_mainspace_edits_14d": len(m14),                            # Y2
         "cross_day_any_14d": int(days(m14) >= 2),                     # Y3
         "unique_ns": int(e14.ns.nunique()),                           # Y4
         "active_days_30d": days(e30),                                 # Y5
         "reverted_any": int(e14.reverted.any()),                      # Y6
         "primary_constructive": int(len(kept(m14)) >= 1),
         "cross_day_constructive_14d": int(days(kept(m14)) >= 2),
         "sec2": int(len(win(14, 28)) >= 1),
         "constructive_edit_15_60d": int(len(kept(win(14, 60))) >= 1),
         "active_days_14d": days(e14),
         "constructive_days_30d": days(kept(e30)),
         "mainspace_15_30d": int(len(win(14, 30)) >= 1),
         "n_mainspace_15_30d": len(win(14, 30)),
         "active_days_15_30d": days(win(14, 30, True))}
    for w in WINDOWS:
        o[f"mainspace_{w}d"] = int(len(win(0, w)) >= 1)
    for name, lo, hi in (("15_28d", 14, 28), ("29_60d", 28, 60), ("61_180d", 60, 180)):
        n = len(win(lo, hi))
        o[f"th_1plus_{name}"], o[f"th_5plus_{name}"] = int(n >= 1), int(n >= 5)
    for w in WINDOWS:
        o[f"eligible_{w}d"] = int(a + pd.Timedelta(days=w) <= CUTOFF)
    return o

MAIN_KEYS = list(hand_outcomes(EMPTY, CUTOFF))                     # 35 main outcomes
VAR_KEYS = [k for k in MAIN_KEYS if not k.startswith("eligible_")]  # the 28 outcomes that have pseudo-lag versions
VARIANTS = ("median", "p25", "p75", "zero", "multi")

def first_reply_from_turns(turns, q_ts):
    """My reading of R5.3: the first turn by anyone who is not the newcomer, at or after the question minute, before the cutoff."""
    q_min = ts(q_ts).floor("min")
    for i, t in enumerate(turns):
        if t["role"] != "mentee" and t.get("ts") and q_min <= ts(t["ts"]) < CUTOFF:
            return i
    return None

def turns_table(turns, first_idx):
    return pd.DataFrame([{"#": i, "author": t.get("author"), "role": t["role"], "ts": t.get("ts"), "level": t.get("level"),
                          "text": str(t.get("text") or "")[:80].replace("\n", " "),
                          "note": "<- FIRST REPLY" if i == first_idx else ("question" if i == 0 and t["role"] == "mentee" else "")}
                         for i, t in enumerate(turns)])
print("main outcomes:", len(MAIN_KEYS), "| outcomes with pseudo-lag versions:", len(VAR_KEYS))
'''),
md("""
## 2. Files, manifest, and data dictionary

Files written by M5, the manifest (when it ran, against which cutoff, with which seed), and the sample funnel
(paper Appendix Table A).
"""),
code(r'''
for p in (M5 / "sample.parquet", M5 / "lag_info.json", M5 / "manifest.json", config.RESULTS / "tables" / "sample_funnel.csv"):
    print(f"  {p.relative_to(config.ROOT)}  {p.stat().st_size:>12,} bytes  modified {pd.Timestamp(p.stat().st_mtime, unit='s', tz='UTC'):%Y-%m-%d %H:%M} UTC")
print("\nfinished:", man["finished_at"], "| cutoff:", man["data_cutoff"], "| params:", man["params"])
print(json.dumps(man["counts"], indent=1))
display(fun)
'''),
md("""
### Data dictionary: `sample.parquet`

One row per asker. The `checked in` column names the cell that verifies the column (3 = invariants,
4 = paper comparison, 5a/5b/5c/5d/5e = tracer and random samples, 6 = edge cases). The last cell of this
section checks that every column of the file is listed here.
"""),
code(r'''
D = {}  # column -> (meaning, checked in)
def add(cols, meaning, where):
    for c in cols.split():
        D[c] = (meaning, where)
add("mentee_uid", "user id of the asker (primary key)", "3, 5b")
add("first_question_revid", "revision id of the asker's first tagged question (chosen by M4)", "3, 5b")
add("question_revid", "the same revision id as it appears in the M2 question table", "3")
add("first_q_ts", "time of the first question, as M4 recorded it (UTC)", "3")
add("q_ts", "time of the first question, as M2 recorded it (UTC); used for lags and anchors", "3, 5b")
add("registered_at", "registration time of the account (M3 profile)", "5b, 5d")
add("creation_action", "how the account was created according to the newusers log (create, autocreate, byemail, ...)", "5b, 5d")
add("eligible", "M4: True when no exclusion reason applies", "3, 5b")
add("exclude_reason", "M4: the first exclusion reason that applies (empty when eligible)", "5d")
add("exclude_reasons", "M4: every exclusion reason that applies", "5d")
add("eligible_unrestricted", "M4: eligible when the D10 limits (account type, registration date) are ignored", "5b")
add("claimed_after_question", "M4: a mentor claimed this newcomer by hand only after the first question", "5b")
add("n_claims", "M4: number of claim log entries for this newcomer before the cutoff", "5b, 5d")
add("n_manual_claims", "M4: number of those claims that were manual (not bulk transfers)", "5b, 5d")
add("primary_mentor_uid", "M4: the newcomer's primary mentor at registration", "5b")
add("q_primary_mentor_uid", "M2: the primary mentor named in the question (away note or page owner)", "5b")
add("page_mentor_uid", "M2: owner of the talk page where the question was posted", "5b")
add("mentor_on_roster_at_registration", "M4 diagnostic: the primary mentor was on the mentor list at registration", "5b")
add("mentor_in_auto_pool_at_registration", "M4 diagnostic: the primary mentor was in the automatic assignment pool at registration", "5b")
add("assigned_later", "M4 diagnostic: the mentor seems to have been assigned after registration", "5b")
add("possible_forcementor", "M4 diagnostic: possible use of the force-mentor setting", "5b")
add("tag", "change tag of the question revision (mentorship module question / panel question)", "5b")
add("heading", "section heading of the question", "5b")
add("page_title_at_q", "talk page the question was posted on", "5b, 5d")
add("away_routed", "M2: the primary mentor was away, so the question went to a backup mentor", "5b")
add("editing_page", "M2: the newcomer asked from the editor of this page (panel question)", "5b")
add("article_context", "M2: an article was mentioned as the context of the question", "5b")
add("question_text", "plain text of the question body after M2 cleaning", "5b, 5d, 5e")
add("has_any_reply", "M2: someone other than the newcomer replied before the cutoff", "3")
add("first_reply_ts", "M2: time of the first reply (signature time, whole minutes)", "3, 5b")
add("first_replier_role", "M2: role of the first replier (page_mentor, other_mentor, third_party, unknown)", "5b, 6")
add("has_mentor_reply", "M2: the talk page owner or the primary mentor replied", "3")
add("mentor_first_reply_ts", "M2: time of the first reply by the talk page owner or primary mentor", "3")
add("thread_origin", "M2: where the thread came from (cutoff page, history fallback, ...)", "4, 5d")
add("parse_flags", "M2: problems found while locating the thread (thread_not_found, reply_before_question, ...)", "5d")
add("removal_kind", "M2: how the question section was removed, when it was (revert, self_removed, ...)", "5d")
add("mentee_name", "current user name of the asker", "5b")
add("late_fetch", "M3: some edits of this person were fetched after the cutoff date (top-up)", "5b")
add("lang", "language label of the question (fastText lid.176 plus rules, D5)", "3, 5e")
add("lang_prob", "fastText probability of the chosen label (empty for the short-text rule)", "3, 5e")
add("lang_method", "which rule decided the language", "3, 5e")
add("is_english", "lang == 'en'", "3, 5e")
add("is_english_ascii_rule", "comparison only: the paper-version rule (at least 80% ASCII letters)", "5e, 7")
add("lang_langdetect", "comparison only: the label from langdetect", "5e, 7")
add("treat", "1 = someone other than the newcomer replied before the cutoff (main treatment)", "3, 5b")
add("treat_mentor", "1 = the talk page owner or the primary mentor replied", "3, 5b")
add("reply_lag_days", "first reply time minus question time, in days (treated only)", "3, 5b")
add("treat_48h", "1 = treated and reply_lag_days <= 2", "3, 5b")
add("long_reply_lag", "reply_lag_days >= 365 (flag only)", "3, 6")
add("exclude_step", "first funnel step that removed this asker (empty = in the main sample)", "3, 5d")
add("in_sample", "main analysis sample (exclude_step is empty)", "3, 5d")
add("in_sample_unrestricted", "sample without the D10 limits (robustness check)", "3, 5d")
add("anchor", "clock start: first reply time (treated) or question time + pseudo lag (control)", "3, 5b")
add("pseudo_lag_days", "pseudo lag drawn for controls (days); empty for treated", "3, 5b")
MEAN = {"primary": "Y1: at least one article (ns 0) edit in (anchor, anchor+14d]",
        "n_mainspace_edits_14d": "Y2: number of article edits in (0, 14d]",
        "cross_day_any_14d": "Y3: article edits on at least 2 distinct UTC dates in (0, 14d]",
        "unique_ns": "Y4: number of distinct even namespaces edited in (0, 14d]",
        "active_days_30d": "Y5: distinct UTC dates with an even-namespace edit in (0, 30d]",
        "reverted_any": "Y6: some even-namespace edit in (0, 14d] was reverted before the cutoff",
        "primary_constructive": "at least one article edit in (0, 14d] that was not reverted",
        "cross_day_constructive_14d": "not-reverted article edits on at least 2 dates in (0, 14d]",
        "sec2": "at least one article edit in (14d, 28d]",
        "constructive_edit_15_60d": "at least one not-reverted article edit in (14d, 60d]",
        "active_days_14d": "distinct dates with an even-namespace edit in (0, 14d]",
        "constructive_days_30d": "distinct dates with a not-reverted even-namespace edit in (0, 30d]",
        "mainspace_15_30d": "at least one article edit in (14d, 30d]",
        "n_mainspace_15_30d": "number of article edits in (14d, 30d]",
        "active_days_15_30d": "distinct dates with an even-namespace edit in (14d, 30d]"}
for w in WINDOWS:
    MEAN[f"mainspace_{w}d"] = f"at least one article edit in (0, {w}d] (window sensitivity)"
    MEAN[f"eligible_{w}d"] = f"follow-up complete: anchor + {w} days <= cutoff (R7.4)"
for name, lo, hi in (("15_28d", 14, 28), ("29_60d", 28, 60), ("61_180d", 60, 180)):
    for k in (1, 5):
        MEAN[f"th_{k}plus_{name}"] = f"at least {k} article edits in ({lo}d, {hi}d]"
for k in MAIN_KEYS:
    D[f"oc_{k}"] = (MEAN[k], "3, 5a, 5b, 5c")
VMEAN = {"median": "control anchor = q_ts + median treated lag", "p25": "control anchor = q_ts + 25th percentile lag",
         "p75": "control anchor = q_ts + 75th percentile lag", "zero": "control anchor = q_ts (zero lag)",
         "multi": "control: mean over 20 random pseudo lags (RandomState(123))"}
for k in VAR_KEYS:
    for v in VARIANTS:
        D[f"oc_{k}__{v}"] = (f"same as oc_{k}, but {VMEAN[v]}; for treated people equal to oc_{k}", "3, 5b, 5c")
dd = pd.DataFrame([{"column": c, "meaning": m, "checked in": w} for c, (m, w) in D.items() if "__" not in c])
display(dd)
print(f"\nplus {sum('__' in c for c in D)} robustness columns oc_<outcome>__<version> for the {len(VAR_KEYS)} outcomes and versions {VARIANTS}:")
for v in VARIANTS: print(f"   __{v}: {VMEAN[v]}")
print("\nOther files: sample_funnel.csv (step, removed, remaining; checked in 5d); lag_info.json (n_lags, median, p25, p75, mean, zero, n_multi; checked in 3)")
'''),
code(r'''
missing = [c for c in s_all.columns if c not in D]
extra = [c for c in D if c not in s_all.columns]
print("columns in the file but not in the dictionary:", missing)
print("columns in the dictionary but not in the file:", extra)
'''),
md("""
## 3. Invariants

Each line is a rule that must hold for every row. Several of them recompute a column from raw data for **all**
rows (treatment from the raw thread turns, pseudo lags by replaying the random draws).
"""),
code(r'''
ck = Checks()
q = ts(s_any.q_ts); a = ts(s_any.anchor); r_ts = ts(s_any.first_reply_ts)
T, C = s_any.treat == 1, s_any.treat == 0
print("Keys and sample flags")
ck.add("one row per asker (mentee_uid unique)", s_all.mentee_uid.is_unique)
ck.add("one first question per asker (first_question_revid unique)", s_all.first_question_revid.is_unique)
ck.add("question_revid == first_question_revid", (s_all.question_revid == s_all.first_question_revid).all())
ck.add("first_q_ts (M4) == q_ts (M2)", (s_all.first_q_ts == s_all.q_ts).all())
ck.add("in_sample <=> exclude_step is empty", (s_all.in_sample == s_all.exclude_step.isna()).all())
ck.add("main sample: all eligible (M4) and English", bool(s.eligible.all() and s.is_english.all()))
ck.add("main sample is inside the unrestricted sample", s.in_sample_unrestricted.all())
ck.add("eligible <=> exclude_reason is empty", (s_all.eligible == s_all.exclude_reason.isna()).all())
ck.add("registered before the first question (where the registration time is known)",
       bool((ts(s_all.registered_at.dropna()) <= ts(s_all.loc[s_all.registered_at.notna(), "q_ts"])).all()))
ck.add("n_manual_claims <= n_claims", (s_all.n_manual_claims.fillna(0) <= s_all.n_claims.fillna(0)).all())
ck.add("tag is one of the two mentorship tags", set(s_all.tag.dropna()) <= {"mentorship module question", "mentorship panel question"})
ck.add("mentor_first_reply_ts is never earlier than first_reply_ts",
       bool((ts(s_all.mentor_first_reply_ts.dropna()) >= ts(s_all.loc[s_all.mentor_first_reply_ts.notna(), "first_reply_ts"])).all()))
ck.add("is_english <=> lang == 'en'", (s_all.is_english == s_all.lang.eq("en")).all())
ck.add("lang_method takes only the 5 known values",
       set(s_all.lang_method) <= {"fasttext_lid176", "fasttext_uncertain_en", "fasttext_uncertain", "ascii_fallback_short", "empty"})
ck.add("method fasttext_lid176 has lang_prob >= 0.5; uncertain ones < 0.5",
       bool((s_all[s_all.lang_method == "fasttext_lid176"].lang_prob >= 0.5).all() and (s_all[s_all.lang_method.isin(["fasttext_uncertain"])].lang_prob < 0.5).all()))

print("\nTreatment, recomputed from the raw thread turns for every row with outcomes")
fr = {rv: first_reply_from_turns(TURNS.get(rv, []), qt) for rv, qt in zip(s_any.first_question_revid, s_any.q_ts)}
h_treat = s_any.first_question_revid.map(lambda rv: int(fr[rv] is not None))
h_first = s_any.first_question_revid.map(lambda rv: TURNS[rv][fr[rv]]["ts"] if fr[rv] is not None else None)
def mentor_replied(rv, qt):
    q_min = ts(qt).floor("min")
    return int(any(t["role"] in ("page_mentor", "primary_mentor") and t.get("ts") and q_min <= ts(t["ts"]) < CUTOFF for t in TURNS.get(rv, [])))
h_mentor = pd.Series([mentor_replied(rv, qt) for rv, qt in zip(s_any.first_question_revid, s_any.q_ts)], index=s_any.index)
ck.add(f"treat == recomputed from turns ({int((h_treat != s_any.treat).sum())} differ)", (h_treat == s_any.treat).all())
ck.add("first_reply_ts == ts of the first non-newcomer turn", (h_first.fillna("-") == s_any.first_reply_ts.fillna("-")).all())
ck.add(f"treat_mentor == recomputed from turns ({int((h_mentor != s_any.treat_mentor).sum())} differ)", (h_mentor == s_any.treat_mentor).all())
ck.add("treat == has_any_reply; treat_mentor == has_mentor_reply",
       bool((s_any.treat == s_any.has_any_reply.fillna(False).astype(int)).all() and (s_any.treat_mentor == s_any.has_mentor_reply.fillna(False).astype(int)).all()))
lag = (r_ts - q).dt.total_seconds() / 86400
ck.add("reply_lag_days == (first_reply_ts - q_ts) in days", np.allclose(lag[T], s_any.reply_lag_days[T], atol=1e-9))
ck.add("reply never earlier than the question minute", bool((r_ts[T] >= q[T].dt.floor("min")).all()))
ck.add("treat_48h == treat and reply_lag_days <= 2", (s_any.treat_48h == (T & (s_any.reply_lag_days <= 2)).astype(int)).all())
ck.add("treat_mentor <= treat and treat_48h <= treat", bool((s_any.treat_mentor <= s_any.treat).all() and (s_any.treat_48h <= s_any.treat).all()))
ck.add("long_reply_lag == reply_lag_days >= 365", (s_any.long_reply_lag == (s_any.reply_lag_days >= 365)).all())

print("\nAnchors and pseudo lags")
ck.add("treated: anchor == first reply time", bool((a[T] == r_ts[T]).all()))
ck.add("treated: no pseudo lag", s_any[T].pseudo_lag_days.isna().all())
ck.add("control: has a pseudo lag in [0, 365)", bool(s_any[C].pseudo_lag_days.between(0, 365, inclusive="left").all()))
ck.add("control: anchor == q_ts + pseudo lag (within 1 s)",
       bool(((a[C] - (q[C] + pd.to_timedelta(s_any[C].pseudo_lag_days, unit="D"))).abs() <= pd.Timedelta(seconds=1)).all()))
ck.add("anchor never earlier than the question minute", bool((a >= q.dt.floor("min")).all()))
# replay the draws: lags of the main-sample treated with 0 <= lag < 365, rows sorted by (q_ts, revid), one draw per control
order = s_any.sort_values(["q_ts", "first_question_revid"])
main_sorted = order[order.in_sample]
LAGS = main_sorted.loc[(main_sorted.treat == 1) & (main_sorted.reply_lag_days >= 0) & (main_sorted.reply_lag_days < 365), "reply_lag_days"].to_numpy()
rng = np.random.RandomState(config.SEED)
replay = {rv: float(rng.choice(LAGS)) for rv, t in zip(order.first_question_revid, order.treat) if t == 0}
ck.add(f"control pseudo lags == replayed draws from RandomState({config.SEED})",
       bool(np.allclose(s_any[C].first_question_revid.map(replay), s_any[C].pseudo_lag_days, atol=0)))
rng = np.random.RandomState(123)
MULTI = {rv: [float(rng.choice(LAGS)) for _ in range(20)] for rv, t in zip(order.first_question_revid, order.treat) if t == 0}
hand_info = {"n_lags": len(LAGS), "median": float(np.median(LAGS)), "p25": float(np.percentile(LAGS, 25)),
             "p75": float(np.percentile(LAGS, 75)), "mean": float(LAGS.mean()), "zero": 0.0, "n_multi": 20}
ck.add("lag_info.json == recomputed from the treated lags", all(np.isclose(hand_info[k], info[k]) for k in hand_info))

print("\nOutcomes")
oc_cols = [c for c in s_all.columns if c.startswith("oc_")]
ck.add("no missing outcome for people with anchors", s_any[oc_cols].notna().all().all())
ck.add("outcomes are empty for people without anchors", s_all[~(s_all.in_sample | s_all.in_sample_unrestricted)][oc_cols].isna().all().all())
o = lambda k: s_any[f"oc_{k}"]                                         # noqa: E731
binary = [k for k in MAIN_KEYS if not (k.startswith("n_") or "days" in k or k == "unique_ns")]
ck.add("binary outcomes take only 0 and 1", bool(s_any[[f"oc_{k}" for k in binary]].isin([0, 1]).all().all()))
ck.add("counts are whole numbers >= 0", bool(all((o(k) >= 0).all() and (o(k) % 1 == 0).all() for k in MAIN_KEYS if k not in binary)))
ck.add("primary == mainspace_14d", (o("primary") == o("mainspace_14d")).all())
ck.add("primary == (n_mainspace_edits_14d >= 1)", (o("primary") == (o("n_mainspace_edits_14d") >= 1)).all())
ck.add("mainspace_7d <= 14d <= 21d <= 28d <= 30d <= 60d <= 180d",
       all((o(f"mainspace_{a_}d") <= o(f"mainspace_{b_}d")).all() for a_, b_ in zip(WINDOWS, WINDOWS[1:])))
ck.add("primary_constructive <= primary; cross_day_constructive <= cross_day_any <= primary",
       bool((o("primary_constructive") <= o("primary")).all() and (o("cross_day_constructive_14d") <= o("cross_day_any_14d")).all() and (o("cross_day_any_14d") <= o("primary")).all()))
ck.add("sec2 == th_1plus_15_28d", (o("sec2") == o("th_1plus_15_28d")).all())
ck.add("th_5plus <= th_1plus in every window", all((o(f"th_5plus_{w}") <= o(f"th_1plus_{w}")).all() for w in ("15_28d", "29_60d", "61_180d")))
ck.add("mainspace_15_30d == (n_mainspace_15_30d >= 1)", (o("mainspace_15_30d") == (o("n_mainspace_15_30d") >= 1)).all())
ck.add("active_days_14d <= active_days_30d; constructive_days_30d <= active_days_30d; active_days_15_30d <= active_days_30d",
       bool((o("active_days_14d") <= o("active_days_30d")).all() and (o("constructive_days_30d") <= o("active_days_30d")).all() and (o("active_days_15_30d") <= o("active_days_30d")).all()))
ck.add("active_days_14d <= 15 and active_days_30d <= 31 (a 14-day window touches at most 15 dates)",
       bool((o("active_days_14d") <= 15).all() and (o("active_days_30d") <= 31).all()))
ck.add("unique_ns >= 1 <=> active_days_14d >= 1", ((o("unique_ns") >= 1) == (o("active_days_14d") >= 1)).all())
ck.add("reverted_any = 1 only if there was an edit in (0, 14d]", (o("reverted_any") <= (o("active_days_14d") >= 1)).all())
ck.add("primary = 1 implies active_days_14d >= 1 (articles are an even namespace)", (o("primary") <= (o("active_days_14d") >= 1)).all())
elig = pd.DataFrame({w: (a.dt.floor("s") + pd.Timedelta(days=w) <= CUTOFF).astype(int) for w in WINDOWS})
ck.add("oc_eligible_{w}d == (anchor + w days <= cutoff) for every window", all((elig[w] == o(f"eligible_{w}d")).all() for w in WINDOWS))
ck.add("eligible flags shrink as the window grows", all((o(f"eligible_{a_}d") >= o(f"eligible_{b_}d")).all() for a_, b_ in zip(WINDOWS, WINDOWS[1:])))
ck.add("treated: every robustness version equals the main outcome",
       all((s_any.loc[T, f"oc_{k}__{v}"] == s_any.loc[T, f"oc_{k}"]).all() for k in VAR_KEYS for v in VARIANTS))
ck.add("control: __multi of a 0/1 outcome lies in [0, 1] and is a multiple of 1/20",
       all((s_any.loc[C, f"oc_{k}__multi"].between(0, 1) & np.isclose((s_any.loc[C, f"oc_{k}__multi"] * 20) % 1, 0)).all() for k in binary if not k.startswith("eligible")))
ck.report()
'''),
md("""
## 4. Comparison with the paper version (sample overlap and treatment agreement; I-34, I-50)

The paper sample (`psm_dataset.npz`) is matched to the new sample by the revid of the first question. Where the
treatment disagrees, the usual reason is that the paper version only read the main talk page on the day it was
crawled (replies in archived or deleted sections were lost), while the new version reads the talk page as it was at
the cutoff, its archives, and the revision history.
"""),
code(r'''
z = np.load(config.EXTERNAL["paper_npz"], allow_pickle=True)
s7 = {int(r["conversation_id"]): r["revid"] for r in iter_jsonl(config.EXTERNAL["paper_s7"])}
paper = pd.DataFrame({"revid": [int(s7[int(c)]) if s7.get(int(c)) else None for c in z["cid_order"]],
                      "paper_treat": z["y_treat"].astype(int), "paper_primary": z["oc_primary"].astype(int)}).dropna()
m = s.merge(paper, left_on="first_question_revid", right_on="revid", how="inner")
print(f"paper sample {len(z['cid_order'])}; new sample {len(s)}; overlap {len(m)}")
display(pd.crosstab(m.paper_treat.rename("paper treat"), m.treat.rename("new treat"), margins=True))
print("agreement on Y1 (oc_primary):", round(float((m.paper_primary == m.oc_primary).mean()), 4))
dis = m[m.paper_treat != m.treat]
print("\nwhere the treatment disagrees, source of the new thread:", dis.thread_origin.value_counts().to_dict())
display(pick(dis, 10, SEED)[["first_question_revid", "q_ts", "page_title_at_q", "paper_treat", "treat", "first_replier_role", "thread_origin"]])
'''),
md("""
## 5. Tracer samples and random samples: raw data next to processed values

### 5a. Tracer newcomers (fixed list in `pipeline/common/tracer.py`)

For each tracer newcomer: the anchor, the article edits in the 14 days after it, and every main outcome recomputed
by `hand_outcomes` and compared with the stored value.
"""),
code(r'''
tr = tracer_frame()
for uid in [int(u) for u in tr[tr.kind == "mentee"].uid.dropna()]:
    r = s_all[s_all.mentee_uid == uid]
    if r.empty or not (r.iloc[0].in_sample or r.iloc[0].in_sample_unrestricted):
        print(f"\n### uid {uid}: no outcomes ({'not an asker' if r.empty else r.iloc[0].exclude_step})"); continue
    r = r.iloc[0]
    ed = BY_UID.get(uid, EMPTY)
    print(f"\n### {r.mentee_name} (uid {uid})  treat={r.treat}  q_ts={r.q_ts}  anchor={r.anchor}  pseudo_lag={r.pseudo_lag_days}")
    print("   ", contribs(r.mentee_name), "| question:", oldid(r.first_question_revid))
    a0 = ts(r.anchor)
    w = ed[(ed.ts > a0) & (ed.ts <= a0 + pd.Timedelta(days=14))]
    display(w[["timestamp", "ns", "revid", "tags", "revert_ts", "reverted"]].head(10))
    compare(hand_outcomes(ed, r.anchor), {k: r[f"oc_{k}"] for k in MAIN_KEYS})
'''),
md("""
### 5b. Random treated and random control people: thread, anchor, edits, and every outcome

For each person:
1. the first question (text, time, link) and the raw thread turns from `data/interim/m2/turns.jsonl`, with the turn
   that counts as the first reply marked;
2. the anchor: the first reply time (treated) or question time + pseudo lag (control, with the replayed draw);
3. every edit in (anchor, anchor + 180 days] from `data/interim/m3/mentee_edits`, with namespace, tags, revert time
   (`data/raw/m3/revert_times*.jsonl`) and which windows it falls into;
4. all 35 main outcomes recomputed by `hand_outcomes` and compared with the data;
5. for controls, the robustness versions: `__median/__p25/__p75/__zero` recomputed at the fixed lags, and `__multi`
   recomputed as the mean over the 20 replayed draws.
"""),
code(r'''
def show_person(r):
    rv, uid = int(r.first_question_revid), int(r.mentee_uid)
    print("=" * 110)
    print(f"{r.mentee_name} (uid {uid})  treat={r.treat} treat_mentor={r.treat_mentor} treat_48h={r.treat_48h}  sample={'main' if r.in_sample else 'unrestricted only'}")
    print(f"  question revid {rv}  q_ts {r.q_ts}  on {r.page_title_at_q}  {oldid(rv)}")
    print(f"  contributions: {contribs(r.mentee_name)}")
    print(f"  question text: {str(r.question_text)[:300]!r}")
    print(f"  M2: tag={r.tag!r} heading={str(r.heading)[:60]!r} away_routed={r.away_routed} editing_page={r.editing_page} "
          f"article_context={str(r.article_context)[:40]!r} page_mentor_uid={r.page_mentor_uid} q_primary_mentor_uid={r.q_primary_mentor_uid}")
    print(f"  M3/M4: registered_at={r.registered_at} creation_action={r.creation_action} late_fetch={r.late_fetch} eligible={r.eligible} "
          f"eligible_unrestricted={r.eligible_unrestricted} primary_mentor_uid={r.primary_mentor_uid} n_claims={r.n_claims} "
          f"n_manual_claims={r.n_manual_claims} claimed_after_question={r.claimed_after_question}")
    print(f"  M4 diagnostics: on_roster={r.mentor_on_roster_at_registration} in_auto_pool={r.mentor_in_auto_pool_at_registration} "
          f"assigned_later={r.assigned_later} possible_forcementor={r.possible_forcementor}")
    turns = TURNS.get(rv, [])
    i = first_reply_from_turns(turns, r.q_ts)
    display(turns_table(turns, i))
    if r.treat:
        print(f"  first reply (turn {i}) at {turns[i]['ts']} by {turns[i]['author']} ({turns[i]['role']}); reply lag {r.reply_lag_days:.4f} days"
              f" -> anchor = {r.anchor}")
    else:
        print(f"  no reply by anyone else (turn index {i}); pseudo lag {r.pseudo_lag_days:.4f} days (replayed draw {replay[rv]:.4f})"
              f" -> anchor = q_ts + lag = {ts(r.q_ts) + pd.Timedelta(days=r.pseudo_lag_days)} (stored {r.anchor})")
    ed = BY_UID.get(uid, EMPTY)
    a0 = ts(r.anchor).floor("s")
    w = ed[(ed.ts > a0) & (ed.ts <= a0 + pd.Timedelta(days=180))].copy()
    w["days_after_anchor"] = ((w.ts - a0).dt.total_seconds() / 86400).round(3)
    w["windows"] = w.days_after_anchor.map(lambda d: " ".join(f"<={x}d" for x in (7, 14, 30) if d <= x) or "later")
    print(f"  edits in (anchor, anchor+180d]: {len(w)} (all edits of this person: {len(ed)}); first 25 shown")
    display(w[["timestamp", "days_after_anchor", "windows", "ns", "revid", "tags", "revert_ts", "reverted"]].head(25))
    compare(hand_outcomes(ed, r.anchor), {k: getattr(r, f"oc_{k}") for k in MAIN_KEYS}, title="  all main outcomes:")
    if not r.treat:
        show = ("primary", "n_mainspace_edits_14d", "active_days_30d", "reverted_any", "unique_ns", "mainspace_180d")
        hand, data = {}, {}
        for v in ("median", "p25", "p75", "zero"):
            o_ = hand_outcomes(ed, ts(r.q_ts) + pd.Timedelta(days=info[v]))
            hand.update({f"{k}__{v}": o_[k] for k in show}); data.update({f"{k}__{v}": getattr(r, f"oc_{k}__{v}") for k in show})
        draws = [hand_outcomes(ed, ts(r.q_ts) + pd.Timedelta(days=L)) for L in MULTI[rv]]
        hand.update({f"{k}__multi": float(np.mean([d[k] for d in draws])) for k in show}); data.update({f"{k}__multi": getattr(r, f"oc_{k}__multi") for k in show})
        compare(hand, data, title="  robustness versions (6 outcomes shown here; all 28 are compared in bulk in 5c):")

print("RANDOM TREATED PEOPLE (main sample)")
for r in pick(s[s.treat == 1], N_RANDOM, SEED).itertuples():
    show_person(r)
'''),
code(r'''
print("RANDOM CONTROL PEOPLE (main sample)")
for r in pick(s[s.treat == 0], N_RANDOM, SEED).itertuples():
    show_person(r)
'''),
md("""
### 5c. Bulk recomputation of every outcome on N_BULK random people

The same independent code, run on `N_BULK` random people with outcomes (treated and control), for all 35 main
outcomes and, for the controls among them, all 28 x 5 robustness versions. The table shows the share of rows that
match; rows that do not match are listed.
"""),
code(r'''
bulk = pick(s_any, N_BULK, SEED)
rows, bad = [], []
for r in bulk.itertuples():
    ed = BY_UID.get(int(r.mentee_uid), EMPTY)
    h = hand_outcomes(ed, r.anchor)
    rec = {f"oc_{k}": (h[k], getattr(r, f"oc_{k}")) for k in MAIN_KEYS}
    if not r.treat:
        q0 = ts(r.q_ts)
        for v in ("median", "p25", "p75", "zero"):
            hv = hand_outcomes(ed, q0 + pd.Timedelta(days=info[v]))
            rec.update({f"oc_{k}__{v}": (hv[k], getattr(r, f"oc_{k}__{v}")) for k in VAR_KEYS})
        draws = [hand_outcomes(ed, q0 + pd.Timedelta(days=L)) for L in MULTI[int(r.first_question_revid)]]
        rec.update({f"oc_{k}__multi": (float(np.mean([d[k] for d in draws])), getattr(r, f"oc_{k}__multi")) for k in VAR_KEYS})
    for c, (hv, dv) in rec.items():
        ok = bool(np.isclose(float(hv), float(dv)))
        rows.append((c, ok))
        if not ok:
            bad.append({"uid": r.mentee_uid, "revid": r.first_question_revid, "treat": r.treat, "column": c, "recomputed": hv, "in data": dv})
B = pd.DataFrame(rows, columns=["column", "match"]).groupby("column").match.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "share matching"})
print(f"people: {len(bulk)} ({int(bulk.treat.sum())} treated, {int((bulk.treat == 0).sum())} control); comparisons: {len(rows)}")
print(B.groupby(B["share matching"] == 1).size().rename({True: "columns that match on every row", False: "columns with a mismatch"}).to_string())
display(B[B["share matching"] < 1])
if bad:
    print(f"MISMATCH: {len(bad)} values differ")
    display(pd.DataFrame(bad).head(30))
else:
    print("  -> all fields match")
'''),
md("""
### 5d. Sample funnel: every step recomputed from the flags, with random excluded rows

The steps are applied in the order of `build_sample.funnel`: first the M4 exclusion reasons, then the M2 parse
flags, the empty-text rule, and the language rule. The count at every step is recomputed here and compared with
`results/tables/sample_funnel.csv`; then each person's first failing step is compared with `exclude_step`.
"""),
code(r'''
flags = s_all.parse_flags.map(lambda f: set(f) if f is not None else set())
M4_ORDER = ["temporary_account", "autocreated_account", "created_by_other", "forced_local_account", "no_creation_log",
            "registered_before_mentorship", "alt_or_test_account", "manual_claim_before_question"]
steps = [(r_, f"m4:{r_}", s_all.exclude_reason.eq(r_)) for r_ in M4_ORDER]
steps += [("thread_not_found", "thread_not_found", flags.map(lambda f: "thread_not_found" in f)),
          ("reply_before_question", "reply_before_question", flags.map(lambda f: "reply_before_question" in f)),
          ("question_removed_immediately", "question_removed_immediately", flags.map(lambda f: "section_removed_immediately" in f)),
          ("question_text_hidden", "question_text_hidden", flags.map(lambda f: "unrecoverable_revdel" in f)),
          ("question_text_empty", "question_text_empty", s_all.lang_method.eq("empty")),
          ("not_english", "not_english", s_all.lang.ne("en"))]
left = pd.Series(True, index=s_all.index)
my_step = pd.Series(None, index=s_all.index, dtype=object)
hand = {"tagged first questions before the cutoff (one per asker)": len(s_all)}
for name, label, mask in steps:
    hit = left & mask
    my_step[hit] = label
    left &= ~mask
    hand[f"exclude {name}"] = int(left.sum())
compare(hand, dict(zip(fun.step, fun.remaining)), title="people remaining after each step:")
compare({"removed at each step": [int(hand[a_] - hand[b_]) for a_, b_ in zip(list(hand), list(hand)[1:])]},
        {"removed at each step": list(fun.removed[1:])})
print("each person's first failing step == exclude_step:", bool((my_step.fillna("-") == s_all.exclude_step.fillna("-")).all()))
print("main sample == people who pass every step:", bool((left == s_all.in_sample).all()))
'''),
md("""
Random excluded rows for every step, with the raw evidence:

* M4 reasons: account creation (`creation_action`, `registered_at`; `registered_before_mentorship` means registered
  before 2021-05-04), all reasons, and the claim log entries
  (`data/interim/m4/claims.parquet`) with links;
* `thread_not_found` / `question_text_hidden`: the parse flags and where M2 looked;
* `reply_before_question`: the thread turns with times earlier than the question;
* `question_removed_immediately`: how the section was removed, and the text;
* `question_text_empty`: the raw wikitext. The rule is "nothing is left once URLs are removed from the plain text", so a
  question that is only a bare link also counts as empty;
* `not_english`: the text with the fastText label and probability.
"""),
code(r'''
claims = pd.read_parquet(config.INTERIM / "m4" / "claims.parquet")
for step, g in s_all[s_all.exclude_step.notna()].groupby("exclude_step"):
    print(f"\n{'=' * 100}\n## {step}: {len(g)} people; {min(N_RANDOM, len(g))} random ones")
    for r in pick(g, N_RANDOM, SEED).itertuples():
        rv = int(r.first_question_revid)
        print(f"\n  - {r.mentee_name} (uid {r.mentee_uid}) question {oldid(rv)}  q_ts {r.q_ts}")
        if step.startswith("m4:"):
            print(f"    creation_action={r.creation_action}  registered_at={r.registered_at}  all reasons={list(r.exclude_reasons) if r.exclude_reasons is not None else []}")
            if step in ("m4:alt_or_test_account", "m4:manual_claim_before_question"):
                c = claims[(claims.mentee_uid == r.mentee_uid)]
                for x in c.itertuples():
                    print(f"    claim {x.ts} by {x.claimer} ({x.kind}, alt_or_test={x.alt_or_test}) {logid(x.logid)}  comment: {str(x.comment)[:120]!r}")
        elif step in ("thread_not_found", "question_text_hidden"):
            print(f"    parse_flags={list(r.parse_flags) if r.parse_flags is not None else []}  page={r.page_title_at_q}  thread_origin={r.thread_origin}")
            print(f"    text: {str(r.question_text)[:150]!r}")
        elif step == "reply_before_question":
            q_min = ts(r.q_ts).floor("min")
            for t in TURNS.get(rv, []):
                if t["role"] != "mentee" and t.get("ts") and ts(t["ts"]) < q_min:
                    print(f"    turn by {t['author']} ({t['role']}) at {t['ts']} is BEFORE the question ({r.q_ts}): {str(t['text'])[:90]!r}")
        elif step == "question_removed_immediately":
            print(f"    removal_kind={r.removal_kind}  diff of the question: {diff(rv)}")
            print(f"    text: {str(r.question_text)[:150]!r}")
        elif step == "question_text_empty":
            print(f"    plain text: {r.question_text!r}")
            print(f"    raw wikitext: {str(WIKITEXT.get(rv, ''))[:250]!r}")
        else:
            print(f"    lang={r.lang} prob={r.lang_prob} method={r.lang_method}  text: {str(r.question_text)[:200]!r}")
'''),
md("""
### 5e. Language decisions recomputed with fastText

The rules of `detect_language` (D5) are rewritten here and applied with the fastText model file directly:
remove URLs; empty text -> `empty`; fewer than 3 words -> ASCII rule (`ascii_fallback_short`); mostly capitals ->
lowercase; top probability >= 0.5 -> top label (`fasttext_lid176`); otherwise English if `en` is in the top three and
the ASCII rule passes (`fasttext_uncertain_en`), else the top label (`fasttext_uncertain`). The ASCII rule: at least
80% of the letters are ASCII. For N_RANDOM English and N_RANDOM non-English decisions, and N_RANDOM rows of each
method, we show the text, the top-3 fastText labels, and compare lang, lang_prob, lang_method.
"""),
code(r'''
import fasttext
fasttext.FastText.eprint = lambda *a, **k: None
LID = fasttext.load_model(str(config.RAW / "models" / "lid.176.bin"))

def ascii_ok(t):
    letters = [c for c in t if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.8

def my_language(text):
    t = re.sub(r"https?://\S+", " ", text or "").replace("\n", " ").strip()
    if not t:
        return None, None, "empty", []
    if len(re.findall(r"[^\W\d_]+", t)) < 3:
        return ("en" if ascii_ok(t) else None), None, "ascii_fallback_short", []
    letters = [c for c in t if c.isalpha()]
    if sum(c.isupper() for c in letters) / len(letters) > 0.7:
        t = t.lower()
    top = [(lab.replace("__label__", ""), round(float(p), 4)) for p, lab in LID.f.predict(t, 3, 0.0, "replace")]
    if top[0][1] >= 0.5:
        return top[0][0], top[0][1], "fasttext_lid176", top
    if "en" in dict(top) and ascii_ok(t):
        return "en", dict(top)["en"], "fasttext_uncertain_en", top
    return top[0][0], top[0][1], "fasttext_uncertain", top

def show_lang(g, title):
    print(f"\n## {title}")
    for r in g.itertuples():
        lang, prob, meth, top = my_language(r.question_text or "")
        print(f"\n  revid {r.first_question_revid}  text: {str(r.question_text)[:160]!r}")
        print(f"  fastText top 3: {top}   ascii rule: {r.is_english_ascii_rule}   langdetect: {r.lang_langdetect}")
        compare({"lang": lang, "lang_prob": prob, "lang_method": meth}, {"lang": r.lang, "lang_prob": r.lang_prob, "lang_method": r.lang_method})

show_lang(pick(s_all[s_all.is_english], N_RANDOM, SEED), "random decisions: English")
show_lang(pick(s_all[~s_all.is_english & s_all.lang_method.ne("empty")], N_RANDOM, SEED), "random decisions: not English")
for meth in ("ascii_fallback_short", "fasttext_uncertain_en", "fasttext_uncertain"):
    show_lang(pick(s_all[s_all.lang_method == meth], N_RANDOM, SEED), f"random rows decided by {meth}")
'''),
md("## 6. Edge-case lists"),
code(r'''
print("Treated people whose reply came 365 days or more after the question (flag long_reply_lag; anchor is still the real reply):")
display(s[s.long_reply_lag][["mentee_uid", "first_question_revid", "q_ts", "first_reply_ts", "reply_lag_days"]].head(10))
print(f"\nTreated with the reply in the same minute as the question (reply_lag_days < 0 because signatures have no seconds): "
      f"{int((s.reply_lag_days < 0).sum())}")
display(s[s.reply_lag_days < 0][["mentee_uid", "first_question_revid", "q_ts", "first_reply_ts", "reply_lag_days"]].head(5))
print(f"\nFollow-up incomplete (anchor + 14 days after the cutoff): {int((s.oc_eligible_14d == 0).sum())} people; latest anchors:")
display(s.sort_values("anchor", ascending=False)[["mentee_uid", "treat", "q_ts", "anchor", "oc_eligible_7d", "oc_eligible_14d", "oc_eligible_180d"]].head(5))
ea = E.merge(s_any[["mentee_uid", "anchor"]], left_on="uid", right_on="mentee_uid")
tie = ea[ea.ts == ts(ea.anchor).dt.floor("s")]
print(f"\nEdits saved in exactly the anchor second (not counted, outcomes use edits strictly after the anchor): {len(tie)}")
print("  (for treated people the anchor is a signature time with whole minutes, so an edit at hh:mm:00 may in fact come after the reply)")
display(tie[["uid", "revid", "timestamp", "ns", "anchor"]].head(5))
late = E[E.has_reverted_tag & E.revert_ts.notna() & (E.revert_ts >= CUTOFF)]
print(f"\nEdits with the mw-reverted tag whose revert happened after the cutoff (counted as NOT reverted): {len(late)}")
display(late[["uid", "revid", "timestamp", "revert_ts"]].head(5))
print(f"Edits with the mw-reverted tag and no known revert time (counted as reverted, by the tag): {int((E.has_reverted_tag & E.revert_ts.isna()).sum())}")
print(f"\nPeople only in the unrestricted sample (robustness): {int((s_all.in_sample_unrestricted & ~s_all.in_sample).sum())}; by M4 reason:",
      s_all[s_all.in_sample_unrestricted & ~s_all.in_sample].exclude_step.value_counts().to_dict())
print("\nFirst replier role among treated:", s[s.treat == 1].first_replier_role.value_counts(dropna=False).to_dict())
p = M5 / "_issues" / "summary.json"
if p.exists():
    print("\nM5 issue log:", json.dumps({k: v["n"] for k, v in json.loads(p.read_text()).items()}))
'''),
md("## 7. Distributions: reply lags and pseudo lags, language methods, follow-up, outcome means"),
code(r'''
print("treated reply lag (days), from lag_info.json:", {k: round(v, 3) for k, v in info.items() if isinstance(v, float)})
print("control pseudo lag quantiles:", s[s.treat == 0].pseudo_lag_days.quantile([.25, .5, .75]).round(3).to_dict())
print("treated real lag quantiles:   ", s[s.treat == 1].reply_lag_days.quantile([.25, .5, .75]).round(3).to_dict())
print("\nlanguage method (all askers):", s_all.lang_method.value_counts().to_dict())
print("ASCII rule (paper) vs new decision:")
display(pd.crosstab(s_all.is_english_ascii_rule, s_all.is_english))
print("langdetect label among rows the new rule calls English (top 5):", s_all[s_all.is_english].lang_langdetect.value_counts().head(5).to_dict())
print("\nshare with complete follow-up:", {w: round(float(s[f"oc_eligible_{w}d"].mean()), 4) for w in WINDOWS})
print("\nmain outcomes, mean by treatment (main sample, complete 14-day follow-up):")
display(s[s.oc_eligible_14d == 1].groupby("treat")[[f"oc_{k}" for k in ("primary", "n_mainspace_edits_14d", "cross_day_any_14d", "unique_ns",
        "active_days_30d", "reverted_any", "primary_constructive", "sec2")]].mean().round(4).T)
'''),
md("## 8. Summary (results/m5_report.json)"),
code(r'''
report = {"n_askers": len(s_all), "n_sample": len(s), "n_unrestricted": int(s_all.in_sample_unrestricted.sum()),
          "treated": int(s.treat.sum()), "control": int((s.treat == 0).sum()),
          "funnel": fun.to_dict("records"), "paper_overlap": len(m),
          "treat_agreement_with_paper": round(float((m.paper_treat == m.treat).mean()), 4)}
(config.RESULTS / "m5_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in report.items() if k != "funnel"}, indent=1))
print(f"invariants: {sum(ok for _, ok in ck.items)} / {len(ck.items)} passed; bulk outcome comparisons with a mismatch: {len(bad)}")
'''),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "m5_check.ipynb"
    write_notebook(CELLS, out)
    print("wrote", out)
    if "--run" in sys.argv:
        run_as_script(out, cwd=out.parent.parent)
        print("\nnotebook ran end to end")


if __name__ == "__main__":
    main()
