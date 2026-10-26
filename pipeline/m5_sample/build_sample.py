"""
build_sample.py: M5 analysis sample, treatment variables, start times (anchors), and outcome variables

Module: M5 (PLAN.md M5; CODE_PLAN.md section 5, M5)
Rules: DATA_PROCESSING R5.3, R5.4, R6.4, R7.2 to R7.5, R7.8. Decisions: D5, D10, D11

1. What it does
    1. First question (R6.4): for each asker (by uid), take their earliest tagged question (already given by M4).
    2. Sample funnel (paper Appendix Table A). Remove, in this order:
        - the M4 exclusion reasons (account type, registration time, mentor alt accounts, claimed by hand before
          the question);
        - thread not found (thread_not_found): the treatment cannot be determined;
        - reply earlier than the question (reply_before_question, R5.4): the section location is doubtful, and
          since these cannot all be checked by hand one by one, they are all removed;
        - question deleted in the very next edit (section_removed_immediately, mostly reverted spam): it never
          had a chance to get a reply on the page;
        - question body removed by revision deletion (unrecoverable_revdel): we cannot tell the language or compute
          text features;
        - question body is empty (only a signature, or only templates or links): there is no question content to
          analyze;
        - not English (D5: fastText lid.176, see detect_language).
    3. Treatment variables (R5.3):
        treat     = someone other than the newcomer replied in the thread before the cutoff (main definition);
        treat_mentor = only replies by the assigned mentor (the talk page owner or the primary mentor) count;
        treat_48h = there was a reply within 48 hours.
    4. Start time a_i: treatment group = time of the first reply. Control group = question time + L_i, where L_i is
       drawn at random from the reply delays of the treatment group in the sample (0 to 365 days), with a fixed
       random seed (RandomState(42); draws are made in order after sorting by first_q_ts, revid).
       We also build the same robustness versions as the paper version: median, p25, p75, zero delay, and the
       average over 20 draws (RandomState(123)).
    5. Outcome variables (outcomes.build_outcomes) and flags for whether follow-up is complete.

2. Why it is designed this way
    - We drop the groups "thread not found", "reply earlier than question", "question deleted right away", and
      "body hidden" because their treatment or text cannot be determined reliably. The count for each group goes
      into the funnel table, and the M8 robustness checks add back the groups that can be kept.
    - Language detection (D5) uses fastText's official language identification model lid.176. The paper version
      only looked at the share of ASCII letters (≥ 0.8). That really means "Latin letters", so Spanish,
      Indonesian, and similar languages were also counted as English. langdetect makes many mistakes on short
      English questions (see detect_language). Results from both of these methods are kept as comparison columns
      (is_english_ascii_rule, lang_langdetect).
    - The pseudo start time for the control group uses the same method as the paper version (section R5.3, PLAN M5
      item 4). When a treated person's reply delay is more than 365 days, we still use the real reply time as the
      start time (the paper version switched to a random draw here, which is the same kind of problem as I-14),
      and flag it as long_reply_lag.

3. Inputs
    data/interim/m2/questions.parquet; data/interim/m4/eligibility.parquet; data/interim/m3/mentee_edits/,
    mentee_profile.parquet; data/raw/m3/revert_times*.jsonl (R7.8, used if present).

4. Outputs
    data/interim/m5/sample.parquet: one row per asker. in_sample = main analysis sample; in_sample_unrestricted =
        the sample without the D10 limits (robustness check). People in both samples have start times and outcome
        variables (the pseudo-delay distribution uses only the treatment group of the main sample).
    results/tables/sample_funnel.csv: funnel table (paper Table A)
    data/interim/m5/lag_info.json: distribution of reply delays in the treatment group (median, p25, p75)

8. How to check
    checks/m5_check.ipynb; tests/test_m5.py.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import config
from pipeline.common.io import iter_jsonl, write_json_atomic, write_table
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import parse_iso
from pipeline.m5_sample.outcomes import OC_KEYS, build_outcomes, edit_arrays

OUT = config.INTERIM / "m5"
LAG_MAX_DAYS = 365
N_MULTI = 20
SHORT_WORDS = 5
FIXED_LAGS = ("median", "p25", "p75", "zero")


# ── Language (D5) ────────────────────────────────────────────────────────────────
def ascii_rule(text: str, threshold: float = 0.8) -> bool:
    """Paper-version rule: ASCII letters make up ≥ 0.8 of all letters (in practice this is a "Latin letters" test)."""
    letters = [c for c in text or "" if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= threshold


LID_MODEL = config.RAW / "models" / "lid.176.bin"
LID_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
MIN_WORDS_LID = 3
_LID = None


def _lid():
    """fastText's official language identification model lid.176 (full .bin version,
    https://fasttext.cc/docs/en/language-identification.html).

    In our tests the compressed version lid.176.ftz got short English questions wrong ("Hi, how do I create a new
    page?" was labelled Persian). The full version gets them right.
    """
    global _LID
    if _LID is None:
        import hashlib
        import fasttext
        h = hashlib.sha256(LID_MODEL.read_bytes()).hexdigest()
        if h != LID_SHA256:
            raise RuntimeError(f"{LID_MODEL} sha256 {h} != {LID_SHA256}")
        fasttext.FastText.eprint = lambda *a, **k: None
        _LID = fasttext.load_model(str(LID_MODEL))
    return _LID


def detect_language(text: str) -> tuple[str | None, float | None, str]:
    """D5: detect the language with fastText lid.176 and take the most likely language. Returns (language,
    probability, method).

    Why not langdetect: it is very unstable on short English questions (in our tests "How do I create a
    citation?" was labelled Italian and "How do I add my music duo?" was labelled Welsh), while lid.176 was
    trained for short texts (Wikipedia, Tatoeba, SETimes).
    We call the lower-level model.f.predict directly, because predict in the fasttext package does not work with
    NumPy 2.
    Rules: text in all capitals is lowercased first. If the top probability is ≥ 0.5, take the top language.
    Otherwise, count it as English if en is in the top three and the ASCII rule passes. Texts with fewer than 3
    words fall back to the ASCII rule. The method used is written to lang_method so it can be checked.
    """
    t = re.sub(r"https?://\S+", " ", text or "").replace("\n", " ").strip()
    if not t:
        return None, None, "empty"
    if len(re.findall(r"[^\W\d_]+", t)) < MIN_WORDS_LID:
        # one or two words are not enough to tell the language reliably ("hi" gets labelled Catalan), so fall back to the paper-version ASCII rule
        return ("en" if ascii_rule(t) else None), None, "ascii_fallback_short"
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        t = t.lower()                                      # the model is unreliable on all-caps text ("DO I HAVE A HISTORY?" was labelled Ukrainian)
    top = [(label.replace("__label__", ""), float(p)) for p, label in _lid().f.predict(t, 3, 0.0, "replace")]
    lang, prob = top[0]
    if prob >= 0.5:
        return lang, round(prob, 4), "fasttext_lid176"
    # low confidence (mostly short, informal English, e.g. "hi how do i create a citetation" labelled Vietnamese 0.12):
    # if en is in the top three and the ASCII rule also passes, count it as English
    if any(l == "en" for l, _ in top) and ascii_rule(t):
        return "en", round(dict(top)["en"], 4), "fasttext_uncertain_en"
    return lang, round(prob, 4), "fasttext_uncertain"


def langdetect_language(text: str) -> str | None:
    """For comparison only: langdetect (fixed seed)."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0
    try:
        return detect(re.sub(r"https?://\S+", " ", text or ""))
    except LangDetectException:
        return None


# ── Revert times (R7.8) ─────────────────────────────────────────────────────────
def revert_lookup() -> dict[int, np.datetime64]:
    """revid -> revert time (M3 revert_times; edits with no revert found are left out)."""
    out = {}
    for name in ("revert_times_sample.jsonl", "revert_times.jsonl"):
        p = config.RAW / "m3" / name
        for r in iter_jsonl(p) if p.exists() else []:
            if r.get("revert_ts"):
                out[int(r["revid"])] = np.datetime64(r["revert_ts"].replace("Z", ""), "s")
    return out


# ── Sample funnel ─────────────────────────────────────────────────────────────────
def build_base(log) -> pd.DataFrame:
    """One row per asker: first-question details + M4 eligibility + language."""
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet").set_index("question_revid")
    e = pd.read_parquet(config.INTERIM / "m4" / "eligibility.parquet")
    prof = pd.read_parquet(config.INTERIM / "m3" / "mentee_profile.parquet").set_index("mentee_uid")
    f = q.loc[e.first_question_revid].reset_index()
    df = e.merge(f[["question_revid", "q_ts", "tag", "heading", "page_title_at_q", "page_mentor_uid",
                    "primary_mentor_uid", "away_routed", "editing_page", "article_context", "question_text",
                    "question_wikitext", "has_any_reply", "first_reply_ts", "first_replier_role",
                    "has_mentor_reply", "mentor_first_reply_ts", "thread_origin", "parse_flags", "removal_kind",
                    "mentee_name"]].rename(columns={"primary_mentor_uid": "q_primary_mentor_uid"}),
                 left_on="first_question_revid", right_on="question_revid", how="left")
    df["late_fetch"] = df.mentee_uid.map(prof.late_fetch)
    langs = [detect_language(t) for t in df.question_text.fillna("")]
    df["lang"], df["lang_prob"], df["lang_method"] = zip(*langs)
    df["is_english"] = df.lang == "en"
    df["is_english_ascii_rule"] = df.question_text.fillna("").map(ascii_rule)
    df["lang_langdetect"] = df.question_text.fillna("").map(langdetect_language)
    log.info("language: english %d / %d (ascii rule %d)", int(df.is_english.sum()), len(df),
             int(df.is_english_ascii_rule.sum()))
    return df


def funnel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the exclusion rules in order. Returns (table with exclude_step, funnel table)."""
    flags = df.parse_flags.map(lambda f: set(f) if f is not None else set())
    steps = [
        ("thread_not_found", flags.map(lambda s: "thread_not_found" in s)),
        ("reply_before_question", flags.map(lambda s: "reply_before_question" in s)),
        ("question_removed_immediately", flags.map(lambda s: "section_removed_immediately" in s)),
        ("question_text_hidden", flags.map(lambda s: "unrecoverable_revdel" in s)),
        ("question_text_empty", df.lang_method.eq("empty")),
        ("not_english", ~df.is_english),
    ]
    df = df.copy()
    df["exclude_step"] = None
    rows = [{"step": "tagged first questions before the cutoff (one per asker)", "removed": 0, "remaining": len(df)}]
    order = ["temporary_account", "autocreated_account", "created_by_other", "forced_local_account", "no_creation_log",
             "registered_before_mentorship", "alt_or_test_account", "manual_claim_before_question"]
    for reason in order:
        m = (df.exclude_step.isna()) & (df.exclude_reason == reason)
        df.loc[m, "exclude_step"] = f"m4:{reason}"
        rows.append({"step": f"exclude {reason}", "removed": int(m.sum()), "remaining": int(df.exclude_step.isna().sum())})
    for name, mask in steps:
        m = df.exclude_step.isna() & mask
        df.loc[m, "exclude_step"] = name
        rows.append({"step": f"exclude {name}", "removed": int(m.sum()), "remaining": int(df.exclude_step.isna().sum())})
    # sample without the D10 limits (R2.5 robustness check): skip only the account-type and registration-time reasons; all other steps are the same
    d10 = {"autocreated_account", "created_by_other", "forced_local_account", "no_creation_log",
           "registered_before_mentorship"}
    other = df.exclude_reasons.map(lambda rs: [r for r in (rs if rs is not None else []) if r not in d10])
    data_ok = ~(flags.map(lambda s: bool(s & {"thread_not_found", "reply_before_question", "section_removed_immediately",
                                                "unrecoverable_revdel"}))) & df.is_english
    df["in_sample_unrestricted"] = other.map(len).eq(0) & data_ok
    return df, pd.DataFrame(rows)


# ── Treatment, start times, outcomes ─────────────────────────────────────────────────
def treatments(df: pd.DataFrame) -> pd.DataFrame:
    """The three treatment variables (R5.3) and the reply delay."""
    q_ts = pd.to_datetime(df.q_ts, utc=True)
    r_ts = pd.to_datetime(df.first_reply_ts, utc=True)
    df["treat"] = df.has_any_reply.fillna(False).astype(int)
    df["treat_mentor"] = df.has_mentor_reply.fillna(False).astype(int)
    df["reply_lag_days"] = (r_ts - q_ts).dt.total_seconds() / 86400
    df["treat_48h"] = ((df.treat == 1) & (df.reply_lag_days <= 2)).astype(int)
    df["long_reply_lag"] = df.reply_lag_days >= LAG_MAX_DAYS
    return df


def anchors(s: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Start times (section 1, item 4 above). s contains only people in the sample."""
    s = s.sort_values(["q_ts", "first_question_revid"]).copy()
    main = s[s.in_sample]
    lags = main.loc[(main.treat == 1) & (main.reply_lag_days >= 0) & (main.reply_lag_days < LAG_MAX_DAYS),
                    "reply_lag_days"].to_numpy()
    info = {"n_lags": int(len(lags)), "median": float(np.median(lags)), "p25": float(np.percentile(lags, 25)),
            "p75": float(np.percentile(lags, 75)), "mean": float(lags.mean()), "zero": 0.0, "n_multi": N_MULTI}
    rng = np.random.RandomState(config.SEED)
    q_ts = pd.to_datetime(s.q_ts, utc=True)
    a, pseudo = [], []
    for t, treated, rts in zip(q_ts, s.treat, pd.to_datetime(s.first_reply_ts, utc=True)):
        if treated:
            a.append(rts); pseudo.append(np.nan)
        else:
            lag = float(rng.choice(lags)); a.append(t + timedelta(days=lag)); pseudo.append(lag)
    s["anchor"] = a
    s["pseudo_lag_days"] = pseudo
    return s, info


def all_outcomes(s: pd.DataFrame, info: dict, log) -> pd.DataFrame:
    """For each person: outcomes at the main start time, plus the fixed-delay and multi-draw versions for the control group (for the treatment group these versions equal the main outcomes)."""
    edits = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "revid", "timestamp", "ns", "tags"],
        filter=ds.field("uid").isin(list(s.mentee_uid))).to_pandas()
    edits["timestamp"] = pd.to_datetime(edits.timestamp, utc=True).dt.tz_localize(None)
    groups = dict(tuple(edits.groupby("uid")))
    rv = revert_lookup()
    log.info("edits for sample: %d; revert times known: %d", len(edits), len(rv))
    rng = np.random.RandomState(123)
    lags = None
    rows, t0 = [], time.time()
    empty = edits.iloc[0:0]
    for i, r in enumerate(s.itertuples(), 1):
        arr = edit_arrays(groups.get(r.mentee_uid, empty), rv)
        oc = build_outcomes(arr, r.anchor.to_pydatetime())
        row = {"mentee_uid": r.mentee_uid, **{f"oc_{k}": v for k, v in oc.items()}}
        q = pd.Timestamp(r.q_ts).to_pydatetime()
        for name in FIXED_LAGS:
            o = oc if r.treat else build_outcomes(arr, q + timedelta(days=info[name]))
            row.update({f"oc_{k}__{name}": o[k] for k in OC_KEYS})
        if r.treat:
            row.update({f"oc_{k}__multi": oc[k] for k in OC_KEYS})
        else:
            if lags is None:
                m = s[s.in_sample]
                lags = m.loc[(m.treat == 1) & (m.reply_lag_days >= 0) & (m.reply_lag_days < LAG_MAX_DAYS),
                             "reply_lag_days"].to_numpy()
            draws = [build_outcomes(arr, q + timedelta(days=float(rng.choice(lags)))) for _ in range(N_MULTI)]
            row.update({f"oc_{k}__multi": float(np.mean([d[k] for d in draws])) for k in OC_KEYS})
        rows.append(row)
        if i % 5000 == 0:
            log.info("  outcomes %d/%d (%.0f s)", i, len(s), time.time() - t0)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    base_parser("M5: sample, treatment, anchors, outcomes.").parse_args(argv)
    log = setup_logging("m5")
    t0 = time.time()
    check_upstream("m5", ["m2", "m3", "m4"])
    issues = IssueLog("m5")
    base = treatments(build_base(log))
    df, fun = funnel(base)
    df["in_sample"] = df.exclude_step.isna()
    s = df[df.in_sample | df.in_sample_unrestricted].copy()
    s, info = anchors(s)
    oc = all_outcomes(s, info, log)
    s = s.merge(oc, on="mentee_uid", how="left")
    out = pd.concat([s, df[~(df.in_sample | df.in_sample_unrestricted)]], ignore_index=True)
    s = s[s.in_sample]
    for r in s[s.long_reply_lag].itertuples():
        issues.add("long_reply_lag", rule="R5.3", action="flagged", ids={"uid": r.mentee_uid, "revid": r.first_question_revid},
                   detail={"lag_days": round(r.reply_lag_days, 1)})
    OUT.mkdir(parents=True, exist_ok=True)
    out["anchor"] = out.anchor.astype(str).where(out.anchor.notna(), None)
    write_table(OUT / "sample.parquet", out.drop(columns=["question_wikitext"]))
    write_json_atomic(OUT / "lag_info.json", info)
    (config.RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    fun.to_csv(config.RESULTS / "tables" / "sample_funnel.csv", index=False)
    issues.write()
    counts = {"askers": len(df), "sample": len(s), "treated": int(s.treat.sum()), "control": int((s.treat == 0).sum()),
              "treat_mentor": int(s.treat_mentor.sum()), "treat_48h": int(s.treat_48h.sum()),
              "eligible_14d": int(s.oc_eligible_14d.sum()), "primary_mean_treated": round(float(s[s.treat == 1].oc_primary.mean()), 4),
              "primary_mean_control": round(float(s[s.treat == 0].oc_primary.mean()), 4), "lag_info": info}
    log.info("funnel:\n%s", fun.to_string(index=False))
    log.info("done: %s", counts)
    write_manifest("m5", inputs=[config.INTERIM / "m4" / "eligibility.parquet"],
                   outputs=[OUT / "sample.parquet", config.RESULTS / "tables" / "sample_funnel.csv"],
                   params={"seed": config.SEED, "lag_max_days": LAG_MAX_DAYS, "n_multi": N_MULTI}, counts=counts, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
