"""
build_m6_check.py: writes checks/m6_check.ipynb (checks of the M6 text features, LLM labels, Perspective scores,
embeddings, and reply features)

The notebook follows the fixed 8-section layout of PLAN.md section 12:
    1 module description  2 files, manifest, coverage, data dictionaries  3 invariants (Checks)
    4 comparison with the paper version (labels, scores and old features for the same question)
    5 tracer and random samples: raw wikitext and plain text next to every question feature (simple ones recomputed
      by hand, VADER/TextBlob/ConvoKit called directly, with the words that triggered each politeness strategy);
      LLM labels with the codebook definitions; Perspective extremes; random replies with pronoun tokens,
      imperative sentences and every link; cosine similarity recomputed from the embedding files
    6 edge-case lists  7 distributions (kappa, cosine)  8 summary (results/m6_report.json)

Run:  python checks/build_m6_check.py          # write the notebook
      python checks/build_m6_check.py --run    # write it and run it from top to bottom
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks._nbgen import code, md, run_as_script, write_notebook  # noqa: E402

CELLS = [
md("""
# M6 text features and text labels: checks

## 1. Module description

**What M6 does.** `pipeline/m6_text/` (`build_text.py`, `text_features.py`, `services.py`, `cache.py`,
`agreement.py`) describes the text of every sampled first question and of the first reply in the treated group:

1. **Question features (37 columns, `text_features.question_features`).** Wiki format features parsed with
   mwparserfromhell (signature, links, templates, URL, indentation, length), VADER and TextBlob sentiment, and the
   21 ConvoKit politeness strategies (decision D6).
2. **Question types (5 columns).** DeepSeek labels with the paper codebook and 120 human-labelled examples:
   `q_substantive` (Q0), `q_referent` (Q2), `q_appraisal` (Q3), `q_normative` (Q4), `q_own_work` (Q5).
3. **Perspective scores (8 columns)** and **Qwen embeddings** (text-embedding-v4, 1024 dimensions) of the question
   text with user names removed and links replaced by `[LINK]`. Paper-version results are reused when the text is
   the same (or almost the same); missing values are left missing, never filled with 0.
4. **Reply features (21 columns)** of the first replier's comments: words, personal pronoun rates (spaCy
   morphology), imperative sentences (spaCy dependency parse), links from the DiscussionTools HTML (wikilinks,
   policy or guideline pages, Help pages, external links), lag, readability, sentiment, four politeness strategies,
   lists, question marks, and the cosine similarity between question and reply embeddings (I-27).

**Paper numbers that depend on M6.** The question-text covariates of the propensity score model (Sections 3.4 and
4, through M7), the question-type results and the agreement table (Table 3, I-54), and the reply analyses
(Tables 6 and 7).

**How to use this notebook.** Run it from top to bottom. Each check prints `[OK ]` or `[FAIL]`. Each comparison of
a hand recomputation with the stored value prints a table with a `match` column and ends with `all fields match`
or `MISMATCH in: [...]`. Random rows are drawn with `SEED` in the next cell: change `SEED` and run again to see
fresh rows. spaCy and ConvoKit run only on the sampled rows, so the notebook stays fast.
"""),
code(r'''
import json, sys, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.common.io import iter_jsonl
from pipeline.common.textclean import AWAY_NOTE_RE, make_emb_text
from pipeline.m6_text.cache import text_sha
from pipeline.m6_text.build_text import emb_text_from_plain
from checks._show import Checks, compare, pick, oldid, diff, contribs, logid

SEED = 0        # change SEED to any other integer to draw a fresh random sample in every section below
N_RANDOM = 5    # number of random rows shown in each random check

pd.set_option("display.max_colwidth", 110); pd.set_option("display.width", 230); pd.set_option("display.max_rows", 200)
M6 = config.INTERIM / "m6"
qf = pd.read_parquet(M6 / "question_features.parquet")
rf = pd.read_parquet(M6 / "reply_features.parquet")
s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet")
s = s[s.in_sample | s.in_sample_unrestricted]
S = s.set_index("first_question_revid")
txt = S.question_text
WT = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["question_revid", "question_wikitext"]
                     ).set_index("question_revid").question_wikitext.reindex(S.index).fillna("")
need = set(S.index)
TURNS = {r["question_revid"]: r["turns"] for r in iter_jsonl(config.INTERIM / "m2" / "turns.jsonl") if r["question_revid"] in need}
QI, RI = qf.set_index("first_question_revid"), rf.set_index("first_question_revid")
ts = lambda x: pd.to_datetime(x, utc=True, format="ISO8601")         # noqa: E731
print(f"sampled questions {len(S)} | question feature rows {len(qf)} | reply feature rows {len(rf)}")
'''),
md("""
### Helpers written for this notebook

* `my_first_reply(turns, q_ts)`: the first turn by anyone who is not the newcomer, at or after the question minute,
  and then every following turn by the same person until someone else speaks. The reply text is these turns joined
  by a blank line; their links are added up.
* `my_question_format(wikitext, plain)`: simple regex versions of the wiki format features, written without
  mwparserfromhell, as an independent cross-check. The body is the text before the first signature timestamp
  (`hh:mm, d Month yyyy (UTC)`), after removing the away note (R4.4).
"""),
code(r'''
def my_first_reply(turns, q_ts):
    q_min = ts(q_ts).floor("min")
    i = next((k for k, t in enumerate(turns) if t["role"] != "mentee" and t.get("ts") and ts(t["ts"]) >= q_min), None)
    if i is None:
        return None, "", {"wikilinks": [], "n_external": 0, "n_list_items": 0}, []
    who, parts, links, idx = turns[i]["author"], [], {"wikilinks": [], "n_external": 0, "n_list_items": 0}, []
    for k in range(i, len(turns)):
        t = turns[k]
        if t["author"] != who:
            break
        idx.append(k); parts.append(t["text"])
        lk = t.get("links") or {}
        links["wikilinks"] += lk.get("wikilinks", []); links["n_external"] += lk.get("n_external", 0); links["n_list_items"] += lk.get("n_list_items", 0)
    return i, "\n\n".join(parts), links, idx

SIG = re.compile(r"\d{2}:\d{2},\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}\s*\(UTC\)")
LINK = re.compile(r"\[\[\s*:?\s*([^\[\]|#]*)")
USER_PREFIX = ("user:", "user talk:", "u:", "ut:")
WP_PREFIX = ("wikipedia:", "wp:", "project:", "wikipedia talk:", "wt:")
HELP_PREFIX = ("help:", "h:")

def link_titles(body):
    out = []
    for t in LINK.findall(body):
        t = re.sub(r"\s*:\s*", ":", t.strip().lower().replace("_", " "))
        if not t.startswith(("http://", "https://", "//")):
            out.append(t)
    return out

def top_level_templates(b):
    n = depth = i = 0
    while i < len(b) - 1:
        if b[i:i + 2] == "{{":
            n += depth == 0; depth += 1; i += 2
        elif b[i:i + 2] == "}}" and depth:
            depth -= 1; i += 2
        else:
            i += 1
    return n

def my_question_format(wikitext, plain):
    w = AWAY_NOTE_RE.sub("", wikitext or "", count=1)
    m = SIG.search(w)
    body = w[:m.start()] if m else w
    content = [t for t in link_titles(body) if not t.startswith(USER_PREFIX)]
    return {"q_has_sig": int(m is not None),
            "q_has_unsigned": int(bool(re.search(r"\{\{\s*(subst:\s*)?unsigned", w, re.I))),
            "q_link_count": len(content),
            "q_template_count": top_level_templates(body),
            "q_has_url": int(bool(re.search(r"https?://", body, re.I))),
            "q_mentions_wp": sum(t.startswith(WP_PREFIX[:3]) for t in content),
            "q_mentions_help": sum(t.startswith(HELP_PREFIX) for t in content),
            "q_is_indented": int(w.lstrip().startswith((":", "*"))),
            "q_body_char_len": len(plain or ""),
            "q_body_word_count": len((plain or "").split())}
FORMAT_COLS = list(my_question_format("", ""))

def names_in(rv):
    """User names seen in the thread plus the asker's name (removed from the texts sent to Perspective and the embedding model)."""
    return [t["author"] for t in TURNS.get(rv, []) if t.get("author")] + [S.at[rv, "mentee_name"]]
'''),
md("""
## 2. Files, manifest, coverage, and data dictionaries
"""),
code(r'''
for p in sorted(M6.glob("*")):
    if p.is_file():
        print(f"  {p.relative_to(config.ROOT)}  {p.stat().st_size:>13,} bytes  modified {pd.Timestamp(p.stat().st_mtime, unit='s', tz='UTC'):%Y-%m-%d %H:%M} UTC")
man = json.loads((M6 / "manifest.json").read_text())
print("\nfinished:", man["finished_at"], "| params:", man["params"])
cov = json.loads((M6 / "coverage.json").read_text())
print("coverage.json:", json.dumps(cov))
'''),
md("""
### Data dictionary: `question_features.parquet` (one row per sampled first question) and `reply_features.parquet`
(one row per treated question whose first reply has text)

The `checked in` column names the cell that verifies each column (3 = invariants on all rows, 4 = paper comparison,
5a to 5f = random samples, 6 = edge cases).
"""),
code(r'''
POLITE = {"Please": "contains 'please' inside the sentence", "Please_start": "starts with 'please'",
          "HASHEDGE": "contains a hedge word (think, maybe, suggest, ...)", "Indirect_(btw)": "contains 'by the way'",
          "Hedges": "a hedge word with its context (ConvoKit's second hedge list)", "Factuality": "'in fact', 'actually', 'the point is', ...",
          "Deference": "starts with 'great', 'good', 'nice', 'interesting', ...", "Gratitude": "'thanks', 'thank you', 'I appreciate'",
          "Apologizing": "'sorry', 'apologize', 'excuse me', ...", "1st_person_pl.": "'we', 'our', 'us' inside the text",
          "1st_person": "'I', 'my', 'mine', 'me' inside the text", "1st_person_start": "a sentence starts with 'I' or 'my'",
          "2nd_person": "'you', 'your' inside the text", "2nd_person_start": "a sentence starts with 'you' or 'your'",
          "Indirect_(greeting)": "starts with a greeting ('hi', 'hello', 'hey')", "Direct_question": "a question starting with 'what', 'why', 'who', 'how'",
          "Direct_start": "a sentence starts with 'so' or 'then' or 'and' or 'but'", "HASPOSITIVE": "contains a word from the positive lexicon",
          "HASNEGATIVE": "contains a word from the negative lexicon", "SUBJUNCTIVE": "'could you' or 'would you'",
          "INDICATIVE": "'can you' or 'will you'"}
CODEBOOK = {"q_substantive": "Q0: the message contains a question, request, or intent to contribute about Wikipedia editing, accounts, processes or community (not only a greeting, thanks, gibberish, off-topic, non-English, a bot check, feelings, or a self-introduction).",
            "q_referent": "Q2: the user lacks task direction: no task at all, or only 'create/write a page/article' with no specific topic or content category.",
            "q_appraisal": "Q3: the user asks the mentor to examine and judge their own existing Wikipedia work (review it, check it, is it okay, how can I improve it).",
            "q_normative": "Q4: the user seeks information about rules, norms or permissions: an enforcement action on their work (deleted, declined, reverted) or normative words (allowed, policy, notability, COI, copyright, ...).",
            "q_own_work": "Q5: the user has already done editing on Wikipedia with visible results (created, edited, submitted; 'my draft'; 'my edit was reverted')."}
QD = {"first_question_revid": ("revision id of the first question (key)", "3"),
      "q_has_sig": ("the question has a signature timestamp 'hh:mm, d Month yyyy (UTC)'", "3, 5a"),
      "q_has_unsigned": ("the question uses a template of the {{unsigned}} family", "5a, 6"),
      "q_link_count": ("internal links in the body that do not point to User or User talk pages", "5a, 6"),
      "q_template_count": ("top-level templates in the body", "5a, 6"),
      "q_has_url": ("the body has an external link or a bare URL", "3, 5a"),
      "q_mentions_wp": ("links to the Wikipedia: namespace (ns 4, including WP: aliases)", "3, 5a"),
      "q_mentions_help": ("links to the Help: namespace (ns 12, including the H: shortcuts)", "3, 5a"),
      "q_is_indented": ("the text starts with ':' or '*' (talk-page indentation)", "3, 5a"),
      "q_body_char_len": ("characters in the plain text", "3, 5a"),
      "q_body_word_count": ("whitespace-separated words in the plain text", "3, 5a"),
      "q_vader_neg": ("VADER negative share", "3, 5a"), "q_vader_neu": ("VADER neutral share", "3, 5a"),
      "q_vader_pos": ("VADER positive share", "3, 5a"), "q_vader_compound": ("VADER compound score in [-1, 1]", "3, 4, 5a"),
      "q_tb_polarity": ("TextBlob polarity in [-1, 1]", "3, 5a"), "q_tb_subjectivity": ("TextBlob subjectivity in [0, 1]", "3, 5a")}
for k, v in POLITE.items():
    QD[f"q_poly_feature_politeness_=={k}=="] = (f"ConvoKit politeness strategy {k}: {v}", "3, 5a")
for c, v in CODEBOOK.items():
    QD[c] = (f"LLM label (1 = Y). {v}", "3, 4, 5b")
for a in ("toxicity", "severe_toxicity", "identity_attack", "insult", "profanity", "threat", "sexually_explicit", "flirtation"):
    QD[f"q_persp_{a}"] = (f"Perspective API score for {a.upper()} in [0, 1] (missing when the service gave no score)", "3, 5c")
QD["has_llm"] = ("the five LLM labels exist for this question", "3, 6")
QD["has_persp"] = ("the eight Perspective scores exist", "3, 6")
QD["has_emb_q"] = ("a question embedding exists (row in emb_q_sample.npy is finite)", "3, 5f, 6")
RD = {"first_question_revid": ("revision id of the question that was answered (key)", "3"),
      "r_words": ("words in the reply (textstat.lexicon_count, punctuation removed)", "5d"),
      "r_pronoun_you_rate": ("spaCy personal pronouns with Person=2, divided by the number of alphabetic tokens", "3, 5d"),
      "r_pronoun_i_rate": ("spaCy personal pronouns with Person=1 and not plural, divided by alphabetic tokens", "3, 5d"),
      "r_pronoun_we_rate": ("spaCy personal pronouns with Person=1 and Number=Plur, divided by alphabetic tokens", "3, 5d"),
      "r_vader_compound": ("VADER compound score of the reply", "5d"), "r_tb_subjectivity": ("TextBlob subjectivity of the reply", "5d"),
      "r_flesch_kincaid": ("Flesch-Kincaid grade (textstat)", "5d"),
      "r_n_imperatives": ("sentences whose root is a base-form verb (tag VB) with no subject", "5d"),
      "r_n_resources": ("internal links + external links", "3, 5d"), "r_n_wikilink": ("internal links (not to users) in the reply HTML", "3, 5d"),
      "r_n_policy": ("links whose final target (after redirects) is in Category:Wikipedia policies or guidelines", "3, 5d"),
      "r_n_help": ("links whose final target is in the Help: namespace", "3, 5d"),
      "r_reply_lag_hours": ("first reply time minus question time, in hours", "3, 5d"),
      "r_reply_q_word_ratio": ("r_words divided by the question word count", "3, 5d"),
      "r_has_greeting": ("ConvoKit Indirect_(greeting)", "3, 5d"), "r_polite_gratitude": ("ConvoKit Gratitude", "3, 5d"),
      "r_polite_has_positive": ("ConvoKit HASPOSITIVE", "3, 5d"), "r_polite_deference": ("ConvoKit Deference", "3, 5d"),
      "r_has_list": ("the reply HTML has a list item", "3, 5d"), "r_n_question_marks": ("number of '?' in the reply text", "3, 5d"),
      "r_qr_cosine_sim": ("cosine similarity of the question and reply embeddings", "3, 5d, 5f")}
display(pd.DataFrame([{"table": "question_features", "column": c, "meaning": m, "checked in": w} for c, (m, w) in QD.items()]))
display(pd.DataFrame([{"table": "reply_features", "column": c, "meaning": m, "checked in": w} for c, (m, w) in RD.items()]))
print("""Embedding files: emb_q.npy / emb_r.npy are float32 matrices; row i belongs to the text whose sha256 (after whitespace
normalization) is row i of emb_q_index.parquet / emb_r_index.parquet (columns text_sha, source = legacy or api).
emb_q_sample.npy has one row per row of question_features.parquet (NaN rows where the embedding is missing). Checked in 3, 5f.
coverage.json counts questions, labels, scores, embeddings and cosine values. Checked in 3.""")
'''),
code(r'''
print("question_features columns missing from the dictionary:", [c for c in qf.columns if c not in QD], "| extra:", [c for c in QD if c not in qf.columns])
print("reply_features columns missing from the dictionary:", [c for c in rf.columns if c not in RD], "| extra:", [c for c in RD if c not in rf.columns])
'''),
md("""
## 3. Invariants

Several checks recompute a column from the raw data for **every** row: the simple length features, the reply text
rebuilt from the thread turns (question marks, lists, links), the reply lag, and the word ratio.
"""),
code(r'''
ck = Checks()
pol_cols = [c for c in qf.columns if "politeness" in c]
llm_cols = list(CODEBOOK)
persp_cols = [c for c in qf.columns if c.startswith("q_persp_")]
print("Rows and keys")
ck.add("one row per sampled question (same revids as the sample, no duplicates)",
       set(qf.first_question_revid) == set(S.index) and qf.first_question_revid.is_unique)
ck.add("reply rows are treated questions only, no duplicates",
       set(rf.first_question_revid) <= set(S[S.treat == 1].index) and rf.first_question_revid.is_unique)
REP = {rv: my_first_reply(TURNS.get(rv, []), S.at[rv, "q_ts"]) for rv in S[S.treat == 1].index}
with_text = {rv for rv, x in REP.items() if x[1]}
ck.add(f"reply rows == treated questions whose rebuilt reply text is not empty ({len(with_text)})", set(rf.first_question_revid) == with_text)

print("\nQuestion features")
ck.add("format flags and politeness strategies are 0/1",
       bool(qf[["q_has_sig", "q_has_unsigned", "q_has_url", "q_is_indented"] + pol_cols].isin([0, 1]).all().all()))
ck.add("counts are whole numbers >= 0",
       bool(all((qf[c] >= 0).all() and (qf[c] % 1 == 0).all() for c in ["q_link_count", "q_template_count", "q_mentions_wp", "q_mentions_help", "q_body_char_len", "q_body_word_count"])))
ck.add("q_mentions_wp + q_mentions_help <= q_link_count", bool((qf.q_mentions_wp + qf.q_mentions_help <= qf.q_link_count).all()))
ck.add("q_body_char_len == len(question_text) for every row", bool((qf.q_body_char_len == qf.first_question_revid.map(txt.fillna("").str.len())).all()))
ck.add("q_body_word_count == number of whitespace words for every row", bool((qf.q_body_word_count == qf.first_question_revid.map(txt.fillna("").str.split().str.len())).all()))
H = pd.DataFrame([my_question_format(WT[rv], txt.get(rv) or "") for rv in qf.first_question_revid], index=qf.index)
for c in ("q_has_sig", "q_has_url", "q_is_indented", "q_mentions_wp", "q_mentions_help"):
    ck.add(f"{c} == simple regex recomputation for every row ({int((H[c] != qf[c]).sum())} differ)", bool((H[c] == qf[c]).all()))
for c in ("q_link_count", "q_template_count", "q_has_unsigned"):
    n = int((H[c] != qf[c]).sum())
    ck.add(f"{c} == simple regex recomputation for at least 99.5% of rows ({n} differ; listed in section 6)", n <= 0.005 * len(qf))
v = qf.q_vader_neg + qf.q_vader_neu + qf.q_vader_pos
ck.add("VADER neg + neu + pos == 1 (or all 0 for empty text)", bool((((v - 1).abs() < 0.002) | (v == 0)).all()))
ck.add("VADER compound and TextBlob polarity in [-1, 1]; subjectivity in [0, 1]",
       bool(qf.q_vader_compound.between(-1, 1).all() and qf.q_tb_polarity.between(-1, 1).all() and qf.q_tb_subjectivity.between(0, 1).all()))

print("\nLLM labels, Perspective, embeddings")
ck.add("LLM labels are 0, 1, or missing", bool(qf[llm_cols].stack().isin([0, 1]).all()))
ck.add("has_llm <=> all five labels present; otherwise all five missing",
       bool((qf[llm_cols].notna().all(axis=1) == qf.has_llm).all() and (qf[llm_cols].isna().all(axis=1) == ~qf.has_llm).all()))
viol = qf[(qf.q_substantive == 0) & (qf[llm_cols[1:]].sum(axis=1) > 0)]
ck.add(f"codebook rule: if Q0 = N then Q2 to Q5 are N ({len(viol)} violations; listed in section 6)", len(viol) == 0)
bad_persp = qf[qf.has_persp & qf[persp_cols].isna().all(axis=1)]
ck.add(f"has_persp <=> all eight scores present ({len(bad_persp)} rows have has_persp = True but no score; listed in section 6)", bool((qf[persp_cols].notna().all(axis=1) == qf.has_persp).all() and (qf[persp_cols].isna().all(axis=1) == ~qf.has_persp).all()))
ck.add("Perspective scores in [0, 1]", bool(qf[persp_cols].stack().between(0, 1).all()))
EQS = np.load(M6 / "emb_q_sample.npy", mmap_mode="r")
ck.add("emb_q_sample.npy has one row per question row, 1024 columns", EQS.shape == (len(qf), 1024))
ck.add("has_emb_q <=> that row is finite", bool((np.isfinite(EQS).all(axis=1) == qf.has_emb_q.to_numpy()).all()))

print("\nReply features")
r_q = rf.first_question_revid
ck.add("r_n_question_marks == count of '?' in the rebuilt reply text (all rows)", bool((rf.r_n_question_marks == r_q.map(lambda rv: REP[rv][1].count("?"))).all()))
ck.add("r_has_list == rebuilt reply has a list item (all rows)", bool((rf.r_has_list == r_q.map(lambda rv: int(REP[rv][2]["n_list_items"] > 0))).all()))
ck.add("r_n_wikilink == number of wikilinks in the rebuilt reply (all rows)", bool((rf.r_n_wikilink == r_q.map(lambda rv: len(REP[rv][2]["wikilinks"]))).all()))
ck.add("r_n_resources == wikilinks + external links (all rows)", bool((rf.r_n_resources == r_q.map(lambda rv: len(REP[rv][2]["wikilinks"]) + REP[rv][2]["n_external"])).all()))
ck.add("r_n_policy <= r_n_wikilink and r_n_help <= r_n_wikilink", bool((rf.r_n_policy <= rf.r_n_wikilink).all() and (rf.r_n_help <= rf.r_n_wikilink).all()))
lag_h = r_q.map(lambda rv: (ts(TURNS[rv][REP[rv][0]]["ts"]) - ts(S.at[rv, "q_ts"])).total_seconds() / 3600)
ck.add("r_reply_lag_hours == (first reply turn time - q_ts) in hours (all rows)", bool(np.allclose(lag_h, rf.r_reply_lag_hours, atol=1e-9)))
ck.add("r_reply_q_word_ratio == r_words / max(question words, 1) (all rows)",
       bool(np.allclose(rf.r_reply_q_word_ratio, rf.r_words / r_q.map(QI.q_body_word_count).clip(lower=1), atol=1e-9)))
ck.add("pronoun rates in [0, 1] and their sum <= 1", bool(rf[["r_pronoun_you_rate", "r_pronoun_i_rate", "r_pronoun_we_rate"]].stack().between(0, 1).all()
       and (rf[["r_pronoun_you_rate", "r_pronoun_i_rate", "r_pronoun_we_rate"]].sum(axis=1) <= 1).all()))
ck.add("reply politeness flags and r_has_list are 0/1", bool(rf[["r_has_greeting", "r_polite_gratitude", "r_polite_has_positive", "r_polite_deference", "r_has_list"]].isin([0, 1]).all().all()))
ck.add("r_words and r_n_imperatives are whole numbers >= 0", bool(((rf.r_words >= 0) & (rf.r_words % 1 == 0) & (rf.r_n_imperatives >= 0)).all()))
ck.add("cosine similarity in [-1, 1]", bool(rf.r_qr_cosine_sim.dropna().between(-1, 1).all()))
ck.add("cosine is missing only when the question embedding is missing or the reply embedding is missing",
       bool(rf[rf.r_qr_cosine_sim.isna()].first_question_revid.map(QI.has_emb_q).eq(False).sum() <= rf.r_qr_cosine_sim.isna().sum()))

print("\ncoverage.json")
hand_cov = {"questions": len(qf), "llm": int(qf.has_llm.sum()), "persp": int(qf.has_persp.sum()), "emb_q": int(qf.has_emb_q.sum()),
            "replies": len(rf), "qr_cosine": int(rf.r_qr_cosine_sim.notna().sum())}
ck.add("coverage.json counts == counts recomputed from the tables", all(hand_cov[k] == cov[k] for k in hand_cov))
ck.report()
'''),
md("""
## 4. Comparison with the paper version

For the same question (matched by revid): do the paper version and the new version give the same question types,
and how strongly do the old features correlate with the new ones (which now use standard tools)?
"""),
code(r'''
z = np.load(config.EXTERNAL["paper_npz"], allow_pickle=True)
s7 = {int(r["conversation_id"]): r["revid"] for r in iter_jsonl(config.EXTERNAL["paper_s7"])}
rev = [int(s7[int(c)]) if s7.get(int(c)) else -1 for c in z["cid_order"]]
P = pd.DataFrame(z["X_Qtext"], columns=[str(c) for c in z["Qtext_cols"]]).assign(first_question_revid=rev)
P[[str(c) for c in z["Qtype_cols"]]] = z["X_Qtype"]
m = qf.merge(P, on="first_question_revid", suffixes=("", "_paper"))
print("questions in both versions:", len(m))
rows = []
for c in [str(c) for c in z["Qtype_cols"]] + ["q_link_count", "q_body_word_count", "q_vader_compound",
          "q_poly_feature_politeness_==Gratitude==", "q_poly_feature_politeness_==Please==", "q_mentions_wp"]:
    a, b = m[c], m[f"{c}_paper"]
    ok = a.notna()
    binary_ = set(pd.unique(b)) <= {0, 1}
    rows.append({"feature": c, "measure": "share equal" if binary_ else "correlation",
                 "value": round(float((a[ok] == b[ok]).mean() if binary_ else np.corrcoef(a[ok], b[ok])[0, 1]), 4)})
display(pd.DataFrame(rows))
'''),
md("""
## 5. Tracer and random samples: raw text next to every feature

### 5a. Random questions: raw wikitext, plain text, and all 37 text features

For each question: the simple features recomputed by `my_question_format` (regex, no mwparserfromhell), VADER and
TextBlob recomputed by calling the libraries on the plain text, and the 21 ConvoKit politeness strategies
recomputed by calling `PolitenessStrategies` directly, with the words (markers) that triggered each one. The
tracer questions from `pipeline/common/tracer.py` that are in the sample are shown first.
"""),
code(r'''
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy, textstat
from convokit import PolitenessStrategies
VADER = SentimentIntensityAnalyzer()
NLP = spacy.load("en_core_web_sm")
PS = PolitenessStrategies(verbose=0)
KEYS = list(POLITE)

def polite(text):
    """0/1 per strategy and the marker words, straight from ConvoKit."""
    if not (text or "").strip():
        return {k: 0 for k in KEYS}, {k: [] for k in KEYS}
    u = PS.transform_utterance(text, spacy_nlp=NLP, markers=True)
    st, mk = u.meta["politeness_strategies"], u.meta["politeness_markers"]
    flags = {k: int(st.get(f"feature_politeness_=={k}==", 0) > 0) for k in KEYS}
    words = {k: [w for grp in mk.get(f"politeness_markers_=={k}==", []) for (w, _, _) in grp] for k in KEYS}
    return flags, words

def show_question(rv):
    row = QI.loc[rv]
    plain = txt.get(rv) or ""
    print("=" * 110)
    print(f"question revid {rv}  {oldid(rv)}  q_ts {S.at[rv, 'q_ts']}  asker {S.at[rv, 'mentee_name']}")
    print(f"RAW WIKITEXT: {WT[rv][:700]!r}")
    print(f"PLAIN TEXT:   {plain[:700]!r}")
    h = my_question_format(WT[rv], plain)
    w = AWAY_NOTE_RE.sub("", WT[rv], count=1); mm = SIG.search(w)
    body = w[:mm.start()] if mm else w
    print(f"  evidence: signature timestamp = {mm.group(0) if mm else None!r}; links in body = {link_titles(body)}")
    compare(h, {c: row[c] for c in FORMAT_COLS}, title="  wiki format features (regex recomputation):")
    vd, tb = VADER.polarity_scores(plain) if plain.strip() else {"neg": 0, "neu": 0, "pos": 0, "compound": 0}, TextBlob(plain).sentiment
    compare({"q_vader_neg": vd["neg"], "q_vader_neu": vd["neu"], "q_vader_pos": vd["pos"], "q_vader_compound": vd["compound"],
             "q_tb_polarity": tb.polarity, "q_tb_subjectivity": tb.subjectivity},
            {c: row[c] for c in ("q_vader_neg", "q_vader_neu", "q_vader_pos", "q_vader_compound", "q_tb_polarity", "q_tb_subjectivity")},
            title="  sentiment (VADER and TextBlob called directly):")
    flags, words = polite(plain)
    print("  politeness strategies (ConvoKit called directly); 'markers' are the words that triggered the strategy:")
    display(pd.DataFrame([{"strategy": k, "recomputed": flags[k], "in data": int(row[f"q_poly_feature_politeness_=={k}=="]),
                           "match": flags[k] == int(row[f"q_poly_feature_politeness_=={k}=="]), "markers": ", ".join(words[k])} for k in KEYS]))
    ok = all(flags[k] == int(row[f"q_poly_feature_politeness_=={k}=="]) for k in KEYS)
    print("  -> all fields match" if ok else "  -> MISMATCH in politeness strategies")

from pipeline.common.tracer import tracer_frame
tr = tracer_frame()
tracer_q = [int(x) for x in tr.revid.dropna() if int(x) in QI.index]
print("TRACER QUESTIONS IN THE SAMPLE:", tracer_q)
for rv in tracer_q:
    show_question(rv)
print("\nRANDOM QUESTIONS")
for rv in pick(qf, N_RANDOM, SEED).first_question_revid:
    show_question(int(rv))
'''),
md("""
### 5b. LLM question types: text, labels, and the codebook definition

The labels are judgements, so they cannot be recomputed; the reader compares each text with the one-line codebook
definition. The full codebook is `services.codebook_text()` (paper `codebook_prompt.md` plus 120 labelled
examples). The rule "if Q0 = N then Q2 to Q5 are N" is checked on all rows in section 3.
"""),
code(r'''
for c, d in CODEBOOK.items():
    print(f"  {c:14s} {d}")
print("\nRANDOM QUESTIONS WITH ALL FIVE LABELS")
for rv in pick(qf[qf.has_llm], N_RANDOM, SEED).first_question_revid:
    lab = {c: int(QI.at[rv, c]) for c in CODEBOOK}
    print(f"\n  revid {rv} {oldid(rv)}\n  text: {str(txt.get(rv))[:500]!r}")
    print("  labels:", "  ".join(f"{c}={'Y' if v else 'N'}" for c, v in lab.items()))
'''),
code(r'''
for c, d in CODEBOOK.items():
    print(f"\n{'=' * 110}\n## {c}: {d}")
    pos = qf[qf[c] == 1]
    neg = qf[(qf[c] == 0) & ((qf.q_substantive == 1) if c != "q_substantive" else True)]
    print(f"   share labelled Y: {qf[c].mean():.3f}. Negatives below are drawn among questions with Q0 = Y, where the label is a real decision.")
    for name, g in (("Y", pos), ("N", neg)):
        for rv in pick(g, N_RANDOM, SEED).first_question_revid:
            print(f"   {c}={name}: {str(txt.get(rv, ''))[:220]!r}")
'''),
md("""
### 5c. Perspective scores: the highest and lowest questions for each attribute

The input to Perspective is the question text with user names removed and links replaced by `[LINK]`. High
scores should look rude, profane, sexual, and so on; low scores should look harmless. `source` says where the score
came from: `legacy` = copied from the paper version (`s10_perspective_mentee.csv`, matched by revid and text
similarity), `api` = fetched again by M6.

**Known problem (found with this notebook, 2026-10-26).** Several `legacy` scores do not fit their text, for
example "How to write an essay" (revid 1278478330) has TOXICITY 0.854 and "Hi! Thank you so much!" (revid
1334684029) has 0.543. Sending these texts to the Perspective API again gives 0.017 and 0.009. Of the 46 legacy
questions with TOXICITY > 0.5, 30 get a fresh score below 0.3. The paper-version CSV seems to hold scores of other
texts for part of its rows. Set `RECHECK_ONLINE = True` in the second cell below to repeat the comparison.
"""),
code(r'''
import sqlite3
from pipeline.m6_text.cache import DB as CACHE_DB
with sqlite3.connect(CACHE_DB) as con:
    PSRC = dict(con.execute("select text_sha, source from results where task = 'perspective_q'").fetchall())
persp_source = lambda rv: PSRC.get(text_sha(make_emb_text(WT[rv], names_in(rv))))   # noqa: E731
for c in persp_cols:
    g = qf[qf[c].notna()]
    print(f"\n## {c} (mean {g[c].mean():.3f})")
    for label, part in (("highest", g.nlargest(3, c)), ("lowest", g.nsmallest(2, c))):
        for r in part.itertuples():
            print(f"   {label} {getattr(r, c):.3f} [{persp_source(r.first_question_revid)}]: {str(txt.get(r.first_question_revid, ''))[:150]!r}")
top = qf.nlargest(200, "q_persp_toxicity").first_question_revid
print("\nsource of the 200 highest TOXICITY scores:", pd.Series([persp_source(rv) for rv in top]).value_counts().to_dict(),
      "| source of all scores:", pd.Series(list(PSRC.values())).value_counts().to_dict())
'''),
code(r'''
RECHECK_ONLINE = False   # True: send the texts below to the Perspective API again (needs PERSPECTIVE_API_KEY in .env; about 1 s per text)
if RECHECK_ONLINE:
    import logging
    from pipeline.m6_text import services
    picks = [int(x) for x in qf.nlargest(N_RANDOM, "q_persp_toxicity").first_question_revid] + \
            [int(x) for x in pick(qf[qf.q_persp_toxicity.notna()], N_RANDOM, SEED).first_question_revid]
    texts = {rv: make_emb_text(WT[rv], names_in(rv)) for rv in picks}
    fresh = services.perspective(list(texts.values()), logging.getLogger())
    for rv, t_ in texts.items():
        if t_ in fresh:
            print(f"\nrevid {rv} [{persp_source(rv)}] {t_[:120]!r}")
            compare({c: round(fresh[t_][c.replace("q_persp_", "").upper()], 2) for c in persp_cols}, {c: round(float(QI.at[rv, c]), 2) for c in persp_cols})
else:
    print("RECHECK_ONLINE is False: no texts were sent to the Perspective API.")
'''),
md("""
### 5d. Random replies: text, pronouns, imperatives, links, and all 21 reply features

For each reply: the thread turns that make up the reply (rebuilt by `my_first_reply`), the spaCy tokens counted as
*you*, *I* and *we*, the sentences counted as imperatives, and every link with its namespace, the target after
redirects, and the policy and Help decisions (from the cached API answers, `wikimeta.resolve_titles` and
`policy_status`). Then every reply feature is recomputed and compared. The cosine similarity is recomputed from the
embedding files (see 5f for the method).
"""),
code(r'''
from pipeline.common.api import MediaWikiClient
from pipeline.common.wikimeta import resolve_titles, policy_status
CLIENT = MediaWikiClient(online=False)
EQ, ER = np.load(M6 / "emb_q.npy", mmap_mode="r"), np.load(M6 / "emb_r.npy", mmap_mode="r")
QROW = {sha: i for i, sha in enumerate(pd.read_parquet(M6 / "emb_q_index.parquet").text_sha)}
RROW = {sha: i for i, sha in enumerate(pd.read_parquet(M6 / "emb_r_index.parquet").text_sha)}
# Redirects and policy categories come from cached API answers. The cache key is the exact batch of 50 titles, so we
# ask for the same full set of titles that M6 asked for (every wikilink in every reply); only cached answers are used.
ALL_TITLES = {l["title"] for rv in with_text for l in REP[rv][2]["wikilinks"]}
RES = resolve_titles(ALL_TITLES, CLIENT)
POL_STATUS = policy_status({v["final"] for v in RES.values() if v.get("ns") == 4}, CLIENT)
POL = {t for t, v in POL_STATUS.items() if v}
print(f"link titles {len(ALL_TITLES)}; resolved {len(RES)}; policy or guideline targets {len(POL)}")

def my_cosine(rv, reply_text):
    qv = QROW.get(text_sha(make_emb_text(WT[rv], names_in(rv))))
    rv_ = RROW.get(text_sha(emb_text_from_plain(reply_text, names_in(rv))))
    if qv is None or rv_ is None:
        return np.nan
    a, b = np.asarray(EQ[qv], dtype=np.float64), np.asarray(ER[rv_], dtype=np.float64)
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 6)   # vectors are float32, so compare 6 decimals

def show_reply(rv):
    row = RI.loc[rv]
    i, text, links, idx = REP[rv]
    print("=" * 110)
    print(f"question revid {rv}  {oldid(rv)}  q_ts {S.at[rv, 'q_ts']}  question words {QI.at[rv, 'q_body_word_count']}")
    print(f"QUESTION: {str(txt.get(rv))[:250]!r}")
    print(f"reply = turns {idx} by {TURNS[rv][i]['author']} ({TURNS[rv][i]['role']}), first at {TURNS[rv][i]['ts']}")
    print(f"REPLY TEXT: {text[:900]!r}")
    doc = NLP(text)
    alpha = [t for t in doc if t.is_alpha]
    prs = [t for t in doc if "Prs" in t.morph.get("PronType")]
    you = [t.text for t in prs if "2" in t.morph.get("Person")]
    we = [t.text for t in prs if "1" in t.morph.get("Person") and "Plur" in t.morph.get("Number")]
    me = [t.text for t in prs if "1" in t.morph.get("Person") and "Plur" not in t.morph.get("Number")]
    print(f"  alphabetic tokens: {len(alpha)} | you-tokens {you} | I-tokens {me} | we-tokens {we}")
    imps = [sn.text.strip()[:100] for sn in doc.sents if sn.root.tag_ == "VB" and sn.root.pos_ in ("VERB", "AUX")
            and not any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in sn.root.children)]
    print(f"  imperative sentences ({len(imps)}): {imps}")
    res, pol = RES, POL
    lt = pd.DataFrame([{"title": l["title"], "ns in HTML": l["ns"], "target after redirects": res.get(l["title"], {}).get("final", l["title"]),
                        "target ns": res.get(l["title"], {}).get("ns", l["ns"]),
                        "policy or guideline": res.get(l["title"], {}).get("final", l["title"]) in pol,
                        "category": POL_STATUS.get(res.get(l["title"], {}).get("final", l["title"])),
                        "help page": res.get(l["title"], {}).get("ns", l["ns"]) == 12} for l in links["wikilinks"]])
    print(f"  wikilinks ({len(lt)}), external links {links['n_external']}, list items {links['n_list_items']}:")
    if len(lt): display(lt)
    flags, words = polite(text)
    print(f"  politeness markers: greeting {words['Indirect_(greeting)']}, gratitude {words['Gratitude']}, positive {words['HASPOSITIVE']}, deference {words['Deference']}")
    nw = textstat.lexicon_count(text, removepunct=True)
    vd = VADER.polarity_scores(text)
    hand = {"r_words": nw, "r_pronoun_you_rate": len(you) / max(len(alpha), 1), "r_pronoun_i_rate": len(me) / max(len(alpha), 1),
            "r_pronoun_we_rate": len(we) / max(len(alpha), 1), "r_vader_compound": vd["compound"],
            "r_tb_subjectivity": TextBlob(text).sentiment.subjectivity,
            "r_flesch_kincaid": textstat.flesch_kincaid_grade(text) if nw else 0.0, "r_n_imperatives": len(imps),
            "r_n_resources": len(lt) + links["n_external"], "r_n_wikilink": len(lt),
            "r_n_policy": int(lt["policy or guideline"].sum()) if len(lt) else 0, "r_n_help": int(lt["help page"].sum()) if len(lt) else 0,
            "r_reply_lag_hours": (ts(TURNS[rv][i]["ts"]) - ts(S.at[rv, "q_ts"])).total_seconds() / 3600,
            "r_reply_q_word_ratio": nw / max(len((txt.get(rv) or "").split()), 1),
            "r_has_greeting": flags["Indirect_(greeting)"], "r_polite_gratitude": flags["Gratitude"],
            "r_polite_has_positive": flags["HASPOSITIVE"], "r_polite_deference": flags["Deference"],
            "r_has_list": int(links["n_list_items"] > 0), "r_n_question_marks": text.count("?"),
            "r_qr_cosine_sim": my_cosine(rv, text)}
    print(f"  (simple word count for comparison: {len(re.findall(r'[A-Za-z0-9]+', text))} alphanumeric runs)")
    data = {c: row[c] for c in hand}
    data["r_qr_cosine_sim"] = round(float(data["r_qr_cosine_sim"]), 6) if pd.notna(data["r_qr_cosine_sim"]) else np.nan
    compare(hand, data, title="  all reply features (cosine rounded to 6 decimals because the vectors are float32):")

for rv in pick(rf, N_RANDOM, SEED).first_question_revid:
    show_reply(int(rv))
'''),
md("""
### 5e. Replies with policy, Help, or many links

Random replies with at least one policy or guideline link, so the reader can see the link decisions on rows where
they matter.
"""),
code(r'''
for rv in pick(rf[rf.r_n_policy > 0], 2, SEED).first_question_revid:
    show_reply(int(rv))
'''),
md("""
### 5f. Cosine similarity and embedding files

For N_RANDOM random replies, the question text for embeddings is rebuilt (`make_emb_text`: plain text with user
names removed and links replaced), the reply text likewise (`emb_text_from_plain`); each text's sha256 gives its
row in `emb_q.npy` / `emb_r.npy` through the index files. The cosine is computed with numpy and compared to 6 decimals
(the vectors are stored as float32, so later digits depend on the order of the arithmetic). We also
check that the question vector equals the row of `emb_q_sample.npy` that belongs to the same question.
"""),
code(r'''
pos = {rv: i for i, rv in enumerate(qf.first_question_revid)}
for rv in pick(rf[rf.r_qr_cosine_sim.notna()], N_RANDOM, SEED).first_question_revid:
    rv = int(rv)
    qtext = make_emb_text(WT[rv], names_in(rv)); rtext = emb_text_from_plain(REP[rv][1], names_in(rv))
    qi, ri = QROW.get(text_sha(qtext)), RROW.get(text_sha(rtext))
    print(f"\nrevid {rv}: question row {qi} in emb_q.npy, reply row {ri} in emb_r.npy")
    print(f"   question text for embedding: {qtext[:150]!r}\n   reply text for embedding:    {rtext[:150]!r}")
    same = bool(np.allclose(np.asarray(EQ[qi]), np.asarray(EQS[pos[rv]]))) if qi is not None else None
    compare({"r_qr_cosine_sim": my_cosine(rv, REP[rv][1]), "emb_q row == emb_q_sample row": True},
            {"r_qr_cosine_sim": round(float(RI.at[rv, "r_qr_cosine_sim"]), 6), "emb_q row == emb_q_sample row": same})
'''),
md("## 6. Edge-case lists"),
code(r'''
for name, flag in (("LLM labels", "has_llm"), ("Perspective scores", "has_persp"), ("question embedding", "has_emb_q")):
    miss = qf[~qf[flag]]
    print(f"\n{name} missing: {len(miss)}")
    for rv in miss.first_question_revid.head(8):
        print(f"   {rv}  {str(txt.get(rv, ''))[:110]!r}")
print(f"\nhas_persp = True but every Perspective score is missing: {len(bad_persp)} (the stored result is all null; these")
print("are counted as covered in coverage.json although they have no score):")
for rv in bad_persp.first_question_revid:
    print(f"   {rv}  {str(txt.get(rv, ''))[:110]!r}")
miss_c = rf[rf.r_qr_cosine_sim.isna()]
print(f"\nReplies without a cosine similarity: {len(miss_c)} (question embedding missing: {int((~miss_c.first_question_revid.map(QI.has_emb_q)).sum())})")
for rv in miss_c.first_question_revid.head(5):
    print(f"   {rv}  reply: {REP[rv][1][:100]!r}")
empty = sorted(set(S[S.treat == 1].index) - with_text)
print(f"\nTreated questions whose first reply has no text (no reply row): {len(empty)}")
for rv in empty[:5]:
    i = REP[rv][0]
    print(f"   {rv}  first reply turn {i}: {TURNS[rv][i] if i is not None else None}")
print(f"\nLLM labels that break the codebook rule 'Q0 = N implies Q2 to Q5 = N': {len(viol)}")
display(viol[["first_question_revid"] + llm_cols].head(10).assign(text=lambda d: d.first_question_revid.map(lambda rv: str(txt.get(rv))[:90])))
print("\nRows where the simple regex and mwparserfromhell disagree (q_link_count, q_template_count, q_has_unsigned).")
print("Typical reasons: a link written as [[https://...]] (an external link, not a wikilink), a template inside an HTML tag")
print("(not top level), a broken template without its closing braces, or a template whose name is itself a link.")
for c in ("q_link_count", "q_template_count", "q_has_unsigned"):
    d = qf[H[c] != qf[c]]
    print(f"\n  {c}: {len(d)} rows")
    for rv in d.first_question_revid.head(4):
        print(f"     {rv}: regex {H.loc[qf.first_question_revid == rv, c].iloc[0]}, data {QI.at[rv, c]}; wikitext {WT[rv][:160]!r}")
print("\nLongest replies (words):")
display(rf.nlargest(3, "r_words")[["first_question_revid", "r_words", "r_n_wikilink", "r_n_policy", "r_flesch_kincaid"]])
print("Shortest replies (words):")
for rv in rf.nsmallest(3, "r_words").first_question_revid:
    print(f"   {rv}: r_words={RI.at[rv, 'r_words']} text {REP[rv][1][:100]!r}")
'''),
md("## 7. Distributions: agreement (paper Table 3; I-54), label shares, cosine similarity"),
code(r'''
ag = pd.read_csv(config.RESULTS / "m6" / "agreement.csv")
display(ag[ag.set.str.startswith("test")].pivot(index="pair", columns="dim", values="kappa").round(3))
p = config.RESULTS / "m6" / "llm_reproducibility.csv"
if p.exists(): display(pd.read_csv(p).round(3))
print("\nshare labelled Y:", qf[llm_cols].mean().round(3).to_dict())
print("\nquestion features (summary):")
display(qf[FORMAT_COLS + ["q_vader_compound", "q_tb_polarity"]].describe().T.round(3))
print("\nreply features (summary):")
display(rf.drop(columns="first_question_revid").describe().T.round(3))
RT = rf.set_index("first_question_revid")
print("\nhighest and lowest question-reply cosine similarity:")
for rv in list(rf.nlargest(3, "r_qr_cosine_sim").first_question_revid) + list(rf.nsmallest(3, "r_qr_cosine_sim").first_question_revid):
    print(f"   {RT.r_qr_cosine_sim[rv]:.3f}  Q: {str(txt.get(rv, ''))[:80]!r}  R: {REP[rv][1][:80]!r}")
'''),
md("## 8. Summary (results/m6_report.json)"),
code(r'''
cov = json.loads((M6 / "coverage.json").read_text())
(config.RESULTS / "m6_report.json").write_text(json.dumps(cov, indent=1))
print(json.dumps(cov, indent=1))
print(f"invariants: {sum(ok for _, ok in ck.items)} / {len(ck.items)} passed")
'''),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "m6_check.ipynb"
    write_notebook(CELLS, out)
    print("wrote", out)
    if "--run" in sys.argv:
        run_as_script(out, cwd=out.parent.parent)
        print("\nnotebook ran end to end")


if __name__ == "__main__":
    main()
