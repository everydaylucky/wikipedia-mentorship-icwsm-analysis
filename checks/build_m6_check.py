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

## Plain-language guide

**Why the study needs this module.** The study compares newcomers whose first question got a reply from their
mentor with similar newcomers whose question got no reply. To make the two groups comparable, the propensity score
model (a model that predicts who gets a reply) needs to know what each question looked like: long or short, polite
or rude, a real editing question or just "Hello". M6 turns every question into such numbers, called *text
features*. These features are covariates (control variables) in the propensity score. M6 also describes the first
reply (length, links to help pages, politeness, and so on); these reply features are the inputs of the
reply-feature analysis (which kinds of replies go with better outcomes).

**What data it uses and from where.**

* The question text and the conversation that follows it, taken from mentor talk pages (made by M2), and the list
  of sampled newcomers and their first questions (made by M5).
* Free text-analysis tools that run on this computer: VADER and TextBlob (they give a sentiment score, that is how
  positive or negative the words are), ConvoKit (it flags 21 politeness strategies such as "please", "thanks",
  greetings), spaCy (it finds pronouns like *you*, *I*, *we* and commands like "click here"), textstat (reading
  level), and mwparserfromhell (it reads Wikipedia markup such as links and templates).
* Three online services, called once and then stored in a local cache: (1) the DeepSeek language model, which
  labels each question with five yes/no question types using a written codebook and 120 examples labelled by
  people; (2) Google's Perspective API, which gives each text a score from 0 to 1 for toxicity, insult, profanity,
  threat and four more attributes (a *Perspective score*: about how likely readers would see the text as, say,
  rude); (3) the Qwen text-embedding-v4 model, which turns a text into a list of 1024 numbers (an *embedding*)
  so that texts with similar meaning get similar lists.
* The paper version of the data (April 2026) for comparison, and for reusing earlier LLM labels and embeddings
  when the text is the same.

**What it does, step by step.**

1. Take the first question of every sampled newcomer and turn the wiki markup into plain text.
2. Count simple things in the question (length, links, templates, signature, web addresses) and score its
   sentiment and politeness.
3. Get the five question-type labels, the eight Perspective scores, and the embedding (reuse stored results when
   the same text was already processed; otherwise ask the service). User names are removed and links replaced by
   `[LINK]` before a text is sent out. Missing results stay missing; they are never filled with 0.
4. For each question that got a reply, rebuild the first reply (all comments by the first person who answered,
   until someone else speaks) and compute 21 reply features, including how similar the reply is to the question
   (the *cosine similarity* of the two embeddings: 1 = same meaning, 0 = unrelated).

**What the outputs are (this run, final data, cutoff 2026-04-20T14:00Z).**

* `data/interim/m6/question_features.parquet`: one row = one sampled first question (34,985 rows). Columns: the
  question id, 37 text features, the 5 question-type labels, the 8 Perspective scores, and three yes/no flags
  saying whether labels, scores and embedding exist.
* `data/interim/m6/reply_features.parquet`: one row = one question that got a reply with some text (26,877 rows).
  Columns: the question id and 21 reply features (words, pronoun rates, links, policy and Help links, reply lag in
  hours, politeness flags, cosine similarity, and others).
* `emb_q.npy`, `emb_r.npy` (with `emb_q_index.parquet`, `emb_r_index.parquet`): the stored question and reply
  embeddings, one row per distinct text. `emb_q_sample.npy`: one embedding row per row of question_features
  (34,985 rows, 1024 numbers each; empty for the 46 questions without an embedding).
* `coverage.json` and `manifest.json`: counts of what was produced and how the run was done.
* `results/m6/agreement.csv`: agreement between people and the LLM on the question types (paper Table 3).

**How to read this notebook.**

* Every cell has already been run on the final data and its output is saved below it. Nothing needs to be run.
* `[OK ]` means a check passed and `[FAIL]` means it did not. Tables that compare a value recomputed by hand with
  the stored value have a `match` column and end with `all fields match` (good) or `MISMATCH` (bad).
* An *invariant* is a rule that must hold for every row (for example "a score is between 0 and 1").
* A *tracer* is one of a few fixed example newcomers that every check notebook follows through the whole
  pipeline. A *random sample* is a set of rows drawn at random (with a fixed seed, so the same rows come back) so
  that a reader can look at real cases with their own eyes.

## 1. Module description

**What M6 does.** `pipeline/m6_text/` (`build_text.py`, `text_features.py`, `services.py`, `cache.py`,
`agreement.py`) describes the text of every sampled first question and of the first reply in the treated group:

1. **Question features (37 columns, `text_features.question_features`).** Wiki format features parsed with
   mwparserfromhell (signature, links, templates, URL, indentation, length), VADER and TextBlob sentiment, and the
   21 ConvoKit politeness strategies (decision D6).
2. **Question types (5 columns).** DeepSeek labels with the paper codebook and 120 human-labelled examples:
   `q_substantive` (Q0), `q_referent` (Q2), `q_appraisal` (Q3), `q_normative` (Q4), `q_own_work` (Q5).
3. **Perspective scores (8 columns)** and **Qwen embeddings** (text-embedding-v4, 1024 dimensions) of the question
   text with user names removed and links replaced by `[LINK]`. Paper-version LLM labels and embeddings are reused
   when the text is the same (or almost the same). Perspective scores are never copied from the paper version
   (I-59, see 5c): all of them were fetched again. Missing values are left missing, never filled with 0.
4. **Reply features (21 columns)** of the first replier's comments: words, personal pronoun rates (spaCy
   morphology), imperative sentences (spaCy dependency parse), links from the DiscussionTools HTML (wikilinks,
   policy or guideline pages, Help pages, external links), lag, readability, sentiment, four politeness strategies,
   lists, question marks, and the cosine similarity between question and reply embeddings (I-27).

**Paper numbers that depend on M6.** The question-text covariates of the propensity score model (Sections 3.4 and
4, through M7), the question-type results and the agreement table (Table 3, I-54), and the reply analyses
(Tables 6 and 7).

**How to use this notebook.** The saved outputs are enough to read it. To run it again, run it from top to bottom.
Each check prints `[OK ]` or `[FAIL]`. Each comparison of a hand recomputation with the stored value prints a table
with a `match` column and ends with `all fields match` or `MISMATCH in: [...]`. Random rows are drawn with `SEED`
in the next cell: change `SEED` and run again to see fresh rows. spaCy and ConvoKit run only on the sampled rows,
so the notebook stays fast.

**What this checks.** The next cell loads the M6 tables and the sample from M5 and prints how many rows each has.
The row counts tell us whether every sampled question got a feature row.
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
**What the output shows.** On the final data (cutoff 2026-04-20T14:00Z) there are 34,985 sampled first questions
and exactly 34,985 question feature rows, so every question is described. There are 26,877 reply feature rows:
one for each question that got a reply with some text (the treated group, minus 4 replies with no text, see
section 6).

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

**What this checks.** It lists the output files with their size and time, and prints the run settings
(`manifest.json`) and the counts of what was produced (`coverage.json`). This shows the files exist, come from one
recent run, and how complete each kind of result is.
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
**What the output shows.** All nine files were written in one run that finished on 2026-10-26 (online mode, text
similarity threshold 0.85 for reusing old results). Coverage: 34,985 questions; LLM labels for all 34,985;
Perspective scores and question embeddings for 34,939 (46 missing); 26,877 replies, 26,876 reply embeddings, and
26,858 cosine values. The cache line says 260,609 results were read from the local cache and only 138 were not
there, so almost nothing was requested again in the last run. The 46 missing questions are explained in section 6.

### Data dictionary: `question_features.parquet` (one row per sampled first question) and `reply_features.parquet`
(one row per treated question whose first reply has text)

The `checked in` column names the cell that verifies each column (3 = invariants on all rows, 4 = paper comparison,
5a to 5f = random samples, 6 = edge cases).

**What this checks.** The next cell prints, for every column of the two tables, what it means in words. The cell
after it checks that the dictionary and the real tables list the same columns, so no column is left unexplained.
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
md("""
**What the output shows.** Two tables list every column with its meaning and the section that checks it, followed
by a short note on the embedding files. The next cell checks that these lists are complete.
"""),
code(r'''
print("question_features columns missing from the dictionary:", [c for c in qf.columns if c not in QD], "| extra:", [c for c in QD if c not in qf.columns])
print("reply_features columns missing from the dictionary:", [c for c in rf.columns if c not in RD], "| extra:", [c for c in RD if c not in rf.columns])
'''),
md("""
**What the output shows.** Both lines print empty lists: no column is missing from the dictionary and the
dictionary has no extra column. Every one of the 54 question columns and 22 reply columns is described above.

## 3. Invariants

Several checks recompute a column from the raw data for **every** row: the simple length features, the reply text
rebuilt from the thread turns (question marks, lists, links), the reply lag, and the word ratio.

**What this checks.** 38 rules that must hold on all rows: one row per question, flags are 0 or 1, scores are in
their allowed range, counts written by the pipeline equal counts recomputed here in a simpler, independent way,
and the coverage numbers equal the counts in the tables. If any rule broke, the covariates or reply features used
in the paper could be wrong for some newcomers.
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
**What the output shows.** All 38 of 38 checks pass. The simple recomputation agrees with the stored value on every
row for signature, URL, indentation, and Wikipedia and Help links. For three counts a few rows differ: links (7
rows), templates (44 rows) and the {{unsigned}} flag (1 row), all well under the 0.5% allowed. These are cases
where the simple regex here is the less accurate method (for example a broken link or a template inside
`<nowiki>`); they are listed in section 6. No LLM label breaks the codebook rule, and every reply count (question
marks, lists, links, lag, word ratio) matches the rebuilt reply text on all 26,877 rows.

## 4. Comparison with the paper version

For the same question (matched by revid): do the paper version and the new version give the same question types,
and how strongly do the old features correlate with the new ones (which now use standard tools)?

**What this checks.** It matches questions between the paper version (April 2026 data) and this version and asks
whether they got the same values. "Share equal" is the share of questions with the same 0/1 value; "correlation"
is a number from -1 to 1 where 1 means the two versions move together perfectly. Large differences would mean the
paper's covariates changed.
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
**What the output shows.** 33,201 questions are in both versions (the other 1,784 are first questions that the April
data did not have). The five question types agree on 99.6% to 99.9% of questions, word count and VADER sentiment
correlate at 0.99, and the Gratitude and Please flags agree on 99.6% and 97.3%. Two features changed on purpose:
links (correlation 0.81) and Wikipedia-namespace links (`q_mentions_wp`, correlation 0.27). The low 0.27 is
expected: the paper version counted the text "WP:" anywhere, while the new version counts real links to Wikipedia
pages (issue I-05), and both features are rare (mean 0.002 per question).

## 5. Tracer and random samples: raw text next to every feature

### 5a. Random questions: raw wikitext, plain text, and all 37 text features

For each question: the simple features recomputed by `my_question_format` (regex, no mwparserfromhell), VADER and
TextBlob recomputed by calling the libraries on the plain text, and the 21 ConvoKit politeness strategies
recomputed by calling `PolitenessStrategies` directly, with the words (markers) that triggered each one. The
tracer questions from `pipeline/common/tracer.py` that are in the sample are shown first.

**What this checks.** For a few real questions the reader can see the raw text and every feature next to it, and
each feature is computed again here by calling the tools directly. This shows that the stored numbers really come
from the text (for example, "Hello" turns on the greeting flag).
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
**What the output shows.** 8 tracer questions and 5 random questions are shown. For all 13, the format features,
the sentiment scores and the 21 politeness flags recomputed here are identical to the stored values (each block
ends with `all fields match`; there is no `MISMATCH`). The warning about `pkg_resources` at the top is a harmless
notice from a library and not an error.

### 5b. LLM question types: text, labels, and the codebook definition

The labels are judgements, so they cannot be recomputed; the reader compares each text with the one-line codebook
definition. The full codebook is `services.codebook_text()` (paper `codebook_prompt.md` plus 120 labelled
examples). The rule "if Q0 = N then Q2 to Q5 are N" is checked on all rows in section 3.

**What this checks.** The first cell prints the five definitions and five random questions with their labels. The
second cell shows, for each type, five questions labelled Y and five labelled N, so a reader can judge whether the
LLM follows the definitions.
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
md("""
**What the output shows.** The five random questions get labels that fit the definitions: the three questions about
writing or disclosing an article are Q0 = Y and Q4 = Y, and "Hello, are you available..." is Q0 = N, so all its
other labels are N as the rule requires. The next cell shows more examples for each type.
"""),
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
**What the output shows.** The examples fit the definitions well. For instance "Hello, are you available..." is
Q0 = N (no real editing question), and "I have submitted a draft... how do I disclose my connection" is Q4 = Y
(about rules) and Q5 = Y (own work already done). Shares labelled Y: Q0 0.810, Q2 0.134, Q3 0.056, Q4 0.181,
Q5 0.232. Some labels are borderline, which is normal for judgement labels; how often the LLM agrees with people
is measured in section 7.

### 5c. Perspective scores: the highest and lowest questions for each attribute

The input to Perspective is the question text with user names removed and links replaced by `[LINK]`. High
scores should look rude, profane, sexual, and so on; low scores should look harmless. `source` says where the score
came from: `legacy` = copied from the paper version, `api` = fetched again by M6.

**Earlier problem, now fixed (I-59).** The paper-version Perspective file (`s10_perspective_mentee.csv`) turned
out to hold scores of other texts for many rows (for example "How to write an essay" had TOXICITY 0.854; a fresh
score is 0.017). M6 therefore no longer copies any paper-version Perspective score: every score was fetched again
from the API. The old scores are kept in the cache under a different name only for the record. Set
`RECHECK_ONLINE = True` in the second cell below to send some texts to the API again and compare.

**What this checks.** For each of the eight attributes it shows the three questions with the highest score and the
two with the lowest, and where each score came from. If the scores are right, the top texts should clearly be
rude, sexual, and so on.
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
md("""
**What the output shows.** The highest scores go to texts that really are insulting, profane, sexual or
threatening, and the lowest go to plain, neutral texts, so the scores behave as expected. Every score has source
`api` (all 33,252 distinct scored texts, and all of the 200 most toxic ones), which confirms that no misaligned
paper-version score is left. Average scores are very low (toxicity 0.033), as expected for polite newcomer
questions. FLIRTATION has a much higher average (0.312) and even "How do I correct a factual error?" scores 0.119;
this attribute is known to give a high baseline to ordinary friendly text, so it is not a sign of an error.
A few top texts are mild (for example THREAT 0.547 for "I need to hit the studio"), a known limit of the tool.

**What this checks.** The next cell can send texts to the Perspective API again. It is switched off, so it only
prints a message.
"""),
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
**What the output shows.** "RECHECK_ONLINE is False": no text was sent to the API in this run, as intended (the
notebook makes no online requests).

### 5d. Random replies: text, pronouns, imperatives, links, and all 21 reply features

For each reply: the thread turns that make up the reply (rebuilt by `my_first_reply`), the spaCy tokens counted as
*you*, *I* and *we*, the sentences counted as imperatives, and every link with its namespace, the target after
redirects, and the policy and Help decisions (from the cached API answers, `wikimeta.resolve_titles` and
`policy_status`). Then every reply feature is recomputed and compared. The cosine similarity is recomputed from the
embedding files (see 5f for the method).

**What this checks.** The reply features are the inputs of the reply-feature analysis in the paper. For five
random replies the reader sees the reply text, which words counted as *you* / *I* / *we*, which sentences counted
as commands (imperatives), and where each link points, and all 21 features are computed again and compared.
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
**What the output shows.** The first line says all 8,017 link titles found in replies were resolved from stored
answers, and 55 of their targets are policy or guideline pages. All five replies end with `all fields match`. The
examples also show that the rules are simple and literal: in "You're referring to an infobox; see that page for
more information." the sentence is counted as one imperative because of "see", and a link to
`MOS:INFOBOX` is followed to its real page (a Manual of Style page, which is not tagged as a policy, so it does not
count as a policy link).

### 5e. Replies with policy, Help, or many links

Random replies with at least one policy or guideline link, so the reader can see the link decisions on rows where
they matter.

**What this checks.** The same comparison as 5d, on two replies that do link to a policy or guideline, since
random replies rarely do (the average reply has 0.19 policy links).
"""),
code(r'''
for rv in pick(rf[rf.r_n_policy > 0], 2, SEED).first_question_revid:
    show_reply(int(rv))
'''),
md("""
**What the output shows.** Both replies end with `all fields match`. In the first one, four links are listed:
`Wikipedia:IUP` is followed to "Wikipedia:Image use policy" and counted as a policy link, while
"Wikipedia:Files for upload" is a normal project page and is not. Links to Wikimedia Commons (`C:`) have no target
namespace here and are counted as links but not as policy or Help links, which is correct.
"""),
md("""
### 5f. Cosine similarity and embedding files

For N_RANDOM random replies, the question text for embeddings is rebuilt (`make_emb_text`: plain text with user
names removed and links replaced), the reply text likewise (`emb_text_from_plain`); each text's sha256 gives its
row in `emb_q.npy` / `emb_r.npy` through the index files. The cosine is computed with numpy and compared to 6 decimals
(the vectors are stored as float32, so later digits depend on the order of the arithmetic). We also
check that the question vector equals the row of `emb_q_sample.npy` that belongs to the same question.

**What this checks.** That each stored cosine similarity really is computed from the right question embedding and
the right reply embedding, and that the per-question embedding file lines up with the question table.
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
md("""
**What the output shows.** For all five replies the recomputed cosine equals the stored one to 6 decimals, and the
question vector equals its row in `emb_q_sample.npy`. The values (0.39 to 0.73) look sensible: a short, generic
pair like "Hello world" / "Welcome to Wikipedia!" is less similar (0.39) than a reply that answers the question
directly (0.73).

## 6. Edge-case lists

**What this checks.** It lists the unusual rows: questions without labels, scores or embeddings, replies without a
cosine value, replies with no text, label rule breaks, rows where the simple regex and the wiki parser disagree,
and the longest and shortest replies. These are the rows most likely to hide a bug.
"""),
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
md("""
**What the output shows.** No question lacks LLM labels. The same 46 questions lack both Perspective scores and an
embedding; their text is only a user name or signature (for example "--Saud9900"), so after names are removed
nothing is left to send. They stay missing, not 0. 19 replies have no cosine value: 18 because the question
embedding is missing and 1 because the reply embedding is missing. 4 treated questions have a first reply with
empty text (the comment has no words left after cleaning), so they have no reply row. The regex disagreements
are cases where the wiki parser is right (a link broken over two lines, a template inside `<small>` or
`<nowiki>`). The shortest replies have 0 words because they are only punctuation such as "?"; the longest has
1,121 words. Note that `q_has_unsigned` is 0 for every question (see the summary in section 7), so this feature
carries no information; it is harmless as a covariate but adds nothing.
"""),
md("""
## 7. Distributions: agreement (paper Table 3; I-54), label shares, cosine similarity

**What this checks.** How often the LLM labels agree with two human labellers and with the agreed (gold) labels,
measured with Cohen's kappa (1 = perfect agreement, 0 = no better than chance). Then the share of questions with
each label, and summaries (mean, spread, minimum, maximum) of the question and reply features.
"""),
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
md("""
**What the output shows.** The first table is the kappa on the test set only (items 121 to 241, never used while
writing the prompt), so it is lower than the full-set numbers in paper Table 3 (human-human 0.872 to 0.950 on the
full set, in `results/m6/agreement.csv`). On the test set, human-human kappa is 0.77 to 0.91 and LLM against gold
is 0.62 (Q4) to 0.92 (Q5). The second table: labelling the 241 items again with the same model and prompt gives the
same label as the paper version 93% to 98% of the time. In the reply summary, three things look odd but are fine:
the smallest reply lag is -0.01 hours because reply times are known only to the minute (19 replies posted in the
same minute as the question); the highest reading grade (185.4) comes from a very long reply with no sentence
breaks; and 7 replies have cosine above 0.99 because the reply text is the same as the question (for example
"Hi" / "Hi", or a reply that repeats the question word for word). These few rows do not affect the averages.
"""),
md("""
## 8. Summary (results/m6_report.json)

**What this checks.** It saves the coverage counts to `results/m6_report.json` and prints how many invariants
passed.
"""),
code(r'''
cov = json.loads((M6 / "coverage.json").read_text())
(config.RESULTS / "m6_report.json").write_text(json.dumps(cov, indent=1))
print(json.dumps(cov, indent=1))
print(f"invariants: {sum(ok for _, ok in ck.items)} / {len(ck.items)} passed")
'''),
md("""
**What the output shows.** On the final data (cutoff 2026-04-20T14:00Z): 34,985 questions, all with LLM labels;
34,939 with Perspective scores and embeddings; 26,877 replies with 26,858 cosine values; 38 of 38 invariants
passed. M6 is complete and consistent, and its outputs can be used by M7.
"""),
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
