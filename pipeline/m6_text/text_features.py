"""
text_features.py: text features computed locally: question features (the question-text block of M7) and reply
features (the reply analyses of M8)

Module: M6 (PLAN.md M6 items 5, 6; CODE_PLAN.md section 5, M6)
Rules: DATA_PROCESSING R4.4 to R4.6. Decisions: D6 (use ConvoKit for politeness), D9 (standard methods).
Issues: I-04, I-05, I-06, I-27

1. What it does
    question_features(wikitext, plain): the 37 columns of the paper version's X_Qtext. The names stay the same, but
    the definitions now use standard methods:
        q_has_sig            the body has a signature timestamp (MediaWiki signature format "hh:mm, d Month yyyy (UTC)").
        q_has_unsigned       uses a template from the {{unsigned}} family (template names parsed with mwparserfromhell).
        q_link_count         number of internal links in the body that do not point to user pages (mwparserfromhell;
                             namespaces follow the official table).
        q_template_count     number of templates in the body (mwparserfromhell, top-level templates only).
        q_has_url            has an external link or a bare URL.
        q_mentions_wp        number of links to the Wikipedia: namespace (ns 4, including official aliases such as WP:)
                             (I-05: the paper version counted the string "WP:").
        q_mentions_help      number of links to the Help: namespace (ns 12, including H: pseudo-namespace redirects).
        q_is_indented        the body starts with a talk-page indent mark ":" or "*" (I-05: the paper called this a
                             "reply", but this is what it actually measures).
        q_body_char_len, q_body_word_count  length and word count of the plain text.
        q_vader_*            VADER (vaderSentiment); q_tb_*  TextBlob.
        q_poly_feature_politeness_==…==  the 21 strategies of ConvoKit PolitenessStrategies (D6, I-06).
    reply_features(...): the 21 reply features used in paper Tables 6 and 7 (s15_2 REPLY_FEATURES). The definitions
    now use standard methods:
        r_words              word count (textstat.lexicon_count).
        r_pronoun_{you,i,we}_rate  personal pronoun rates: spaCy UD morphology features PronType=Prs with Person=2 /
                             Person=1 singular / Person=1 plural, divided by the word count.
        r_vader_compound, r_tb_subjectivity.
        r_flesch_kincaid     textstat.flesch_kincaid_grade.
        r_n_imperatives      number of imperative sentences: from the spaCy dependency parse, the sentence root is a
                             base-form verb (VB) and has no subject (nsubj, nsubjpass, expl).
        r_n_wikilink, r_n_policy, r_n_help, r_n_resources  links come from the reply HTML given by DiscussionTools:
                             internal links (not counting links to users); of these, the ones pointing to policy or
                             guideline pages (official categories Category:Wikipedia policies/guidelines, after
                             resolving redirects); the ones pointing to the Help: namespace; total resource links =
                             internal links + external links (the paper version added up all four, so it double counted).
        r_reply_lag_hours, r_reply_q_word_ratio.
        r_has_greeting, r_polite_gratitude, r_polite_has_positive, r_polite_deference  the matching ConvoKit strategies.
        r_has_list           the reply has a list item (HTML <li>); r_n_question_marks  number of question marks.
        r_qr_cosine_sim      cosine similarity of the question and reply embeddings (I-27; computed by build_text when
                             embeddings exist).

2. Why it is designed this way
    The paper version used hand-written regexes (the 21 politeness strategies, pronouns, greetings, and so on).
    Decisions D6 and D9 require switching to tools that are standard in the field. The variable names stay the
    same so the M8 analysis code can be reused as is. Changes in meaning are written into the data dictionary.

3. Inputs / 4. Outputs
    Text in; a dict out.

8. How to check
    tests/test_m6.py; in checks/m6_check.ipynb, each feature is checked against the original text on a sample.
"""
from __future__ import annotations

import re
from functools import lru_cache

import mwparserfromhell as mwp

from pipeline.common.textclean import AWAY_NOTE_RE
from pipeline.common.timeutil import SIG_TS_RE
from pipeline.common.wikimeta import namespace_table

POLITE_KEYS = ["Please", "Please_start", "HASHEDGE", "Indirect_(btw)", "Hedges", "Factuality", "Deference", "Gratitude",
               "Apologizing", "1st_person_pl.", "1st_person", "1st_person_start", "2nd_person", "2nd_person_start",
               "Indirect_(greeting)", "Direct_question", "Direct_start", "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE",
               "INDICATIVE"]
QTEXT_COLS = (["q_has_sig", "q_has_unsigned", "q_link_count", "q_template_count", "q_has_url", "q_mentions_wp",
               "q_mentions_help", "q_is_indented", "q_body_char_len", "q_body_word_count", "q_vader_neg", "q_vader_neu",
               "q_vader_pos", "q_vader_compound", "q_tb_polarity", "q_tb_subjectivity"]
              + [f"q_poly_feature_politeness_=={k}==" for k in POLITE_KEYS])
REPLY_COLS = ["r_words", "r_pronoun_you_rate", "r_pronoun_i_rate", "r_pronoun_we_rate", "r_vader_compound",
              "r_tb_subjectivity", "r_flesch_kincaid", "r_n_imperatives", "r_n_resources", "r_n_wikilink", "r_n_policy",
              "r_n_help", "r_reply_lag_hours", "r_reply_q_word_ratio", "r_has_greeting", "r_polite_gratitude",
              "r_polite_has_positive", "r_polite_deference", "r_has_list", "r_n_question_marks", "r_qr_cosine_sim"]
_UNSIGNED = {"unsigned", "unsigned2", "unsignedip", "unsigned ip", "unsignedip2", "unsigned-ip", "subst:unsigned"}
_URL_RE = re.compile(r"https?://", re.I)
_USER_NS = {2, 3}
# version number of the feature definitions: add one whenever any feature's method changes, so old cached results are no longer used
FEATURE_VERSION = "1"


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def _politeness():
    from convokit import PolitenessStrategies
    return PolitenessStrategies(verbose=0)


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def politeness(text: str) -> dict[str, int]:
    """ConvoKit PolitenessStrategies: 0/1 for each of the 21 strategies (D6)."""
    if not (text or "").strip():
        return {k: 0 for k in POLITE_KEYS}
    u = _politeness().transform_utterance(text, spacy_nlp=_nlp())
    st = u.meta["politeness_strategies"]
    return {k: int(st.get(f"feature_politeness_=={k}==", 0) > 0) for k in POLITE_KEYS}


def sentiment(text: str) -> dict[str, float]:
    """The four VADER scores and the two TextBlob scores."""
    from textblob import TextBlob
    if not (text or "").strip():
        return {"vader_neg": 0.0, "vader_neu": 0.0, "vader_pos": 0.0, "vader_compound": 0.0,
                "tb_polarity": 0.0, "tb_subjectivity": 0.0}
    v = _vader().polarity_scores(text)
    tb = TextBlob(text).sentiment
    return {"vader_neg": v["neg"], "vader_neu": v["neu"], "vader_pos": v["pos"], "vader_compound": v["compound"],
            "tb_polarity": tb.polarity, "tb_subjectivity": tb.subjectivity}


def wiki_format(wikitext: str, plain: str) -> dict[str, int]:
    """Formatting features of the question wikitext (see section 1 above, R4.5). Away notes are removed first (R4.4). The part before the signature counts as the body."""
    nst = namespace_table()
    w = AWAY_NOTE_RE.sub("", wikitext or "", count=1)
    m = SIG_TS_RE.search(w)
    code = mwp.parse(w)
    body = mwp.parse(w[:m.start()]) if m else code
    links = [nst.split(str(l.title).strip().lstrip(":")) for l in body.filter_wikilinks()]
    content = [p for p in links if p.ns not in _USER_NS]
    tnames = {str(t.name).strip().lower() for t in code.filter_templates()}
    return {"q_has_sig": int(m is not None),
            "q_has_unsigned": int(bool(tnames & _UNSIGNED)),
            "q_link_count": len(content),
            "q_template_count": len(body.filter_templates(recursive=False)),
            "q_has_url": int(bool(body.filter_external_links()) or bool(_URL_RE.search(str(body)))),
            "q_mentions_wp": sum(p.ns == 4 for p in content),
            "q_mentions_help": sum(p.ns == 12 or p.full.lower().startswith("h:") for p in content),
            "q_is_indented": int(w.lstrip().startswith((":", "*"))),
            "q_body_char_len": len(plain or ""),
            "q_body_word_count": len((plain or "").split())}


def question_features(wikitext: str, plain: str) -> dict[str, float]:
    """The 37 columns of the paper version's X_Qtext (column names are in QTEXT_COLS)."""
    out = wiki_format(wikitext, plain)
    out.update({f"q_{k}": v for k, v in sentiment(plain).items()})
    out.update({f"q_poly_feature_politeness_=={k}==": v for k, v in politeness(plain).items()})
    return {c: out[c] for c in QTEXT_COLS}


def pronoun_rates(doc) -> tuple[float, float, float]:
    """spaCy UD morphology: among personal pronouns (PronType=Prs), the counts of second person, first person singular, and first person plural, each divided by the word count."""
    words = [t for t in doc if t.is_alpha]
    n = max(len(words), 1)
    you = i = we = 0
    for t in doc:
        m = t.morph
        if "Prs" not in m.get("PronType"):
            continue
        person, number = m.get("Person"), m.get("Number")
        if "2" in person:
            you += 1
        elif "1" in person and "Plur" in number:
            we += 1
        elif "1" in person:
            i += 1
    return you / n, i / n, we / n


def count_imperatives(doc) -> int:
    """Imperative sentence: the sentence root is a base-form verb (tag VB), and the root has no subject (nsubj, nsubjpass, expl)."""
    n = 0
    for sent in doc.sents:
        root = sent.root
        if root.tag_ == "VB" and root.pos_ in ("VERB", "AUX") and \
                not any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in root.children):
            n += 1
    return n


def reply_text_features(text: str) -> dict[str, float]:
    """The reply features that depend only on the reply text (spaCy, ConvoKit, VADER, TextBlob, textstat). These can be cached by text."""
    import textstat
    doc = _nlp()(text or "")
    you, i, we = pronoun_rates(doc)
    words = textstat.lexicon_count(text or "", removepunct=True)
    s = sentiment(text)
    pol = politeness(text)
    return {"r_words": words, "r_pronoun_you_rate": you, "r_pronoun_i_rate": i, "r_pronoun_we_rate": we,
            "r_vader_compound": s["vader_compound"], "r_tb_subjectivity": s["tb_subjectivity"],
            "r_flesch_kincaid": textstat.flesch_kincaid_grade(text) if words else 0.0,
            "r_n_imperatives": count_imperatives(doc),
            "r_has_greeting": pol["Indirect_(greeting)"], "r_polite_gratitude": pol["Gratitude"],
            "r_polite_has_positive": pol["HASPOSITIVE"], "r_polite_deference": pol["Deference"],
            "r_n_question_marks": (text or "").count("?")}


def reply_features(text: str, links: dict, q_words: int, lag_hours: float | None, policy_titles: set[str],
                   resolved: dict[str, dict], base: dict | None = None) -> dict[str, float]:
    """Reply features for one first replier (see section 1 above; the caller fills in r_qr_cosine_sim).

    Args:
        links: the html_links results of all the reply's comments, merged:
            {"wikilinks": [...], "n_external": int, "n_list_items": int}.
        policy_titles: the set of final titles of policy or guideline pages.
        resolved: title -> resolve_titles result (redirects).
        base: the result of reply_text_features(text). Pass it in when it was found in the cache, so the text does
            not need to be analyzed again.
    """
    base = reply_text_features(text) if base is None else base
    wl = links.get("wikilinks", [])
    final = [resolved.get(l["title"], {}).get("final", l["title"]) for l in wl]
    fns = [resolved.get(l["title"], {}).get("ns", l["ns"]) for l in wl]
    out = dict(base)
    out.update({"r_n_resources": len(wl) + links.get("n_external", 0), "r_n_wikilink": len(wl),
                "r_n_policy": sum(f in policy_titles for f in final), "r_n_help": sum(ns == 12 for ns in fns),
                "r_reply_lag_hours": lag_hours, "r_reply_q_word_ratio": base["r_words"] / max(q_words, 1),
                "r_has_list": int(links.get("n_list_items", 0) > 0), "r_qr_cosine_sim": None})
    return {c: out[c] for c in REPLY_COLS}


# ── Multiprocessing workers (called by build_text.cached_features; they must be top-level functions so child processes can import them) ──────────
def question_features_batch(pairs: list[tuple[str, str]]) -> list[dict]:
    """Question features for a batch of (wikitext, plain) pairs."""
    return [question_features(w, p) for w, p in pairs]


def reply_text_features_batch(texts: list[str]) -> list[dict]:
    """Text features for a batch of reply texts."""
    return [reply_text_features(t) for t in texts]
