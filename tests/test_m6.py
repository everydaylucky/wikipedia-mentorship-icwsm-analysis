"""
test_m6.py: M6 text features and cache (CODE_PLAN.md section 7.1).
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
from pipeline.m6_text.agreement import kappa
from pipeline.m6_text.build_text import emb_text_from_plain, first_reply, similar
from pipeline.m6_text.cache import TextCache
from pipeline.m6_text.services import parse_labels
from pipeline.m6_text.text_features import _nlp, count_imperatives, pronoun_rates, question_features

M6 = config.INTERIM / "m6"
needs_m6 = pytest.mark.skipif(not (M6 / "question_features.parquet").exists(), reason="M6 has not been run")


def test_cache_roundtrip(tmp_path):
    c = TextCache(tmp_path / "c.sqlite")
    c.put_many({"Hello  world": {"Q0": "Y"}}, "t", "m", "v", source="test")
    assert c.get_many(["Hello world"], "t", "m", "v") == {"Hello world": {"Q0": "Y"}}   # same key after whitespace is normalized
    assert c.missing(["Hello world", "other"], "t", "m", "v") == ["other"]
    # two texts that differ only in whitespace share one key, and both must get a result (real example "Hello,\nHow do I create a citation?")
    assert set(c.get_many(["Hello world", "Hello\nworld"], "t", "m", "v")) == {"Hello world", "Hello\nworld"}


def test_reply_features_split_matches_columns():
    from pipeline.m6_text.text_features import REPLY_COLS, reply_features, reply_text_features
    t = "Hi, thanks for asking! Please read WP:RS. You can add a citation."
    links = {"wikilinks": [{"title": "Wikipedia:Reliable sources", "ns": 4}], "n_external": 1, "n_list_items": 0}
    a = reply_features(t, links, 10, 2.0, {"Wikipedia:Reliable sources"}, {})
    b = reply_features(t, links, 10, 2.0, {"Wikipedia:Reliable sources"}, {}, base=reply_text_features(t))
    assert list(a) == REPLY_COLS and a == b and a["r_n_policy"] == 1 and a["r_n_resources"] == 2


def test_question_features_wiki_format():
    w = ("Hi! How do I cite a source in [[Draft:My page]]? I read [[WP:RS]] and [[H:CITE]]. https://x.org "
         "--[[User:A|A]] ([[User talk:A|talk]]) 10:00, 1 March 2026 (UTC)")
    f = question_features(w, "Hi! How do I cite a source in Draft:My page? I read WP:RS and H:CITE. https://x.org")
    assert f["q_has_sig"] == 1 and f["q_link_count"] == 3 and f["q_mentions_wp"] == 1 and f["q_mentions_help"] == 1
    assert f["q_has_url"] == 1 and f["q_poly_feature_politeness_==Indirect_(greeting)=="] == 1


def test_imperatives_and_pronouns():
    d = _nlp()("Click the edit button. You should then add a reference. Please read the guide. We can help you.")
    assert count_imperatives(d) == 2
    you, i, we = pronoun_rates(d)
    assert you > 0 and we > 0 and i == 0


def test_parse_labels_and_kappa():
    assert parse_labels('```json\n{"Q0":"Y","Q2":"N","Q3":"N","Q4":"N","Q5":"Y"}\n```')["Q5"] == "Y"
    assert parse_labels("no answer") is None
    assert kappa(list("YYNN"), list("YYNN")) == 1.0


def test_legacy_similarity_and_reply_text():
    assert similar("How do I add  a citation?", "how do i add a citation?")[0]
    assert not similar("Can I earn money using Wikipedia?",
                       "Can I earn money using Wikipedia? Hello! If you are referring to getting paid")[0]
    turns = [{"author": "N", "role": "mentee", "ts": "2025-01-01T00:00:00Z", "text": "Q?", "links": None},
             {"author": "M", "role": "page_mentor", "ts": "2025-01-01T01:00:00Z", "text": "First part.", "links": None},
             {"author": "M", "role": "page_mentor", "ts": "2025-01-01T01:05:00Z", "text": "Second part.", "links": None},
             {"author": "N", "role": "mentee", "ts": "2025-01-01T02:00:00Z", "text": "Thanks", "links": None}]
    text, links, names = first_reply(turns, "2025-01-01T00:00:00Z")
    assert text == "First part.\n\nSecond part."
    assert emb_text_from_plain("Hi Newbie, see https://x.org", ["Newbie"]) == "Hi , see [LINK]"


@needs_m6
def test_question_features_cover_sample():
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet")
    q = pd.read_parquet(M6 / "question_features.parquet")
    want = set(s[s.in_sample | s.in_sample_unrestricted].first_question_revid)
    assert want <= set(q.first_question_revid)
