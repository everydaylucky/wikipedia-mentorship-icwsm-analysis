"""
test_common.py: fixed behavior of the shared layer (CODE_PLAN.md sections 4 and 7.1).

These tests turn the rules in DATA_PROCESSING.md into checks that the computer can run.
When a rule changes, change these tests first, then change the code.
All tests run offline. They never use the network.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import config
from pipeline.common import checks, io, timeutil, users
from pipeline.common.api import MediaWikiClient, OfflineCacheMiss
from pipeline.common.issues import FatalIssue, IssueLog
from pipeline.common.signatures import find_signatures
from pipeline.common.textclean import (count_list_items, extract_links, extract_templates,
                                       make_emb_text, strip_away_note, to_plain_text)
from pipeline.common.wikimeta import NamespaceTable, namespace_table


# ── config ──────────────────────────────────────────────────────────────────
def test_config_cutoff_matches_paper_collection():
    """The last question in the paper version is at 2026-04-20T13:54:14Z. The cutoff time of 14:00 just includes it."""
    assert config.DATA_CUTOFF == datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc)
    assert config.DATA_CUTOFF_ISO == "2026-04-20T14:00:00Z"
    assert config.DATA_CUTOFF.isoformat().replace("+00:00", "Z") == config.DATA_CUTOFF_ISO
    assert timeutil.before_cutoff("2026-04-20T13:54:14Z")


def test_config_raw_files_exist():
    for k, p in config.RAW_FILES.items():
        assert p.exists(), f"{k} missing: {p}"


# ── timeutil: cutoff date (PLAN section 0.1) ──────────────────────────────────────────
def test_time_before_cutoff_excludes_cutoff_instant():
    assert timeutil.before_cutoff("2026-04-20T13:59:59Z")
    assert not timeutil.before_cutoff("2026-04-20T14:00:00Z")
    assert not timeutil.before_cutoff("2026-04-23T00:00:00Z")
    assert not timeutil.before_cutoff("2026-06-01T00:00:00Z")
    assert not timeutil.before_cutoff(None)          # records with unknown time are left out of the analysis


def test_time_parse_sig_ts():
    dt = timeutil.parse_sig_ts("--[[User:X|X]] 15:55, 27 March 2024 (UTC)")
    assert dt == datetime(2024, 3, 27, 15, 55, tzinfo=timezone.utc)
    assert timeutil.parse_sig_ts("no timestamp here") is None
    assert timeutil.parse_sig_ts("10:00, 31 February 2024 (UTC)") is None


def test_time_naive_iso_is_utc():
    assert timeutil.parse_iso("2024-01-01T00:00:00") == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_time_add_days_is_exact_hours():
    t0 = datetime(2024, 1, 1, 13, 30, tzinfo=timezone.utc)
    assert timeutil.add_days(t0, 14) == datetime(2024, 1, 15, 13, 30, tzinfo=timezone.utc)


# ── users: R6.2, R5.1, R5.2 ──────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("dr_vulpes", "Dr vulpes"),
    ("User talk:Foo__Bar ", "Foo Bar"),
    ("lajmmoore", "Lajmmoore"),
    ("Foo Bar", "Foo Bar"),
    ("方志維", "方志維"),
    ("", None),
])
def test_norm_user(raw, expected):
    assert users.norm_user(raw) == expected


def test_strip_subpage_and_alt_accounts():
    assert users.strip_subpage("Gommeh/guestbook") == "Gommeh"
    assert users.canonical_person("Yoshi24517 (mobile)") == "Yoshi24517"
    assert users.canonical_person("bsoyka/guestbook") == "Bsoyka"


def test_user_index_time_aware_lookup():
    idx = users.UserIndex()
    # The talk page of 方志維 was moved to the new account 楊文祖 on 2025-06-30 (two different uids)
    idx.add(45329123, "方志維", end="2025-06-30T20:35:19Z", source="move_log")
    idx.add(49604360, "楊文祖", start="2025-04-30T00:04:11Z", source="api_users")
    assert idx.lookup("方志維", at="2024-07-19T10:36:17Z") == 45329123
    assert idx.lookup("方志維", at="2025-07-01T00:00:00Z") is None
    assert idx.lookup("楊文祖") == 49604360


def test_user_index_ambiguous_returns_none():
    idx = users.UserIndex()
    idx.add(1, "Same name")
    idx.add(2, "Same name")
    assert idx.lookup("Same name") is None
    assert "Same name" in idx.ambiguous


# ── signatures: R5.1 ─────────────────────────────────────────────────────────
def test_signatures_subpage_and_alt():
    txt = ("Thanks! [[User:Gommeh/guestbook|Gommeh]] 16:52, 12 February 2026 (UTC)\n"
           ":Reply [[User:Yoshi24517 (mobile)|Yoshi]] ([[User talk:Yoshi24517 (mobile)|talk]]) "
           "00:05, 28 August 2024 (UTC)")
    sigs = find_signatures(txt)
    assert [s.user for s in sigs] == ["Gommeh", "Yoshi24517"]
    assert sigs[0].ts == datetime(2026, 2, 12, 16, 52, tzinfo=timezone.utc)


def test_signatures_ignore_bare_timestamp():
    assert find_signatures("The meeting is at 15:55, 27 March 2024 (UTC) online.") == []


# ── textclean: R4.4 to R4.6 ──────────────────────────────────────────────────
AWAY_1343488016 = (":''Note: [[User talk:SashankInAnitquity|SashankInAnitquity]]'s mentor "
                   "[[User talk:Dr vulpes|Dr vulpes]] is away.''\nHey do you know how to raise questions?")


def test_strip_away_note_real_example():
    text, note = strip_away_note(AWAY_1343488016)
    assert note is not None
    assert note.primary_mentor == "Dr vulpes" and note.mentee == "SashankInAnitquity"
    assert text.startswith("Hey do you know")


def test_strip_away_note_after_leading_template():
    t = "{{Main|Wikipedia:AIV}}\n" + AWAY_1343488016
    text, note = strip_away_note(t)
    assert note is not None and "is away" not in text and text.startswith("{{Main")


def test_strip_away_note_i18n_variant():
    t = ("Note: [[User:A|A]]'s mentor <span class=\"plainlinks\">"
         "[https://en.wikipedia.org/wiki/User:B B]</span> is away.\nQuestion")
    _, note = strip_away_note(t)
    assert note is not None and note.primary_mentor == "B"


def test_to_plain_text_removes_signature_templates_userlinks():
    raw = ("Hi {{ping|Foo}}, please see [[WP:REFB|this page]]. "
           "--[[User:UtherSRG|UtherSRG]] ([[User talk:UtherSRG|talk]]) 15:51, 26 September 2025 (UTC)")
    assert to_plain_text(raw) == "Hi, please see this page."


def test_make_emb_text_removes_participant_names():
    raw = "Hello Mujinga, SpacedFarmer asked about [[WP:TWINK]]."
    assert "Mujinga" not in make_emb_text(raw, ["Mujinga", "SpacedFarmer"])


def test_extract_links_uses_official_namespaces():
    links = extract_links("See [[WP:RS]], [[Help:Talk pages]], [[Project:NPOV]] and "
                          "[[User:X|X]] and [[File:A.png]] and [[Some article]].")
    got = {(l["full"], l["ns"]) for l in links}
    assert ("Wikipedia:RS", 4) in got and ("Help:Talk pages", 12) in got
    assert ("Wikipedia:NPOV", 4) in got and ("Some article", 0) in got
    assert all(l["ns"] not in (2, 3, 6) for l in links)     # no user links and no embedded files


def test_pseudo_namespace_is_mainspace():
    # H: is not an official alias. It is a redirect in the article namespace (R4.5), so redirects must be resolved separately
    assert namespace_table().split("H:PICTURE").ns == 0


def test_list_items_and_templates():
    t = "Steps:\n* one\n* two\n# three\n: indent is not a list\n{{Tpw}} {{re|X}}"
    assert count_list_items(t) == 3
    assert extract_templates(t) == ["Tpw", "Re"]


# ── checks ───────────────────────────────────────────────────────────────────
def test_checks_unique_and_cutoff():
    df = pd.DataFrame({"k": [1, 2, 2], "ts": ["2026-01-01T00:00:00Z"] * 3})
    with pytest.raises(checks.InvariantError):
        checks.assert_unique(df, "k")
    late = pd.DataFrame({"ts": ["2026-04-23T00:00:00Z", "2026-04-24T00:00:00Z"]})
    with pytest.raises(checks.InvariantError):
        checks.assert_no_after_cutoff(late, "ts")


def test_checks_constant_and_duplicate_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "zero": [0, 0, 0], "b": [1, 2, 3]})
    with pytest.raises(checks.InvariantError, match="constant"):
        checks.assert_no_constant_cols(df)
    with pytest.raises(checks.InvariantError, match="identical"):
        checks.assert_no_constant_cols(df[["a", "b"]])


def test_checks_leakage():
    df = pd.DataFrame({"edit_ts": ["2024-01-01T00:00:00Z"], "q_ts": ["2024-01-01T00:00:00Z"]})
    with pytest.raises(checks.InvariantError):
        checks.assert_no_leakage(df, "edit_ts", "q_ts")


# ── io ───────────────────────────────────────────────────────────────────────
def test_io_jsonl_roundtrip_and_seed(tmp_path):
    p = tmp_path / "x.jsonl"
    assert io.write_jsonl_atomic(p, [{"a": 1}, {"名": "楊文祖"}]) == 2
    assert io.read_jsonl(p) == [{"a": 1}, {"名": "楊文祖"}]
    assert io.derive_seed(42, "m5.pseudo_anchor") == io.derive_seed(42, "m5.pseudo_anchor")
    assert io.derive_seed(42, "a") != io.derive_seed(42, "b")


# ── issues ───────────────────────────────────────────────────────────────────
def test_issue_log_writes_and_fatal(tmp_path):
    log = IssueLog("test")
    log.add("reply_before_question", rule="R5.4", action="quarantined",
            ids={"question_revid": 1215863591}, evidence="x" * 900)
    d = log.write(tmp_path)
    rec = json.loads((d / "reply_before_question.jsonl").read_text().splitlines()[0])
    assert rec["wiki_url"].endswith("oldid=1215863591") and len(rec["evidence"]) == 500
    with pytest.raises(FatalIssue):
        log.add("after_cutoff", action="fatal", ids={"revid": 1})


# ── api: cache and offline mode (no network) ──────────────────────────────────────────────────
def test_api_cache_offline(tmp_path):
    c = MediaWikiClient(online=False, cache_dir=tmp_path)
    params = {"action": "query", "list": "users", "ususers": "Nobody"}
    with pytest.raises(OfflineCacheMiss):
        c.get(params)
    # after we put an entry in the cache by hand, it can be read offline
    norm = c._normalize(params)
    path = c._cache_path(norm)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"params": norm, "response": {"ok": 1}}))
    assert c.get(params) == {"ok": 1}


# ── wikimeta ─────────────────────────────────────────────────────────────────
def test_namespace_table():
    nst = NamespaceTable.load()
    assert nst.split("WP:RS").full == "Wikipedia:RS"
    assert nst.split("wp:rs").ns == 4                      # namespace names are not case sensitive
    assert nst.split("TM:Ping").ns == 10                   # official alias: TM means Template
    assert nst.split("User_talk:Dr vulpes").ns == 3
    assert nst.split("Article#Section").full == "Article"
    assert nst.split("Talk:Foo").ns == 1
