"""
test_m2.py: M2 questions and replies (CODE_PLAN.md section 7.1).

The parsing functions are tested on a real piece of DiscussionTools output (User talk:Liu1126, cutoff-date revision
revid 1339996650, thread h-Sketpick-20260219165600, HTML shortened). The other tests read the M2 output and are
skipped automatically if M2 has not been run yet.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

import config
from pipeline.m2_conversations.build_questions import (heading_author, is_question_turn, match_thread, norm_heading,
                                                       reply_fields, strip_merged_question, thread_index, title_at)
from pipeline.m2_conversations.collect_topup import extract_section, heading_marker
from pipeline.m2_conversations.turns import heading_key, html_to_text, html_links, thread_from_wikitext, threads_from_dt

UTC = timezone.utc
DT_ITEM = {
    "headingLevel": "2", "name": "h-Sketpick-20260219165600", "type": "heading", "level": "0",
    "html": 'Question from <a rel="mw:WikiLink" href="./User:Sketpick" title="User:Sketpick">Sketpick</a> (16:56, 19 February 2026)',
    "replies": [{
        "timestamp": "2026-02-19T16:56:00Z", "author": "Sketpick", "type": "comment", "level": "1",
        "id": "c-Sketpick-20260219165600-Question_from_Sketpick_(16:56,_19_February_2026)",
        "html": '<p>Hello! In the article <a rel="mw:WikiLink" href="./List_of_World_War_I_flying_aces_from_Germany" '
                'title="List of World War I flying aces from Germany">List</a>, there are multiple aces. '
                '<a rel="mw:WikiLink" href="./User:Sketpick" title="User:Sketpick">Sketpick</a> '
                '(<a rel="mw:WikiLink" href="./User_talk:Sketpick" title="User talk:Sketpick">talk</a>) 16:56, 19 February 2026 (UTC)</p>',
        "replies": [{
            "timestamp": "2026-02-22T12:34:00Z", "author": "Liu1126", "type": "comment", "level": "2",
            "id": "c-Liu1126-20260222123400-Sketpick-20260219165600",
            "html": 'Hi @<a rel="mw:WikiLink" href="./User:Sketpick" title="User:Sketpick">Sketpick</a>, feel free to go ahead. '
                    '<a rel="mw:WikiLink" href="./User:Liu1126" title="User:Liu1126">Liu1126</a> '
                    '(<a rel="mw:WikiLink" href="./User_talk:Liu1126#top" title="User talk:Liu1126">talk</a>) 12:34, 22 February 2026 (UTC)',
            "replies": [{
                "timestamp": "2026-02-23T10:12:00Z", "author": "Sketpick", "type": "comment", "level": "3",
                "id": "c-Sketpick-20260223101200-Liu1126-20260222123400", "html": "Thank you!", "replies": []}]}]}]}


# ── turns.py ───────────────────────────────────────────────────────────────
def test_heading_key():
    assert heading_key("h-Brayden.frankcom-20251106224000") == \
        ("Brayden.frankcom", datetime(2025, 11, 6, 22, 40, tzinfo=UTC))
    assert heading_key("h-Some_user-20250101000000")[0] == "Some user"
    assert heading_key("h-Question_without_signature") == (None, None)
    # old-style key (real example: archive of the Rubbish computer talk page, revid 1136724027)
    assert heading_key("h-Rubbish_computer-2015-10-04T14:01:00.0") == \
        ("Rubbish computer", datetime(2015, 10, 4, 14, 1, tzinfo=UTC))
    assert heading_key("h-WhisperToMe-2015-07-09T19:33:00.000Z")[1] == datetime(2015, 7, 9, 19, 33, tzinfo=UTC)


def test_flatten_thread_order_and_authors():
    th = threads_from_dt([DT_ITEM], "User talk:Liu1126", 1339996650)[0]
    assert th.key_author == "Sketpick" and th.heading_text == "Question from Sketpick (16:56, 19 February 2026)"
    assert [t.author for t in th.turns] == ["Sketpick", "Liu1126", "Sketpick"]
    assert [t.level for t in th.turns] == [1, 2, 3]


def test_html_to_text_removes_signature_keeps_content():
    text, away = html_to_text(DT_ITEM["replies"][0]["html"])
    assert away is None
    assert text.startswith("Hello! In the article List, there are multiple aces.")
    assert "Sketpick" not in text and "UTC" not in text and "talk" not in text


def test_html_to_text_reads_away_note():
    html = ('<dl><dd><i>Note: <a rel="mw:WikiLink" href="./User_talk:SashankInAnitquity" title="User talk:SashankInAnitquity">'
            'SashankInAnitquity</a>\'s mentor <a rel="mw:WikiLink" href="./User_talk:Dr_vulpes" title="User talk:Dr vulpes">'
            'Dr vulpes</a> is away.</i></dd></dl><p>How do I cite a book?</p>')
    text, away = html_to_text(html)
    assert away == {"mentee": "SashankInAnitquity", "primary_mentor": "Dr vulpes"}
    assert text == "How do I cite a book?"


def test_html_links_excludes_user_links():
    out = html_links(DT_ITEM["replies"][0]["html"])
    assert [l["title"] for l in out["wikilinks"]] == ["List of World War I flying aces from Germany"]
    assert out["wikilinks"][0]["ns"] == 0


# ── build_questions.py ─────────────────────────────────────────────────────
def _q(**kw):
    base = {"question_revid": 1, "q_ts": "2026-02-19T16:56:31Z", "heading": "question from sketpick (16:56, 19 february 2026)",
            "page_mentor_uid": 7, "mentee_uid": 99}
    return {**base, **kw}


def test_match_by_minute_and_author():
    th = threads_from_dt([DT_ITEM], "User talk:Liu1126", 1339996650)[0]
    th.owner_uid = 7
    by_min, by_head = thread_index([th])
    got, info = match_thread(_q(), by_min, by_head, {"sketpick"})
    assert got is th and info["score"] == 5
    # another question by the same newcomer (different time, different title) must not be matched to this thread (R5.5)
    got, _ = match_thread(_q(q_ts="2026-03-01T10:00:00Z", heading="question from sketpick (10:00, 1 march 2026)"),
                          by_min, by_head, {"sketpick"})
    assert got is None


def test_reply_fields_respect_cutoff_and_question_time():
    ts = lambda s: datetime.fromisoformat(s).replace(tzinfo=UTC)   # noqa: E731
    turns = [{"role": "mentee", "ts": ts("2026-02-19T16:56"), "author": "A", "signer_uid": 99},
             {"role": "third_party", "ts": ts("2026-02-19T16:50"), "author": "Early", "signer_uid": None},
             {"role": "page_mentor", "ts": ts("2026-02-22T12:34"), "author": "M", "signer_uid": 7},
             {"role": "third_party", "ts": ts("2026-05-01T00:00"), "author": "Late", "signer_uid": None}]
    f = reply_fields(turns, "2026-02-19T16:56:31Z")
    assert f["has_any_reply"] and f["first_replier_name"] == "M" and f["n_reply_turns"] == 1
    assert f["has_mentor_reply"] and f["mentor_first_reply_ts"] == ts("2026-02-22T12:34")


def test_title_at_walks_moves_backwards():
    moves = [{"timestamp": "2022-07-13T10:00:00Z", "from": "User talk:Femkemilene", "to": "User talk:Femke"}]
    assert title_at("User talk:Femke", "2022-01-01T00:00:00Z", moves) == "User talk:Femkemilene"
    assert title_at("User talk:Femke", "2023-01-01T00:00:00Z", moves) == "User talk:Femke"


def test_heading_helpers():
    h = "Question from Satyaranjan sasmal on User:Satyaranjan sasmal (06:36, 19 February 2022)"
    assert norm_heading(h) == h.lower()
    assert heading_author(h) == "satyaranjan sasmal"
    pat = heading_marker(norm_heading(h))
    assert pat.search("== question from [[user:satyaranjan sasmal|x]] on [[:user:satyaranjan sasmal]] (06:36, 19 february 2022) ==")
    assert not pat.search("text 06:36, 19 february 2022 (utc)")


def test_extract_section():
    text = "== Old ==\nold\n== Question from X (10:00, 1 March 2026) ==\nHow do I?\n== Later ==\nz"
    assert extract_section(text, "Question from X (10:00, 1 March 2026)") == "How do I?"


# ── Outputs ────────────────────────────────────────────────────────────────────
M2 = config.INTERIM / "m2"
needs_m2 = pytest.mark.skipif(not (M2 / "questions.parquet").exists(), reason="M2 has not been run")


@needs_m2
def test_questions_before_cutoff_and_unique():
    q = pd.read_parquet(M2 / "questions.parquet")
    assert q.question_revid.is_unique
    assert (pd.to_datetime(q.q_ts, utc=True) < config.DATA_CUTOFF).all()
    r = q.first_reply_ts.dropna()
    assert (pd.to_datetime(r, utc=True) < config.DATA_CUTOFF).all()


@needs_m2
def test_away_routed_case():
    """revid 1343488016: the main mentor Dr vulpes left, and the question is on the talk page of 1AmNobody24 (R4.2)."""
    q = pd.read_parquet(M2 / "questions.parquet").set_index("question_revid")
    r = q.loc[1343488016]
    assert r.away_routed and r.primary_mentor_name == "Dr vulpes" and r.primary_mentor_uid != r.page_mentor_uid


@needs_m2
def test_third_party_reply_case():
    """revid 1313486198: newcomer Couscous1812 asks Sungodtemple, and the reply comes from Uthersrg (R5.3)."""
    q = pd.read_parquet(M2 / "questions.parquet").set_index("question_revid")
    r = q.loc[1313486198]
    assert r.has_any_reply and r.first_replier_role in ("third_party", "other_mentor")


def test_thread_from_wikitext_fallback():
    sec = ("How do I add a source? [[User:Newbie|Newbie]] ([[User talk:Newbie|talk]]) 10:00, 1 March 2026 (UTC)\n"
           ":Use the cite tool. [[User:Gommeh/guestbook|Gommeh]] 11:30, 1 March 2026 (UTC)\n")
    th = thread_from_wikitext(sec, "Question from Newbie (10:00, 1 March 2026)", "User talk:Gommeh", 1)
    assert [(t.author, t.level) for t in th.turns] == [("Newbie", 1), ("Gommeh", 2)]
    assert th.turns[1].text == "Use the cite tool." and th.origin == "fallback_regex"


def test_thread_from_section_html():
    """DiscussionTools comment markers in the HTML of action=parse&section=N (structure taken from User talk:Oshwah, revision 1286478866)."""
    from pipeline.m2_conversations.turns import thread_from_section_html
    hid = "h-Question_from_A_(12:50,_30_March_2025)-20250330125000"
    c1 = "c-A-20250330125000-Question_from_A_(12:50,_30_March_2025)"
    c2 = "c-Oshwah-20250403081400-A-20250330125000"
    html = (f'<h2><span data-mw-comment-start="" id="{hid}"></span>Question from A<span data-mw-comment-end="{hid}"></span></h2>'
            f'<p><span data-mw-comment-start="" id="{c1}"></span>How do I publish? A 12:50, 30 March 2025 (UTC)'
            f'<span data-mw-comment-end="{c1}"></span></p>'
            f'<dl><dd><span data-mw-comment-start="" id="{c2}"></span>Use AfC. Oshwah 08:14, 3 April 2025 (UTC)'
            f'<span data-mw-comment-end="{c2}"></span></dd></dl>')
    th = thread_from_section_html(html, "Question from A", "User talk:Oshwah", 1286478866)
    assert [(t.author, t.level) for t in th.turns] == [("A", 1), ("Oshwah", 2)]
    assert "Use AfC." in th.turns[1].html and "How do I" not in th.turns[1].html


def test_question_turn_with_mismatched_signature():
    """I-55: first comment where the signature name does not match the account name (real examples revid 1219183673, 1220075407)."""
    q_min = datetime(2024, 4, 16, 6, 45, tzinfo=UTC)
    q = "Hello, I need guidance in creating a Wikipedia page for my company name. Could you please assist me?"
    first = {"ts": q_min, "text": q + " --Rambabu Reply"}
    assert is_question_turn(first, q_min, q)
    fast_reply = {"ts": q_min, "text": "You should almost certainly not do this, see the conflict of interest guideline."}
    assert not is_question_turn(fast_reply, q_min, q)               # a real fast reply in the same minute must not be changed
    late = {"ts": datetime(2024, 4, 16, 16, 19, tzinfo=UTC), "text": q}
    assert not is_question_turn(late, q_min, q)


def test_strip_merged_question():
    """An unsigned question was merged into the first reply (real example revid 1138913383, User talk:Cullen328/Archive 94)."""
    q = "Hello, I want to add Deputy Chief Minister's name. How do I do that?"
    t = q + " 09:48, at 12 February 2023 (UTC)\n\nHello,. I do not see why it would be appropriate"
    out = strip_merged_question(t, q)
    assert out.startswith("09:48") and "Deputy" not in out
    assert strip_merged_question("A normal reply about citations.", q) is None


def test_nested_headings_are_separate_threads():
    """I-56: an archive bot grouped questions as level-3 sections under one level-2 heading that shares the key
    of its first sub-section (real case: User talk:AirshipJungleman29/Archive 10, h-Cplover13-20250603233800)."""
    c = lambda author, ts, cid: {"type": "comment", "author": author, "timestamp": ts, "level": "1", "id": cid,
                                 "html": "x", "replies": []}
    sub1 = {"type": "heading", "headingLevel": "3", "name": "h-Cplover13-20250603233800", "html": "Question from Cplover13",
            "replies": [c("Cplover13", "2025-06-03T23:38:00Z", "c-Cplover13-1")]}
    sub2 = {"type": "heading", "headingLevel": "3", "name": "h-PASC01-20250605091300", "html": "Question from PASC01",
            "replies": [c("PASC01", "2025-06-05T09:13:00Z", "c-PASC01-1")]}
    container = {"type": "heading", "headingLevel": "2", "name": "h-Cplover13-20250603233800", "html": "June 2025",
                 "replies": [sub1, sub2]}
    ths = threads_from_dt([container, sub1], "User talk:X/Archive 10", 1)   # DiscussionTools also lists sub1 at the top
    assert [[t.author for t in th.turns] for th in ths] == [["Cplover13"], ["PASC01"]]
    empty = {"type": "heading", "headingLevel": "2", "name": "h-A-20250101000000", "html": "Question from A", "replies": []}
    assert len(threads_from_dt([empty], "p", 1)) == 1                          # an empty heading with no sub-headings is kept


def test_away_note_until():
    """The away note can say "is away until <date>." (real case revid 1342874815)."""
    from pipeline.common.textclean import strip_away_note
    t = (":''Note: [[User talk:TheeCitationist|TheeCitationist]]'s mentor [[User talk:Sunny Cryolite|Sunny Cryolite]] "
         "is away until October 2026.''\nHi, asking you for a review")
    text, note = strip_away_note(t)
    assert text == "Hi, asking you for a review" and note.primary_mentor == "Sunny Cryolite"


def test_question_from_diff():
    from pipeline.m2_conversations.question_diffs import question_from_diff
    row = lambda kind, text="": {  # noqa: E731
        "ctx": f'<tr><td class="diff-marker"></td><td class="diff-context diff-side-added"><div>{text}</div></td></tr>',
        "add": f'<tr><td colspan="2" class="diff-empty"></td><td class="diff-marker" data-marker="+"></td>'
               f'<td class="diff-addedline diff-side-added"><div>{text}</div></td></tr>',
        "no": '<tr><td colspan="2" class="diff-lineno">Line 9:</td></tr>'}[kind]
    body = "".join([row("ctx", "old reply"), row("add", ""), row("add", "== Question from [[User:A|A]] (1:00, 1 May 2024) =="),
                    row("add", "== My own heading =="), row("add", "How do I cite? --[[User:A|A]] 1:00"),
                    row("ctx", "later section"), row("add", "unrelated line")])
    r = question_from_diff(body)
    assert r["heading_found"] and r["text"] == "== My own heading ==\nHow do I cite? --[[User:A|A]] 1:00"
    assert question_from_diff(row("ctx", "x")) is None
