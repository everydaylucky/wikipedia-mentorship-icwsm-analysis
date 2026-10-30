"""
build_questions.py: the processing part of M2 (offline): the question table and the multi-turn conversations

Module: M2 (PLAN.md M2; CODE_PLAN.md section 5, M2)
Rules: DATA_PROCESSING R4.1 to R4.6, R5.1 to R5.6, R6.1, R6.2; Issues: I-34, I-45

1. What it does
    1. The set of questions (R4.1): eval2's tag-based collection plus the extra tag revisions found in
       a later sweep. One row per tagged revision, keeping only revisions made before the cutoff.
    2. Talk page owner (R4.2, I-45): follow the move log backwards to find the talk page title at the
       moment of the question, then turn it into a mentor uid.
    3. Primary mentor (R4.2, R4.4): read the primary mentor from the away notice at the start of the
       question wikitext. If there is no notice, the primary mentor is the talk page owner.
    4. Entry point and the page being edited (R4.3).
    5. Finding the thread (R5.5): use the DiscussionTools thread key (asker + minute) and the heading
       text to find the thread holding this question, among all talk pages and archives as they stood
       at the cutoff. Questions not found are written to fallback_needed.csv, and
       collect_topup.py --phase fallback recovers them from revision history (R5.6).
    6. Speaker roles (R5.2): each turn is labeled mentee / page_mentor / primary_mentor / other_mentor /
       third_party / unknown. The decision uses uids: for mentors, M1's name table and alternate
       account table; for newcomers, mentee_uid, the signature name at question time, and the uids
       resolved by name in the fallback phase (this handles renames, for example AlphaBetaGamma).
    7. Reply fields (R5.3): using only turns before the cutoff, recompute has_any_reply, first_reply_ts,
       first_replier_*, has_mentor_reply, and mentor_first_reply_ts.
    8. Problem flags (R5.4, R5.6): reply_before_question, unrecoverable_revdel, first_turn_not_mentee, turn_source_regex,
       section_removed_immediately (the question was reverted or deleted in the very next edit),
       thread_not_found, ambiguous_thread, page_owner_differs, and others.
    9. Cross-check: where our own signature parsing (eval2's all_turns, cut at the cutoff) and the
       DiscussionTools turns disagree, log issue `turn_source_mismatch`; where the reply status differs
       from eval2, log `reply_status_changed` (I-34).

2. Why it is designed this way
    - A question is defined by its tagged revision, not by a section on the talk page: the link inside
      an away notice would make "find conversations by section" invent an extra one
      (DATA_PROCESSING section 4, "Processing errors that may occur").
    - Turns come from DiscussionTools: it is the official parser behind Wikipedia's "Reply" feature,
      and it gives the author and time of each turn directly (R5.1).
    - Threads are not matched by heading text alone: when the same newcomer asks the same mentor
      several times, the headings can be identical (R5.5). The time in the thread key comes from the
      first signature, which falls in the same minute as the tagged revision (the signature is made at
      save time). So we match on "minute ±2" plus a matching author or heading.
    - Freezing at the cutoff: DiscussionTools parses the revision at the cutoff moment, so it already
      has no turns after the cutoff. Here we cut again by turn time, in case a signature time is wrong.

3. Inputs
    config.EXTERNAL["eval2_conversations"] (eval2's tag-based collection and its own signature splitting, 2026-06-12);
    data/raw/m2/{tag_topup,move_log,pages}.jsonl, dt/, fallback_dt/, fallback_wikitext/, fallback_map.json,
    question_wikitext.json, replier_uids.json (made in the fallback phase), removal_edits.json (removals phase); skipped if missing;
    data/interim/m1/{mentor_names,mentor_periods}.parquet.

4. Outputs (data/interim/m2/)
    questions.parquet   one row per question, primary key question_revid (see the notes on QUESTION_COLUMNS for the fields)
    turns.jsonl         one row per question: question_revid and all turns [{author, signer_uid, role, ts, level, text, comment_id, links}]
                        links = the result of turns.html_links (internal links with namespaces, number of external links, number of list items), used by M6 reply features
    fallback_needed.csv questions whose thread was not found and must be recovered from revision history
    names_to_resolve.csv signature names of speakers who are neither mentors nor known newcomer names (uids resolved in the fallback phase)
    _issues/            all kinds of problems (see section 8)

5. Key parameters
    MATCH_MINUTES = 2: how many minutes the thread key time may differ from the tagged revision time.

6. Possible errors and how they are handled
    - The same revid appears twice: keep one, log issue `duplicate_revid`;
    - The revision is hidden (empty question text): flag unrecoverable_revdel; reply fields are still computed from the thread;
    - The thread is not found: flag thread_not_found and send it to fallback; if it is still not found after fallback, the reply fields are set to missing;
    - The talk page owner differs from the mentor eval2 recorded: trust the result rebuilt from the move log, log issue `page_owner_differs`;
    - A reply is earlier than the question: flag reply_before_question, leave it out of the reply fields, and check each one by hand in the notebook (R5.4).

7. How to run
    python -m pipeline.m2_conversations.build_questions
    After the first run, run collect_topup --online --phase fallback, then run this file once more.

8. How to check
    checks/m2_check.ipynb; tests/test_m2.py.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from datetime import timedelta

import pandas as pd

import config
from pipeline.common.checks import assert_no_after_cutoff, assert_unique
from pipeline.common.io import iter_jsonl, write_jsonl_atomic, write_table
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_sources, check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.textclean import strip_away_note, to_plain_text
from pipeline.common.timeutil import before_cutoff, parse_iso, to_iso
from pipeline.common.users import UserIndex, canonical_person, norm_user
from pipeline.m2_conversations.turns import (Thread, html_links, html_to_text, thread_from_section_html,
                                             thread_from_wikitext, threads_from_dt)

RAW = config.RAW / "m2"
OUT = config.INTERIM / "m2"
MATCH_MINUTES = 2
MENTOR_ROLES = ("page_mentor", "primary_mentor")
_HEAD_TS_RE = re.compile(r"\((\d{1,2}:\d{2}, \d{1,2} [A-Za-z]+ \d{4})\)\s*$")
_HEAD_AUTHOR_RE = re.compile(r"^question from (.+?)(?: on .+)? \(\d{1,2}:\d{2}, ", re.I)
_COMMENT_SECTION_RE = re.compile(r"^/\*\s*(.+?)\s*\*/")


# ── Small helpers ──────────────────────────────────────────────────────────────────
def norm_heading(h: str | None) -> str:
    """Normalize a heading: lower case, underscores to spaces, collapse whitespace. eval2 headings are already lower case."""
    return re.sub(r"\s+", " ", (h or "").replace("_", " ")).strip().lower()


def heading_author(h: str | None) -> str | None:
    """Get the asker (lower case) from a heading "Question from X (on Y) (hh:mm, d Month yyyy)"."""
    m = _HEAD_AUTHOR_RE.match(norm_heading(h))
    return m.group(1) if m else None


def lower_name(n: str | None) -> str | None:
    """The form used to compare names: normalized and lower case (eval2 headings only exist in lower case)."""
    n = norm_user(n)
    return n.lower() if n else None


# ── 1. The set of questions ─────────────────────────────────────────────────────────────
def load_base(log, issues: IssueLog, use_diff: bool = True) -> pd.DataFrame:
    """Read eval2's questions and the top-up sweep, and merge them into one row per tagged revision (R4.1). Keep only those before the cutoff.

    Returned columns: question_revid, parentid, q_ts, page_eval, eval_mentor_uid, mentee_uid, mentee_name,
             heading, question_wikitext, tags, tag, source, eval_turns, eval_has_any_reply
    """
    rows = []
    for r in iter_jsonl(config.EXTERNAL["eval2_conversations"]):
        tags = r.get("tags") or []
        rows.append({"question_revid": int(r["revid"]), "parentid": r.get("parentid"), "q_ts": r["timestamp"],
                     "page_eval": r["page"], "eval_mentor_uid": r.get("mentor_uid"),
                     "mentee_uid": r.get("mentee_uid"), "mentee_name": norm_user(r.get("mentee")),
                     "heading": r.get("heading"), "question_wikitext": r.get("question_text") or "",
                     "tags": list(tags), "tag": next((t for t in config.QUESTION_TAGS if t in tags), None),
                     "source": "eval2", "eval_turns": r.get("all_turns") or [],
                     "eval_has_any_reply": str(r.get("has_any_reply")) == "True"})
    n_eval = len(rows)
    wt_path = RAW / "question_wikitext.json"
    wikitext = json.loads(wt_path.read_text()) if wt_path.exists() else {}
    for r in iter_jsonl(RAW / "tag_topup.jsonl"):
        m = _COMMENT_SECTION_RE.match(r.get("comment") or "")
        rows.append({"question_revid": int(r["revid"]), "parentid": r.get("parentid"), "q_ts": r["timestamp"],
                     "page_eval": r["page"], "eval_mentor_uid": r.get("mentor_uid"),
                     "mentee_uid": r.get("userid"), "mentee_name": norm_user(r.get("user")),
                     "heading": norm_heading(m.group(1)) if m else None,
                     "question_wikitext": (wikitext.get(str(r["revid"])) or {}).get("text") or "",
                     "tags": r.get("tags") or [], "tag": r.get("tag"), "source": "topup",
                     "eval_turns": [], "eval_has_any_reply": None, "texthidden": "texthidden" in r})
    df = pd.DataFrame(rows)
    # Question text as posted (I-60): for the questions whose collected text is obviously broken (see
    # question_diffs.suspects), the lines added by the tagged revision itself replace the collected text.
    diff_path = RAW / "question_text_from_diff.json"
    from_diff = json.loads(diff_path.read_text()) if diff_path.exists() and use_diff else {}
    dtext = df.question_revid.map(lambda rv: (from_diff.get(str(rv)) or {}).get("text") or "")
    use = dtext.str.strip() != ""
    df["question_wikitext_origin"] = use.map({True: "diff", False: "collected"})
    df.loc[use, "question_wikitext"] = dtext[use]
    log.info("question wikitext from the tagged revision's diff: %d of %d", int(use.sum()), len(df))
    if "texthidden" not in df:
        df["texthidden"] = False
    df["texthidden"] = df["texthidden"].fillna(False).astype(bool)
    log.info("questions: eval2 %d + top-up %d", n_eval, len(df) - n_eval)

    after = ~df.q_ts.map(before_cutoff)
    log.info("dropping %d questions at or after the cutoff %s", int(after.sum()), config.DATA_CUTOFF_ISO)
    df = df[~after]
    early = df.q_ts.map(lambda t: parse_iso(t) < config.MENTORSHIP_START)
    if early.any():
        raise RuntimeError(f"{int(early.sum())} tagged questions before mentorship launch {config.MENTORSHIP_START}: "
                           f"{df[early].question_revid.head().tolist()}")
    dup = df.question_revid.duplicated(keep="first")
    for r in df[dup].itertuples():
        issues.add("duplicate_revid", rule="R4.1", action="fixed", ids={"revid": r.question_revid},
                   detail={"source": r.source, "page": r.page_eval})
    df = df[~dup].reset_index(drop=True)
    for c in ("eval_mentor_uid", "mentee_uid", "parentid"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


# ── 2. Mentor index and talk page owner ─────────────────────────────────────────────────
def mentor_index() -> tuple[UserIndex, dict[int, list[tuple]]]:
    """A mentor name → uid index (not limited by time, since no mentor name was ever reused by someone else); and the time spans each mentor was on the roster."""
    names = pd.read_parquet(config.INTERIM / "m1" / "mentor_names.parquet")
    idx = UserIndex()
    for r in names.itertuples():
        idx.add(int(r.mentor_uid), r.name, source=r.source)
    for alt, main in config.MENTOR_ALT_ACCOUNTS.items():
        uid = idx.lookup(main)
        if uid is not None:
            idx.add(uid, alt, source="alt_account")
    per = pd.read_parquet(config.INTERIM / "m1" / "mentor_periods.parquet")
    spans = defaultdict(list)
    for r in per.itertuples():
        spans[int(r.mentor_uid)].append((parse_iso(r.start), parse_iso(r.end)))
    return idx, dict(spans)


def on_roster(spans: dict, uid: int | None, at) -> bool:
    """Is this uid on the mentor roster at time `at` (using M1's status periods)?"""
    at = parse_iso(at)
    return uid is not None and any(s <= at and (e is None or at < e) for s, e in spans.get(uid, []))


def title_at(title: str, at, moves: list[dict]) -> str:
    """Follow the move log backwards: what was the page now called `title` called at time `at` (I-45)?

    Args:
        moves: move records {timestamp, from, to}, sorted by time.
    """
    at = parse_iso(at)
    cur = title
    for m in sorted(moves, key=lambda m: m["timestamp"], reverse=True):
        if parse_iso(m["timestamp"]) > at and m.get("to") and norm_user(m["to"]) == norm_user(cur):
            cur = m["from"]
    return cur


def page_base(title: str) -> str:
    """Example: "User talk:X/Archive 3" → "X"."""
    return norm_user(title.split(":", 1)[-1].split("/", 1)[0])


# ── 3. Threads ──────────────────────────────────────────────────────────────
def load_threads(log, issues: IssueLog, midx: UserIndex) -> list[Thread]:
    """Read the DiscussionTools result for each page at the cutoff moment, plus the threads recovered by fallback.

    Each thread also records owner_uid (the mentor who owns the page) and origin ("cutoff" or "fallback").
    """
    threads = []
    for p in iter_jsonl(RAW / "pages.jsonl"):
        if not p.get("exists") or not p.get("as_of_revid") or p.get("skipped"):
            continue
        path = RAW / "dt" / f"{p['as_of_revid']}.json"
        if not path.exists():
            issues.add("dt_missing", rule="R5.1", action="flagged", ids={"page": p["title"]},
                       detail={"revid": p["as_of_revid"]})
            continue
        for th in threads_from_dt(json.loads(path.read_text()), p["title"], p["as_of_revid"]):
            th.owner_uid = midx.lookup(page_base(p["title"]))
            th.origin = "cutoff"
            threads.append(th)
    fb_map_path = RAW / "fallback_map.json"
    fb_map = json.loads(fb_map_path.read_text()) if fb_map_path.exists() else {}
    for qrev, info in fb_map.items():
        path = RAW / "fallback_dt" / f"{info['revid']}.json"
        if not info.get("revid") or not path.exists():
            continue
        for th in threads_from_dt(json.loads(path.read_text()), info["page"], info["revid"]):
            th.owner_uid = midx.lookup(page_base(info["page"]))
            th.origin = "fallback"
            th.for_question = int(qrev)
            threads.append(th)
    for path in sorted((RAW / "fallback_sections").glob("*.json")) if (RAW / "fallback_sections").exists() else []:
        info = json.loads(path.read_text())
        page = fb_map.get(path.stem, {}).get("page", "")
        th = thread_from_section_html(info["html"], info["line"], page, info["revid"])
        th.owner_uid = midx.lookup(page_base(page)) if page else None
        th.for_question = int(path.stem)
        threads.append(th)
    for path in sorted((RAW / "fallback_wikitext").glob("*.json")) if (RAW / "fallback_wikitext").exists() else []:
        info = json.loads(path.read_text())
        th = thread_from_wikitext(info["text"], info["heading"], info["page"], info["revid"])
        th.owner_uid = midx.lookup(page_base(info["page"]))
        th.for_question = int(path.stem)
        threads.append(th)
    log.info("threads: %d (%d from fallback revisions)", len(threads),
             sum(t.origin != "cutoff" for t in threads))
    return threads


def thread_index(threads: list[Thread]) -> tuple[dict, dict]:
    """Two indexes: by the minute in the thread key, and by the normalized heading."""
    by_min, by_head = defaultdict(list), defaultdict(list)
    for th in threads:
        if th.key_ts is not None:
            by_min[th.key_ts.replace(second=0)].append(th)
        by_head[norm_heading(th.heading_text)].append(th)
    return by_min, by_head


def match_thread(q: dict, by_min: dict, by_head: dict, mentee_lower: set[str]) -> tuple[Thread | None, dict]:
    """Find the thread for one question (R5.5).

    Candidates: threads whose key time is within ±MATCH_MINUTES minutes of the question time, or whose
    heading matches exactly.
    Scoring: matching heading +2; the asker in the key is this newcomer +2; the page owner is the mentor
         who received the question +1; the fallback thread was recovered for exactly this question +1.
         A match needs at least 2 points. On a tie, take the thread with more turns (when the same thread
         exists both in the archive page at the cutoff and in an earlier recovered revision, the one with
         more turns is the later, more complete copy).

    Returns:
        (thread or None, match info {"score", "n_candidates", "tied"})
    """
    t = parse_iso(q["q_ts"]).replace(second=0)
    cands = {}
    for k in range(-MATCH_MINUTES, MATCH_MINUTES + 1):
        for th in by_min.get(t + timedelta(minutes=k), []):
            cands[id(th)] = th
    for th in by_head.get(norm_heading(q["heading"]), []) if q.get("heading") else []:
        cands[id(th)] = th
    scored = []
    for th in cands.values():
        s = 0
        if q.get("heading") and norm_heading(th.heading_text) == norm_heading(q["heading"]):
            s += 2
        if lower_name(th.key_author) in mentee_lower:
            s += 2
        if th.owner_uid is not None and th.owner_uid == q.get("page_mentor_uid"):
            s += 1
        if th.for_question == q["question_revid"]:
            s += 1
        if s >= 2:
            scored.append((s, len(th.turns), th))
    if not scored:
        return None, {"score": 0, "n_candidates": len(cands), "tied": False}
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0]
    # Two snapshots of the same thread at different times (for example the archive page at the cutoff and
    # an earlier recovered revision) are not ambiguous: the author sequence of the copy with fewer turns is
    # a prefix of the other. It only counts as ambiguous when neither author sequence is a prefix of the other.
    def same_thread(a: Thread, b: Thread) -> bool:
        x, y = [t.author for t in a.turns], [t.author for t in b.turns]
        n = min(len(x), len(y))
        return x[:n] == y[:n]
    tied = any(o[0] == best[0] and norm_heading(o[2].heading_text) == norm_heading(best[2].heading_text)
               and not same_thread(o[2], best[2]) for o in scored[1:])
    return best[2], {"score": best[0], "n_candidates": len(cands), "tied": tied}


# ── 4. Speaker roles and reply fields ───────────────────────────────────────────────────────
def assign_role(author: str | None, at, q: dict, midx: UserIndex, spans: dict, resolved: dict) -> tuple[str, int | None]:
    """The speaker role of one turn (R5.2).

    Order: no author → unknown; the name belongs to this newcomer (any known name, or the resolved uid
         equals mentee_uid) → mentee; mentor uid equals the talk page owner → page_mentor; equals the
         primary mentor → primary_mentor; on the mentor roster at the time of the turn → other_mentor;
         anything else → third_party.
    Returns:
        (role, signer_uid or None)
    """
    if not author:
        return "unknown", None
    person = canonical_person(author)
    uid = midx.lookup(person)
    if uid is None:
        uid = resolved.get(person)
    if (lower_name(person) in q["mentee_lower"]) or (uid is not None and uid == q.get("mentee_uid")):
        return "mentee", q.get("mentee_uid")
    if uid is not None and uid == q.get("page_mentor_uid"):
        return "page_mentor", uid
    if uid is not None and uid == q.get("primary_mentor_uid"):
        return "primary_mentor", uid
    if midx.lookup(person) is not None and on_roster(spans, uid, at):
        return "other_mentor", uid
    return "third_party", uid


def reply_fields(turns: list[dict], q_ts) -> dict:
    """Recompute the reply fields (R5.3) using only turns after the question and before the cutoff.

    A reply = a turn after the question by anyone other than the newcomer (including unknown). The
    question time is rounded down to the minute (signatures only go to the minute).
    """
    q_min = parse_iso(q_ts).replace(second=0)
    rep = [t for t in turns if t["role"] != "mentee" and t["ts"] is not None
           and t["ts"] >= q_min and before_cutoff(t["ts"])]
    men = [t for t in rep if t["role"] in MENTOR_ROLES]
    first = rep[0] if rep else None
    return {"has_any_reply": bool(rep), "first_reply_ts": first["ts"] if first else None,
            "first_replier_name": first["author"] if first else None,
            "first_replier_uid": first["signer_uid"] if first else None,
            "first_replier_role": first["role"] if first else None,
            "has_mentor_reply": bool(men), "mentor_first_reply_ts": men[0]["ts"] if men else None,
            "n_reply_turns": len(rep), "n_mentor_turns": len(men),
            "n_third_party_turns": sum(t["role"] == "third_party" for t in rep),
            "n_mentee_turns": sum(t["role"] == "mentee" for t in turns)}


_WORD = re.compile(r"\w+")


def _tokens(s: str | None) -> list[str]:
    """Sequence of lower-case words (used to compare turn text with question text)."""
    return _WORD.findall((s or "").lower())


def is_question_turn(turn: dict, q_min, q_plain: str) -> bool:
    """Is the first turn of the thread the question itself (addition to R5.2, I-55)?

    Conditions: the turn time is between the question minute and MATCH_MINUTES minutes later, and the
    text is basically the same as the question text (the overlap of the two word sets is at least 80%
    of the smaller set). When the signature does not match the account name (HTML escaping, signing with
    another account, an abbreviated signature, or no signature so the text is credited to the next
    signer), the role check would treat the question itself as someone else's reply. This turn comes
    from the question revision, so its author is the newcomer; the revision author is trusted.
    """
    if turn["ts"] is None or not (q_min <= turn["ts"] <= q_min + timedelta(minutes=MATCH_MINUTES)):
        return False
    a, b = set(_tokens(turn["text"])), set(_tokens(q_plain))
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= 0.8


def strip_merged_question(text: str, q_plain: str) -> str | None:
    """When an unsigned question was merged into the first reply after it (the reply text starts with the question text), remove the question part. Otherwise return None."""
    qt, tt = _tokens(q_plain), _tokens(text)
    if len(qt) < 3 or len(tt) <= len(qt) or tt[:len(qt)] != qt:
        return None
    # find where the last word of the question is in the original text, and cut after it
    pos = 0
    for w in qt:
        m = re.compile(re.escape(w), re.I).search(text, pos)
        if not m:
            return None
        pos = m.end()
    return text[pos:].lstrip(" .,:;!?-\n")


def removal_kind(edit: dict | None, mentee_uid, mentee_name: str | None) -> str | None:
    """When a question was deleted in the very next edit, the reason for the deletion (only the metadata of that edit is used; collect_topup --phase removals).

    revert: the edit has the tag mw-rollback, mw-undo, or mw-manual-revert, or its summary starts with Reverted/Undid (a revert);
    self_removed: the editor is the newcomer; merged: the summary mentions combine/merge/moved (a duplicate post was merged or moved elsewhere);
    other: anything else.
    """
    if not edit:
        return None
    tags = set(edit.get("tags") or [])
    comment = (edit.get("comment") or "").lower()
    if (pd.notna(mentee_uid) and edit.get("userid") == mentee_uid) or norm_user(edit.get("user")) == mentee_name:
        return "self_removed"
    if tags & {"mw-rollback", "mw-undo", "mw-manual-revert"} or comment.startswith(("reverted", "undid", "rv")):
        return "revert"
    if re.search(r"combin|merg|moved|move to|duplicate", comment):
        return "merged"
    return "other"


def entry_point(tag: str | None, heading: str | None, thread: Thread | None) -> tuple[bool, str | None]:
    """R4.3: was the newcomer editing a page when asking, and what is that page's title?

    Condition: the tag is mentorship panel question, and the heading is "Question from X on <page> (<time>)".
    The page title is taken first from the second internal link in the DiscussionTools heading HTML
    (the first link is the asker).
    """
    if tag != "mentorship panel question":
        return False, None
    if thread is not None:
        from bs4 import BeautifulSoup
        links = [a.get("title") for a in BeautifulSoup(thread.heading_html, "lxml").find_all("a")
                 if a.get("title")]
        if len(links) >= 2 and " on " in thread.heading_text:
            return True, links[1]
    m = re.match(r"^question from .+? on (.+) \(\d{1,2}:\d{2}, ", norm_heading(heading))
    return (True, m.group(1)) if m else (False, None)


# ── 5. Cross-check with eval2 ──────────────────────────────────────────────────────────
def compare_turn_sources(q: dict, dt_turns: list[dict], issues: IssueLog) -> bool:
    """Compare the turns from eval2's own signature splitting (cut at the cutoff) with the DiscussionTools turns.

    What is compared is the "sequence of signers" (normalized, alternate accounts mapped to the main
    account). If they differ, log issue `turn_source_mismatch` and return True.
    """
    old = [canonical_person(t.get("signer_name")) for t in q["eval_turns"]
           if t.get("ts") and before_cutoff(t["ts"])]
    new = [canonical_person(t["author"]) for t in dt_turns]
    if not q["eval_turns"] or old == new:
        return False
    issues.add("turn_source_mismatch", rule="R5.1", action="flagged", ids={"revid": q["question_revid"]},
               detail={"regex": old[:12], "dt": new[:12]}, url=f"https://en.wikipedia.org/w/index.php?oldid={q['question_revid']}")
    return True


# ── Main flow ─────────────────────────────────────────────────────────────────
def build(log, issues: IssueLog) -> tuple[pd.DataFrame, list[dict]]:
    """Build the question table and the turn table (steps 1 to 9 in section 1)."""
    base = load_base(log, issues)
    midx, spans = mentor_index()
    moves = list(iter_jsonl(RAW / "move_log.jsonl"))
    resolved_path = RAW / "replier_uids.json"
    resolved = {norm_user(k): v for k, v in json.loads(resolved_path.read_text()).items() if v} \
        if resolved_path.exists() else {}
    removals_path = RAW / "removal_edits.json"
    removals = json.loads(removals_path.read_text()) if removals_path.exists() else {}
    threads = load_threads(log, issues, midx)
    by_min, by_head = thread_index(threads)

    rows, turn_rows, fallback, unresolved = [], [], [], defaultdict(int)
    for q in base.to_dict("records"):
        flags = []
        # talk page owner (I-45)
        t_at = title_at(q["page_eval"], q["q_ts"], moves)
        q["page_title_at_q"] = t_at
        owner = midx.lookup(page_base(t_at))
        q["page_mentor_uid"] = owner if owner is not None else q["eval_mentor_uid"]
        if owner is None:
            flags.append("page_owner_unresolved")
        elif pd.notna(q["eval_mentor_uid"]) and owner != q["eval_mentor_uid"]:
            flags.append("page_owner_differs")
            issues.add("page_owner_differs", rule="I-45", action="fixed", ids={"revid": q["question_revid"]},
                       detail={"page_eval": q["page_eval"], "title_at_q": t_at, "eval_mentor_uid": int(q["eval_mentor_uid"]),
                               "owner_uid": owner})
        # away notice and primary mentor (R4.2, R4.4)
        text, away = strip_away_note(q["question_wikitext"])
        q["away_routed"] = away is not None
        q["primary_mentor_name"] = away.primary_mentor if away else None
        q["primary_mentor_uid"] = midx.lookup(away.primary_mentor) if away else q["page_mentor_uid"]
        if away and q["primary_mentor_uid"] is None:
            flags.append("primary_mentor_unresolved")
        # known names of the newcomer (compared in lower case): the revision author, the asker in the heading, the newcomer in the away notice, and names resolved to the same uid
        ml = {lower_name(q["mentee_name"]), heading_author(q["heading"])}
        if away:
            ml.add(lower_name(away.mentee))
        ml |= {lower_name(n) for n, u in resolved.items() if pd.notna(q["mentee_uid"]) and u == q["mentee_uid"]}
        q["mentee_lower"] = {x for x in ml if x}
        if q["texthidden"] or (q["source"] == "eval2" and not q["question_wikitext"]):
            flags.append("unrecoverable_revdel")

        th, info = match_thread(q, by_min, by_head, q["mentee_lower"])
        turns = []
        if th is None:
            flags.append("thread_not_found")
            fallback.append({"question_revid": q["question_revid"], "page": q["page_eval"], "q_ts": q["q_ts"],
                             "heading": q["heading"]})
        else:
            if info["tied"]:
                flags.append("ambiguous_thread")
            if th.origin == "fallback_regex":
                flags.append("turn_source_regex")
            if th.origin != "cutoff" and th.as_of_revid == q["question_revid"]:
                # the section was deleted in the first edit after the question (usually a revert or undo)
                flags.append("section_removed_immediately")
            for t in th.turns:
                role, uid = assign_role(t.author, t.ts, q, midx, spans, resolved)
                text_t = t.text if t.text is not None else html_to_text(t.html)[0]
                turns.append({"author": t.author, "signer_uid": uid, "role": role, "ts": t.ts, "level": t.level,
                              "comment_id": t.comment_id, "text": text_t,
                              "links": html_links(t.html) if t.html else None})
                if role == "third_party" and uid is None and t.author:
                    unresolved[canonical_person(t.author)] += 1
            late = [t for t in turns if t["ts"] is not None and not before_cutoff(t["ts"])]
            if late:
                flags.append("turn_after_cutoff_dropped")
                turns = [t for t in turns if t not in late]
            q_min = parse_iso(q["q_ts"]).replace(second=0)
            q_plain = to_plain_text(text)
            if turns and turns[0]["role"] != "mentee" and is_question_turn(turns[0], q_min, q_plain):
                # the first turn is the question itself, but the signature name does not match (I-55): relabel it as the newcomer, and treat this signature name as the newcomer for the rest of this thread
                flags.append("question_turn_relabelled")
                issues.add("question_turn_relabelled", rule="R5.2", action="fixed", ids={"revid": q["question_revid"]},
                           detail={"signature": turns[0]["author"], "role_was": turns[0]["role"],
                                   "mentee_name": q["mentee_name"]})
                if turns[0]["author"]:
                    q["mentee_lower"] = q["mentee_lower"] | {lower_name(canonical_person(turns[0]["author"]))}
                for t in turns:
                    if lower_name(canonical_person(t["author"] or "")) in q["mentee_lower"] or t is turns[0]:
                        t["role"], t["signer_uid"] = "mentee", q.get("mentee_uid")
            elif turns and turns[0]["role"] != "mentee":
                cut = strip_merged_question(turns[0]["text"] or "", q_plain)
                if cut is not None:
                    # an unsigned question was merged into the first reply (for example in Cullen328's archive): keep the reply time and author, and only remove the question text
                    flags.append("question_merged_into_reply")
                    turns[0]["text"] = cut
            if turns and turns[0]["role"] != "mentee":
                flags.append("first_turn_not_mentee")
            if any(t["role"] != "mentee" and t["ts"] is not None and t["ts"] < q_min for t in turns):
                flags.append("reply_before_question")
            if compare_turn_sources(q, turns, issues):
                flags.append("turn_source_mismatch")

        fields = reply_fields(turns, q["q_ts"]) if th is not None else {
            k: None for k in ("has_any_reply", "first_reply_ts", "first_replier_name", "first_replier_uid",
                              "first_replier_role", "has_mentor_reply", "mentor_first_reply_ts", "n_reply_turns",
                              "n_mentor_turns", "n_third_party_turns", "n_mentee_turns")}
        editing, article = entry_point(q["tag"], q["heading"], th)
        q_dt_text = turns[0]["text"] if turns and turns[0]["role"] == "mentee" else None
        if th is not None and q["eval_has_any_reply"] is not None and fields["has_any_reply"] != q["eval_has_any_reply"]:
            issues.add("reply_status_changed", rule="I-34", action="flagged", ids={"revid": q["question_revid"]},
                       detail={"eval2": q["eval_has_any_reply"], "new": fields["has_any_reply"],
                               "first_replier": fields["first_replier_name"], "role": fields["first_replier_role"]})
        for f in flags:
            if f not in ("page_owner_differs", "turn_source_mismatch"):
                issues.add(f, rule="R5", action="flagged", ids={"revid": q["question_revid"]},
                           detail={"page": q["page_eval"], "q_ts": q["q_ts"], "heading": q["heading"]},
                           url=f"https://en.wikipedia.org/w/index.php?oldid={q['question_revid']}")
        rows.append({
            "question_revid": q["question_revid"], "parentid": q["parentid"], "q_ts": q["q_ts"],
            "mentee_uid": q["mentee_uid"], "mentee_name": q["mentee_name"], "tag": q["tag"], "source": q["source"],
            "heading": q["heading"], "page_eval": q["page_eval"], "page_title_at_q": q["page_title_at_q"],
            "page_mentor_uid": q["page_mentor_uid"], "eval_mentor_uid": q["eval_mentor_uid"],
            "primary_mentor_uid": q["primary_mentor_uid"], "primary_mentor_name": q["primary_mentor_name"],
            "away_routed": q["away_routed"], "editing_page": editing, "article_context": article,
            "question_wikitext": text, "question_text": to_plain_text(text) or (q_dt_text or ""),
            "question_text_source": "wikitext" if to_plain_text(text) else ("discussiontools" if q_dt_text else "none"),
            "question_text_dt": q_dt_text, "question_wikitext_origin": q["question_wikitext_origin"],
            "thread_page": th.page if th else None, "thread_revid": th.as_of_revid if th else None,
            "thread_origin": th.origin if th else None, "match_score": info["score"],
            "n_turns": len(turns), **fields,
            "first_reply_ts": to_iso(fields["first_reply_ts"]), "mentor_first_reply_ts": to_iso(fields["mentor_first_reply_ts"]),
            "removal_kind": removal_kind(removals.get(str(q["question_revid"])), q["mentee_uid"], q["mentee_name"])
            if "section_removed_immediately" in flags else None,
            "eval_has_any_reply": q["eval_has_any_reply"], "parse_flags": flags})
        turn_rows.append({"question_revid": q["question_revid"],
                          "turns": [{**t, "ts": to_iso(t["ts"])} for t in turns]})

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fallback, columns=["question_revid", "page", "q_ts", "heading"]).to_csv(OUT / "fallback_needed.csv", index=False)
    pd.DataFrame(sorted(unresolved.items(), key=lambda x: -x[1]), columns=["name", "n_turns"]).to_csv(
        OUT / "names_to_resolve.csv", index=False)
    log.info("thread not found: %d (written to fallback_needed.csv); unresolved replier names: %d",
             len(fallback), len(unresolved))
    return df, turn_rows


def main(argv: list[str] | None = None) -> None:
    args = base_parser("M2 build: questions and turns as of the cutoff.").parse_args(argv)
    log = setup_logging("m2_build")
    t0 = time.time()
    check_upstream("m2_build", ["m1"])
    check_sources(["eval2_conversations"])
    issues = IssueLog("m2")
    df, turn_rows = build(log, issues)
    if args.limit:
        df = df.head(args.limit)
    assert_unique(df, "question_revid")
    assert_no_after_cutoff(df, "q_ts")
    assert_no_after_cutoff(df.dropna(subset=["first_reply_ts"]), "first_reply_ts")
    write_table(OUT / "questions.parquet", df)
    write_jsonl_atomic(OUT / "turns.jsonl", turn_rows)
    issues.write()
    found = df.thread_page.notna()
    counts = {"questions": len(df), "thread_found": int(found.sum()),
              "has_any_reply": int(df.has_any_reply.fillna(False).sum()),
              "has_mentor_reply": int(df.has_mentor_reply.fillna(False).sum()),
              "away_routed": int(df.away_routed.sum()), "editing_page": int(df.editing_page.sum()),
              "first_replier_role": df.first_replier_role.value_counts().to_dict()}
    log.info("done: %s", counts)
    write_manifest("m2", inputs=[config.EXTERNAL["eval2_conversations"], RAW / "tag_topup.jsonl", RAW / "pages.jsonl"],
                   outputs=[OUT / "questions.parquet", OUT / "turns.jsonl"], params={"match_minutes": MATCH_MINUTES},
                   counts=counts, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
