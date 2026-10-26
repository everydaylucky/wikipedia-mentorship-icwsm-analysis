"""
collect_topup.py: the data collection part of M2 (online, run only once)

Module: M2 (PLAN.md M2; CODE_PLAN.md section 5, M2)
Rules: DATA_PROCESSING R4.1 (questions are found by tag), R5.1 (DiscussionTools splits the talk into turns), I-45 (talk page moves), I-47

1. What it does
    Phase pages (the default):
        1. Scans for tags on the talk pages of mentors that were never scanned before (currently 18 mentors,
           30 questions in total before the cutoff date);
        2. Fetches the talk page move log for every name of every mentor (used to work out who owned the talk page
           at the time of the question, I-45);
        3. Lists each mentor's talk page and all its archive subpages. We parse only the main talk page, plus
           subpages that were edited after 2021-05-04 and whose text contains "Question from" (filtered with full-text
           search, see subpages_with_questions);
        4. For each page, takes the last revision before the cutoff date and parses it with DiscussionTools (R5.1).
           This step also keeps "data frozen at the cutoff date": we parse each page exactly as it was at the cutoff moment.
    Phase fallback (after build_questions.py has run once):
        5. Gets the wikitext of the newly scanned questions: takes the section out of the page content of the
           question revision (mwparserfromhell).
        6. For questions whose thread cannot be found in the cutoff version (the section was deleted, or archived
           somewhere else, before the cutoff), we first use DiscussionTools comment tracking
           (discussiontoolsfindcomment) to find "the last revision that still contains this thread".
           If there is no record, we binary-search the page revisions "after the question and before the cutoff"
           for the last revision that still contains the section.
           Then we parse that revision with DiscussionTools (R5.6). This way we recover replies made before the
           cutoff and before the section was deleted.
        7. For signature names in the turns whose identity is unknown, we find the uid with list=users and the
           rename log (R6.3).

2. Why it is designed this way
    - The set of questions is defined by tag (R4.1). The existing tag-based data (eval2, 2026-06-12) lacks only
      18 mentors, so we scan only these 18 and skip the rest.
    - The author and time of each turn come from Wikipedia's official DiscussionTools parser (R5.1),
      not from our own signature regex. DiscussionTools gives each thread a key
      "h-<asker>-<YYYYMMDDHHMMSS>". We use it to match threads to questions, which is more reliable than matching
      by heading text (R5.5).
    - Sections archived after the cutoff date were still on the main talk page at the cutoff moment; sections
      archived before the cutoff are on the archive pages. So "main talk page + all archive pages" as they were at
      the cutoff moment, taken together, cover all discussion before the cutoff.
    - All requests go through the MediaWikiClient cache, so after an interruption a rerun continues from the cache
      (resume after a stop).

3. Inputs
    data/interim/m1/mentor_list.parquet, mentor_names.parquet (M1);
    config.EXTERNAL["eval2_tagged_raw"], ["eval2_conversations"], ["tag_scan_checkpoint"].

4. Outputs (data/raw/m2/)
    tag_topup.jsonl         tagged revisions found by the new scan (only those before the cutoff date)
    move_log.jsonl          talk page move log
    pages.jsonl             list of pages to parse: title, base, as_of_revid, as_of_ts, exists
    dt/<revid>.json         DiscussionTools result for the cutoff version of each page (another copy is in the API cache)
    fallback_dt/<revid>.json  fallback phase: DiscussionTools result for the last revision before the section was deleted
    fallback_map.json       question revid -> the revision used to recover it {revid, timestamp, page, probes}
    fallback_sections/<question revid>.json  parsed HTML of the section holding the question (with DiscussionTools turn markers)
    fallback_wikitext/<question revid>.json  when even the section HTML cannot be fetched, the section wikitext (split by signatures as a last resort)
    question_wikitext.json  wikitext of the newly scanned questions
    replier_uids.json       signature name -> uid (null for IPs and names that cannot be resolved)

6. Possible errors and how they are handled
    - The page did not exist yet at the cutoff date: exists=False, skipped;
    - DiscussionTools returns an error (for example, the page is not a talk page): log issue `dt_error`; at build
      time the questions on that page go through fallback;
    - Network interruption: just rerun; finished requests are served from the cache.

7. How to run
    python -m pipeline.m2_conversations.collect_topup --online                 # about 1 to 2 hours
    python -m pipeline.m2_conversations.collect_topup --online --phase fallback # after build_questions
    python -m pipeline.m2_conversations.collect_topup --online --phase removals # after the second run of build_questions
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
from pathlib import Path

import pandas as pd

import config
from pipeline.common.api import ApiError, MediaWikiClient
from pipeline.common.io import iter_jsonl, write_json_atomic, write_jsonl_atomic
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_sources, check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import before_cutoff
from pipeline.common.users import norm_user

RAW = config.RAW / "m2"
NS_USER_TALK = 3
# The fallback phase mostly parses old revisions of very large talk pages, and discussiontoolspageinfo on the whole
# page always times out. So fallback parses only the one section that holds the question (fetch_section_html, still
# with DiscussionTools turn markers). Only when even the section cannot be fetched do we split the section wikitext
# by signatures (marked turn_source_regex).


def unscanned_mentors(log) -> pd.DataFrame:
    """Mentors on the M1 list that the existing tag scan did not cover (I-47).

    A mentor counts as "covered" if their uid appears in the raw eval2 questions, or one of their names is in the
    scan checkpoint.
    """
    ml = pd.read_parquet(config.INTERIM / "m1" / "mentor_list.parquet")
    names = pd.read_parquet(config.INTERIM / "m1" / "mentor_names.parquet")
    seen_uid = {int(r["mentor_uid"]) for r in iter_jsonl(config.EXTERNAL["eval2_tagged_raw"]) if r.get("mentor_uid")}
    ck = {norm_user(x) for x in config.EXTERNAL["tag_scan_checkpoint"].read_text().splitlines() if x.strip()}
    rows = []
    for r in ml.itertuples():
        nm = set(names[names.mentor_uid == r.mentor_uid].name)
        if r.mentor_uid in seen_uid or nm & ck:
            continue
        rows.append({"mentor_uid": int(r.mentor_uid), "username": r.username, "names": sorted(nm)})
    log.info("mentors never scanned for tags: %d", len(rows))
    return pd.DataFrame(rows)


def scan_tags(client: MediaWikiClient, title: str) -> list[dict]:
    """All revisions on one talk page before the cutoff date that carry a question tag (R4.1)."""
    out = []
    for tag in config.QUESTION_TAGS:
        params = {"action": "query", "prop": "revisions", "titles": title, "rvtag": tag, "rvlimit": 500,
                  "rvstart": config.DATA_CUTOFF_ISO, "rvdir": "older",
                  "rvprop": "ids|timestamp|user|userid|comment|tags|size"}
        for page in client.query_all(params, ("query", "pages")):
            for r in page.get("revisions", []):
                if before_cutoff(r["timestamp"]):
                    out.append({**r, "page": title, "tag": tag})
    return out


def fetch_move_logs(client: MediaWikiClient, names: list[str]) -> list[dict]:
    """The talk page move log for each name (I-45). A move is logged on the "source page", so we query by the old title."""
    out = []
    for n in names:
        title = f"User talk:{n}"
        for e in client.query_all({"action": "query", "list": "logevents", "letype": "move",
                                   "letitle": title, "lelimit": 500}, ("query", "logevents")):
            p = e.get("params", {})
            out.append({"logid": e.get("logid"), "timestamp": e.get("timestamp"), "from": title,
                        "to": p.get("target_title"), "user": e.get("user"), "comment": e.get("comment", "")})
    return out


def list_pages(client: MediaWikiClient, bases: list[str]) -> list[dict]:
    """The main talk page and all its archive subpages (redirects excluded)."""
    titles = []
    for b in bases:
        titles.append({"title": f"User talk:{b}", "base": b})
        for p in client.query_all({"action": "query", "list": "allpages", "apnamespace": NS_USER_TALK,
                                   "apprefix": f"{b}/", "apfilterredir": "nonredirects", "aplimit": "max"},
                                  ("query", "allpages")):
            titles.append({"title": p["title"], "base": b})
    return list({t["title"]: t for t in titles}.values())


def as_of_cutoff(client: MediaWikiClient, title: str) -> dict | None:
    """The last revision of the page before the cutoff date. Returns None if the page does not exist or had not been created yet at the cutoff."""
    d = client.get({"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
                    "rvstart": config.DATA_CUTOFF_ISO, "rvdir": "older", "rvprop": "ids|timestamp|size"})
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or not pages[0].get("revisions"):
        return None
    r = pages[0]["revisions"][0]
    return {"as_of_revid": r["revid"], "as_of_ts": r["timestamp"], "size": r.get("size")}


def subpages_with_questions(client: MediaWikiClient, bases: list[str]) -> set[str]:
    """Subpages of each mentor's talk page whose current text contains "Question from" (full-text search, CirrusSearch).

    Purpose: only a filter, to cut down on requests. Question headings created by the Growth feature are always
    written in wikitext as "== Question from [[User:X|X]] (time) ==". Archives are only appended to, never deleted
    from, so an archive that held a question at the cutoff date still contains these words today. If there is an
    exception (an archive later blanked), the question will not be found in the cutoff version; it then goes to the
    fallback phase and is recovered from the revision history, so it is not lost.
    """
    out = set()
    for b in bases:
        for r in client.query_all({"action": "query", "list": "search", "srnamespace": NS_USER_TALK,
                                   "srsearch": f'insource:"Question from" prefix:"User talk:{b}/"',
                                   "srlimit": "max", "srprop": ""}, ("query", "search")):
            out.add(r["title"])
    return out


def latest_revisions(client: MediaWikiClient, titles: list[str]) -> dict[str, dict | None]:
    """The current latest revision of each page {revid, timestamp}, 50 titles per request; None if the page does not exist.

    Purpose: filter out pages that cannot hold any question. A page whose latest revision is older than the launch
    of the mentorship program (2021-05-04) was never edited after that, so it cannot hold any question. For a page
    whose latest revision is older than the cutoff date, the cutoff version is the latest revision, so no extra
    query is needed. Only pages edited after the cutoff date need a page-by-page as_of_cutoff() query.
    """
    out = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        d = client.get({"action": "query", "prop": "revisions", "titles": "|".join(batch), "rvprop": "ids|timestamp"})
        q = d.get("query", {})
        back = {}
        for n in q.get("normalized", []):
            back[n["to"]] = n["from"]
        for pg in q.get("pages", []):
            t = back.get(pg["title"], pg["title"])
            r = (pg.get("revisions") or [None])[0]
            out[t] = None if pg.get("missing") or r is None else {"revid": r["revid"], "timestamp": r["timestamp"]}
    return out


def fetch_dt(client: MediaWikiClient, revid: int, issues: IssueLog, out_dir: Path, title: str | None = None,
             max_retries: int | None = None) -> bool:
    """Calls DiscussionTools on one revision and saves the result as out_dir/<revid>.json. Skips it if the file already exists.

    Very large talk pages (old versions of several MB) time out when the server parses them (HTTP 504). If the
    client still fails after 6 retries, we log issue `dt_error` (with the page title and revision ID) and return
    False, without stopping the whole collection. build_questions sends the questions on these pages to the
    fallback phase.
    """
    path = out_dir / f"{revid}.json"
    if path.exists():
        return True
    failed_path = out_dir / "_failed.json"
    failed = json.loads(failed_path.read_text()) if failed_path.exists() else {}
    if str(revid) in failed:
        return False                     # it already failed before (the timeout always happens), so do not retry
    try:
        d = client.get({"action": "discussiontoolspageinfo", "oldid": revid, "prop": "threaditemshtml"},
                       max_retries=max_retries)
    except (ApiError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
        issues.add("dt_error", rule="R5.1", action="flagged", ids={"revid": revid, "page": title},
                   detail={"error": f"{type(e).__name__}: {e}"},
                   url=f"https://en.wikipedia.org/w/index.php?oldid={revid}")
        failed[str(revid)] = {"page": title, "error": f"{type(e).__name__}: {e}"}
        write_json_atomic(failed_path, failed)
        return False
    write_json_atomic(path, d.get("discussiontoolspageinfo", {}).get("threaditemshtml", []), indent=0)
    return True


def phase_pages(client, log, issues) -> dict:
    """Phase pages: tag scan for missing mentors, move logs, the page list, and DiscussionTools parsing of the cutoff versions."""
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "dt").mkdir(exist_ok=True)
    names = pd.read_parquet(config.INTERIM / "m1" / "mentor_names.parquet")

    # 1. Tag scan for mentors not yet scanned
    un = unscanned_mentors(log)
    top = []
    for r in un.itertuples():
        for n in r.names:
            got = scan_tags(client, f"User talk:{n}")
            for g in got:
                g["mentor_uid"] = r.mentor_uid
            top += got
            log.info("  tag scan %-28s %d questions", n, len(got))
    write_jsonl_atomic(RAW / "tag_topup.jsonl", top)

    # 2. Move logs
    all_names = sorted(set(names.name))
    moves = fetch_move_logs(client, all_names)
    write_jsonl_atomic(RAW / "move_log.jsonl", moves)
    log.info("move log: %d moves for %d names", len(moves), len(all_names))

    # 3. Page list: pages recorded in eval2 + the talk page of each name + archive subpages
    bases = {p.split(":", 1)[1] for p in {r["page"] for r in iter_jsonl(config.EXTERNAL["eval2_conversations"])}}
    bases |= set(all_names)
    pages = list_pages(client, sorted(bases))
    log.info("pages to check: %d (bases %d)", len(pages), len(bases))

    # 4. Cutoff versions + DiscussionTools
    latest = latest_revisions(client, [p["title"] for p in pages])
    with_q = subpages_with_questions(client, sorted({p["base"] for p in pages if "/" in p["title"]}))
    log.info("subpages containing 'Question from' now: %d of %d", len(with_q), sum("/" in p["title"] for p in pages))
    start_iso = config.MENTORSHIP_START.isoformat().replace("+00:00", "Z")
    t0 = time.time()
    n_skip = 0
    for i, p in enumerate(pages, 1):
        lt = latest.get(p["title"])
        if lt is not None and lt["timestamp"] < start_iso:
            # Never edited since the mentorship program launched: it cannot hold a question, so do not parse it
            p.update({"as_of_revid": lt["revid"], "as_of_ts": lt["timestamp"], "size": None,
                      "exists": True, "skipped": "no_edit_since_mentorship_start"})
            n_skip += 1
            continue
        if "/" in p["title"] and p["title"] not in with_q:
            p.update({"as_of_revid": None, "as_of_ts": None, "size": None, "exists": True,
                      "skipped": "no_question_heading"})
            n_skip += 1
            continue
        if lt is not None and before_cutoff(lt["timestamp"]):
            a = {"as_of_revid": lt["revid"], "as_of_ts": lt["timestamp"], "size": None}
        else:
            a = as_of_cutoff(client, p["title"])
        p.update(a or {"as_of_revid": None, "as_of_ts": None, "size": None})
        p["exists"] = a is not None
        if a:
            p["dt_ok"] = fetch_dt(client, a["as_of_revid"], issues, RAW / "dt", p["title"])
            if not p["dt_ok"]:
                log.warning("  DiscussionTools failed for %s (revid %s)", p["title"], a["as_of_revid"])
        if i % 100 == 0:
            log.info("  %d/%d pages (%.0f s, api %s)", i, len(pages), time.time() - t0, client.stats)
    write_jsonl_atomic(RAW / "pages.jsonl", pages)
    log.info("pages skipped (no edit since %s, or a subpage without any question heading): %d", start_iso, n_skip)
    return {"pages_skipped_pre_mentorship": n_skip, "unscanned_mentors": len(un), "topup_questions": len(top), "moves": len(moves),
            "pages": len(pages), "pages_existing": sum(p["exists"] for p in pages)}


def revision_content(client: MediaWikiClient, revid: int) -> str | None:
    """The wikitext of one revision; returns None if the revision is hidden."""
    d = client.get({"action": "query", "prop": "revisions", "revids": revid, "rvslots": "main",
                    "rvprop": "ids|timestamp|content"})
    pages = d.get("query", {}).get("pages", [])
    if not pages or not pages[0].get("revisions"):
        return None
    slot = pages[0]["revisions"][0].get("slots", {}).get("main", {})
    return None if slot.get("texthidden") else slot.get("content")


def heading_marker(heading: str | None, q_ts: str | None = None) -> re.Pattern | None:
    """A pattern that recognizes the question heading: a line that starts with ==, contains "question from", and ends with the time "(hh:mm, d month yyyy)" (R5.5).

    When the heading is missing (a few eval2 records have none), we build the same time string from the question
    time: headings created by the Growth feature end with the minute when the question was asked.
    """
    m = re.search(r"\((\d{1,2}:\d{2}, \d{1,2} [a-z]+ \d{4})\)\s*$", heading.lower()) if isinstance(heading, str) else None
    if m:
        when = m.group(1)
    elif isinstance(q_ts, str) and q_ts:
        from pipeline.common.timeutil import parse_iso
        t = parse_iso(q_ts)
        when = f"{t:%H:%M}, {t.day} {t:%B %Y}".lower()
    else:
        return None
    return re.compile(r"^==.*question from.*\(" + re.escape(when) + r"\).*==\s*$", re.M)


def section_present(client, revid: int, marker: re.Pattern) -> bool:
    """Whether the wikitext of revision revid still contains the heading of this question."""
    text = revision_content(client, revid)
    return bool(text) and bool(marker.search(text.lower()))


_HISTORY: dict[str, list[dict]] = {}


def page_history(client, page: str) -> list[dict]:
    """All revisions {revid, timestamp} of the page from the mentorship launch (2021-05-04) to the cutoff date, fetched only once per page.

    One talk page often has hundreds of questions to recover (for example 600 for Oshwah). If each question paged
    through the history on its own, the same requests would be repeated dozens of times.
    """
    if page not in _HISTORY:
        revs = []
        for pg in client.query_all({"action": "query", "prop": "revisions", "titles": page, "rvlimit": "max",
                                    "rvstart": config.MENTORSHIP_START.isoformat().replace("+00:00", "Z"),
                                    "rvend": config.DATA_CUTOFF_ISO, "rvdir": "newer", "rvprop": "ids|timestamp"},
                                   ("query", "pages")):
            revs += [r for r in pg.get("revisions", []) if before_cutoff(r["timestamp"])]
        _HISTORY[page] = revs
    return _HISTORY[page]


def last_revision_with_section(client, page: str, q_revid: int, q_ts: str, marker: re.Pattern) -> tuple[dict | None, int]:
    """Among revisions after the question and before the cutoff date, binary-searches for the last one that still contains this section (R5.6).

    Assumption: the section exists from the moment it is posted until some edit deletes or archives it (once gone,
    it stays gone). After the binary search, we check that the next revision after the result really lacks the
    section, as a verification.

    Returns:
        ({"revid", "timestamp"} or None, number of times revision content was read)
    """
    revs = [r for r in page_history(client, page) if r["timestamp"] >= q_ts]
    revs = sorted({r["revid"]: r for r in revs}.values(), key=lambda r: (r["timestamp"], r["revid"]))
    if not revs or revs[0]["revid"] != q_revid:
        revs = [r for r in revs if r["revid"] >= q_revid]
        revs.insert(0, {"revid": q_revid, "timestamp": q_ts})
    probes = 1
    if not section_present(client, revs[0]["revid"], marker):
        return None, probes
    # Common case: the question is reverted or undone in the very next edit (for example, treated as spam). Check the next revision first; if it matches, only 2 reads are needed
    if len(revs) > 1:
        probes += 1
        if not section_present(client, revs[1]["revid"], marker):
            return revs[0], probes
    lo, hi = (1 if len(revs) > 1 else 0), len(revs) - 1          # revs[lo] contains the section; look in (lo, hi] for the last one that contains it
    while lo < hi:
        mid = (lo + hi + 1) // 2
        probes += 1
        if section_present(client, revs[mid]["revid"], marker):
            lo = mid
        else:
            hi = mid - 1
    return revs[lo], probes


def extract_section(text: str, section_name: str) -> str | None:
    """Takes the body of the level-2 section titled section_name out of the page wikitext (without the heading line)."""
    import mwparserfromhell as mwp
    target = re.sub(r"\s+", " ", section_name).strip().lower()
    for sec in mwp.parse(text).get_sections(levels=[2]):
        h = sec.filter_headings()[0]
        if re.sub(r"\s+", " ", h.title.strip_code()).strip().lower() == target:
            return str(sec)[len(str(h)):].strip("\n")
    return None


def section_index_from_wikitext(text: str, marker: re.Pattern) -> int | None:
    """Works out the number N of the question section from the wikitext, using MediaWiki's section numbering rules (for action=parse&section=N).

    Rule: headings written directly on the page (== ... ==, any level) are numbered from 1 in the order they
    appear. Headings that come from expanded templates do not take a number (their number is "T-n").
    mwparserfromhell does not expand templates, so the order of headings it finds is the numbering order.
    """
    import mwparserfromhell as mwp
    for i, h in enumerate(mwp.parse(text).filter_headings(recursive=False), 1):
        if h.level == 2 and marker.search(str(h).lower()):
            return i
    return None


def fetch_section_html(client, revid: int, marker: re.Pattern, out_dir: Path, qrev: int) -> bool:
    """Parses only the one section that holds the question (action=parse&oldid=...&section=N) and saves the HTML (with DiscussionTools turn markers).

    Steps:
        1. Get the wikitext of the revision (no server-side parsing needed, so it is fast) and work out the number
           N with section_index_from_wikitext;
        2. Get the HTML for section=N and check that its heading contains the question time;
        3. If the check fails (for example, a template creates a heading at the top of the page), ask the server
           to list the numbers with prop=sections instead (slow, but accurate).
    The result is saved as out_dir/<question revid>.json: {revid, index, line, html, index_source}.
    """
    path = out_dir / f"{qrev}.json"
    if path.exists():
        return True

    def section_html(index):
        return client.get({"action": "parse", "oldid": revid, "section": index, "prop": "text",
                           "disableeditsection": 1}, max_retries=2)["parse"]["text"]

    def heading_ok(html):
        m = re.search(r"<h2[^>]*>(.*?)</h2>", html or "", re.S)
        return bool(m) and bool(marker.search("== " + re.sub(r"<[^>]+>", "", m.group(1)).lower() + " =="))

    try:
        text = revision_content(client, revid) or ""
        idx, source = section_index_from_wikitext(text, marker), "wikitext"
        html = section_html(idx) if idx else None
        if not heading_ok(html):
            secs = client.get({"action": "parse", "oldid": revid, "prop": "sections"}, max_retries=2)["parse"]["sections"]
            hit = next((x for x in secs if str(x.get("level")) == "2"
                        and marker.search("== " + re.sub(r"<[^>]+>", "", x.get("line", "")).lower() + " ==")), None)
            if hit is None:
                return False
            idx, source = hit["index"], "prop_sections"
            html = section_html(idx)
    except (ApiError, urllib.error.URLError, TimeoutError, ConnectionError, KeyError):
        return False
    m = re.search(r"<h2[^>]*>(.*?)</h2>", html, re.S)
    line = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip() if m else ""
    write_json_atomic(path, {"revid": revid, "index": idx, "line": line, "html": html, "index_source": source})
    return True


def find_thread_last_revision(client, name: str) -> list[dict]:
    """DiscussionTools comment tracking (action=discussiontoolsfindcomment, the API behind comment permalinks).

    The idorname argument can be a thread key "h-<author>-YYYYMMDDHHMM00". It returns every page that contains this
    thread, and the oldid of "the last revision that contains it". After the thread is deleted, oldid is the last
    revision before the deletion, which is exactly the version R5.6 needs.
    """
    d = client.get({"action": "discussiontoolsfindcomment", "idorname": name})
    return d.get("discussiontoolsfindcomment", []) or []


def revision_timestamps(client, revids: list[int]) -> dict[int, str]:
    """Revision ID -> timestamp, 50 per request."""
    out = {}
    for i in range(0, len(revids), 50):
        d = client.get({"action": "query", "prop": "revisions", "revids": "|".join(map(str, revids[i:i + 50])),
                        "rvprop": "ids|timestamp"})
        for pg in d.get("query", {}).get("pages", []):
            for r in pg.get("revisions", []):
                out[int(r["revid"])] = r["timestamp"]
    return out


def thread_key_name(user: str, q_ts: str) -> str:
    """The thread key of a question: h-<asker, with underscores instead of spaces>-<minute of the question>00."""
    t = q_ts.replace("-", "").replace(":", "").replace("T", "")[:12]
    return f"h-{user.replace(' ', '_')}-{t}00"


def extract_section_by_marker(text: str, marker: re.Pattern) -> tuple[str, str] | None:
    """Takes out the level-2 section that matches the heading pattern: returns (heading plain text, section body); None if not found."""
    import mwparserfromhell as mwp
    for sec in mwp.parse(text).get_sections(levels=[2]):
        h = sec.filter_headings()[0]
        if marker.search(str(h).lower()):
            return re.sub(r"\s+", " ", h.title.strip_code()).strip(), str(sec)[len(str(h)):].strip("\n")
    return None


def resolve_names(client, names: list[str], issues: IssueLog) -> dict:
    """Signature name -> uid (R6.3): first try list=users; if not found (an old name from before a rename), follow the rename log to the current name."""
    from pipeline.m1_roster.collect_roster import resolve_name
    out = {}
    for i in range(0, len(names), 50):
        batch = names[i:i + 50]
        for u in client.get({"action": "query", "list": "users", "ususers": "|".join(batch)})["query"]["users"]:
            out[norm_user(u.get("name"))] = int(u["userid"]) if u.get("userid") else None
    for n in [n for n in names if out.get(norm_user(n)) is None]:
        if re.match(r"^[\d.:a-fA-F]+$", n) and (n.count(".") == 3 or ":" in n):
            continue                     # IP address: a logged-out edit, no uid
        uid, path = resolve_name(client, n)
        out[norm_user(n)] = uid
        if uid is not None and len(path) > 1:
            issues.add("replier_renamed", rule="R6.3", action="fixed", ids={"name": n}, detail={"path": path, "uid": uid})
    return out


def _revisions(client, params: dict) -> list[dict]:
    """One page of prop=revisions results (ids, timestamps)."""
    d = client.get({"action": "query", "prop": "revisions", "rvprop": "ids|timestamp", **params})
    pages = d.get("query", {}).get("pages", [])
    return pages[0].get("revisions", []) if pages and not pages[0].get("missing") else []


def page_history_by_id(client, pageid: int) -> list[dict]:
    """All revisions of a page from the mentorship launch to the cutoff time, by pageid (when a page is moved, its history moves with it)."""
    key = f"id:{pageid}"
    if key not in _HISTORY:
        revs = []
        for pg in client.query_all({"action": "query", "prop": "revisions", "pageids": pageid, "rvlimit": "max",
                                    "rvstart": config.MENTORSHIP_START.isoformat().replace("+00:00", "Z"),
                                    "rvend": config.DATA_CUTOFF_ISO, "rvdir": "newer", "rvprop": "ids|timestamp"},
                                   ("query", "pages")):
            revs += [r for r in pg.get("revisions", []) if before_cutoff(r["timestamp"])]
        _HISTORY[key] = revs
    return _HISTORY[key]


def phase_recheck(client, log, issues) -> dict:
    """Phase recheck: re-checks questions flagged as "deleted in the very next edit" (section_removed_immediately).

    Problem found: some questions in this group were not actually deleted:
        (1) The talk page was moved (the mentor was renamed, and the next edit is "X moved page ..."). The page
            history moves with the page, so querying the history by the old title leaves only a redirect, and the
            binary search wrongly concludes the section disappeared right after the question;
        (2) The same newcomer submitted twice within a few minutes, and the mentor deleted the second heading or
            edited inside the section, so the revision given by comment tracking is not the last one.
    Method: get the content of the next edit by the pageid of the question revision. If the heading is still there,
    binary-search the history of this pageid again for the last revision that still contains the section, then
    parse only that section (fetch_section_html). Results are written to fallback_map.json (method="recheck").
    """
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet")
    g = q[q.parse_flags.map(lambda f: "section_removed_immediately" in list(f))]
    fmap = json.loads((RAW / "fallback_map.json").read_text())
    (RAW / "fallback_sections").mkdir(parents=True, exist_ok=True)
    n_still, n_fixed, t0 = 0, 0, time.time()
    for i, r in enumerate(g.itertuples(), 1):
        marker = heading_marker(r.heading, r.q_ts)
        if marker is None:
            continue
        d = client.get({"action": "query", "prop": "revisions", "revids": int(r.question_revid), "rvprop": "ids"})
        pages = d.get("query", {}).get("pages", [])
        if not pages or not pages[0].get("pageid"):
            continue
        pageid = pages[0]["pageid"]
        nxt = _revisions(client, {"pageids": pageid, "rvstartid": int(r.question_revid), "rvdir": "newer", "rvlimit": 2})
        nxt = [x for x in nxt if x["revid"] != int(r.question_revid) and before_cutoff(x["timestamp"])]
        if not nxt or not section_present(client, nxt[0]["revid"], marker):
            continue                                   # the next edit really did delete the section (needs only 1 to 2 requests)
        revs = [x for x in page_history_by_id(client, pageid) if x["revid"] >= int(r.question_revid)]
        n_still += 1
        lo, hi = 1, len(revs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if section_present(client, revs[mid]["revid"], marker):
                lo = mid
            else:
                hi = mid - 1
        last = revs[lo]
        path = RAW / "fallback_sections" / f"{int(r.question_revid)}.json"
        if path.exists():
            path.unlink()
        ok = fetch_section_html(client, last["revid"], marker, RAW / "fallback_sections", int(r.question_revid))
        title = (client.get({"action": "query", "prop": "info", "pageids": pageid})["query"]["pages"][0]).get("title")
        fmap[str(int(r.question_revid))] = {"revid": last["revid"], "timestamp": last["timestamp"], "page": title,
                                            "method": "recheck", "dt_ok": ok}
        n_fixed += ok
        issues.add("removal_false_positive", rule="R5.6", action="fixed", ids={"revid": int(r.question_revid)},
                   detail={"pageid": pageid, "last_revid": last["revid"], "page": title})
        if i % 100 == 0:
            log.info("  recheck %d/%d (still present %d, %.0f s)", i, len(g), n_still, time.time() - t0)
    write_json_atomic(RAW / "fallback_map.json", fmap)
    log.info("recheck: %d of %d flagged questions still had their section after the next edit; %d re-parsed",
             n_still, len(g), n_fixed)
    return {"rechecked": len(g), "still_present": n_still, "reparsed": n_fixed}


def phase_removals(client, log, issues) -> dict:
    """Phase removals (after the second run of build_questions): for questions deleted in the very next edit, fetches the metadata of that edit.

    Why this is needed: these questions (about 1,100) almost never have replies, but they were deleted for different
    reasons, and M5 must handle them separately: reverting spam or vandalism (mw-rollback, mw-undo,
    mw-manual-revert tags), the newcomer deleting it themselves, a repeat submission by the same newcomer merged into
    another section (the replies are in the other section), or other reasons.
    Output data/raw/m2/removal_edits.json: question revid -> the next edit {revid, timestamp, user, userid, comment, tags}.
    """
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet")
    g = q[q.parse_flags.map(lambda f: "section_removed_immediately" in list(f))]
    out = {}
    for i, r in enumerate(g.itertuples(), 1):
        d = client.get({"action": "query", "prop": "revisions", "titles": r.page_eval, "rvstartid": int(r.question_revid),
                        "rvdir": "newer", "rvlimit": 2, "rvprop": "ids|timestamp|user|userid|comment|tags"})
        revs = (d.get("query", {}).get("pages") or [{}])[0].get("revisions", [])
        nx = next((x for x in revs if x["revid"] != int(r.question_revid)), None)
        out[str(r.question_revid)] = nx
        if i % 200 == 0:
            log.info("  removals %d/%d", i, len(g))
    write_json_atomic(RAW / "removal_edits.json", out)
    log.info("removal edits fetched: %d", len(out))
    return {"removals": len(out)}


def phase_fallback(client, log, issues) -> dict:
    """Phase fallback (after the first run of build_questions):

    1. Get the wikitext of the newly scanned questions (eval2 questions already have their text; the newly scanned
       ones have only revision metadata);
    2. For questions whose thread is not in the cutoff version: find in the revision history the last revision
       before the section was deleted or archived, and parse it with DiscussionTools (R5.6);
    3. For signature names in the turns whose identity is unknown, find the uid (this handles renamed newcomers,
       for example AlphaBetaGamma).
    """
    out = config.INTERIM / "m2"
    need = out / "fallback_needed.csv"
    if not need.exists():
        raise FileNotFoundError(f"{need} missing: run build_questions.py first")
    (RAW / "fallback_dt").mkdir(parents=True, exist_ok=True)

    # 1. Wikitext of the newly scanned questions
    wt = {}
    for r in iter_jsonl(RAW / "tag_topup.jsonl"):
        m = re.match(r"^/\*\s*(.+?)\s*\*/", r.get("comment") or "")
        text = revision_content(client, r["revid"])
        sec = extract_section(text, m.group(1)) if (text and m) else None
        wt[str(r["revid"])] = {"heading": m.group(1) if m else None, "text": sec}
        if sec is None:
            issues.add("topup_text_not_found", rule="R4.1", action="flagged", ids={"revid": r["revid"]},
                       detail={"comment": r.get("comment"), "hidden": text is None})
    write_json_atomic(RAW / "question_wikitext.json", wt)
    log.info("top-up question wikitext: %d/%d", sum(v["text"] is not None for v in wt.values()), len(wt))

    # 2. The last revision before deletion or archiving
    #    2a. First use DiscussionTools comment tracking to find "the last revision that contains this thread" (one request).
    #        If that revision is after the cutoff date (the thread was deleted later, or it is on a page not in our
    #        list), use that page's cutoff version instead.
    #    2b. If comment tracking has no record, binary-search the page revision history (R5.6).
    fb = pd.read_csv(need)
    fb_head = dict(zip(fb.question_revid.astype(int), fb.heading))
    fb_ts = dict(zip(fb.question_revid.astype(int), fb.q_ts))
    (RAW / "fallback_sections").mkdir(parents=True, exist_ok=True)
    q = pd.read_parquet(out / "questions.parquet").set_index("question_revid")
    fmap, t0 = {}, time.time()
    found = {}
    for i, r in enumerate(fb.itertuples(), 1):
        user = q.loc[r.question_revid, "mentee_name"]
        if not isinstance(user, str) or not user:
            continue                    # user name is hidden: there is no thread key, so go straight to the binary search
        name = thread_key_name(user, r.q_ts)
        hits = find_thread_last_revision(client, name)
        base = r.page.split(":", 1)[1].split("/", 1)[0]
        # Prefer the copy on the same mentor's talk page (including archives)
        hits.sort(key=lambda h: (h["title"].split(":", 1)[-1].split("/", 1)[0] != base, h["title"]))
        if hits:
            # oldid is empty: the thread is still on this page now (not an old revision); below we use this page's cutoff version instead
            oid = hits[0].get("oldid")
            found[int(r.question_revid)] = {"name": name, "title": hits[0]["title"], "oldid": int(oid) if oid else None,
                                            "n_hits": len(hits)}
        if i % 200 == 0:
            log.info("  findcomment %d/%d (found %d, %.0f s)", i, len(fb), len(found), time.time() - t0)
    ts = revision_timestamps(client, sorted({v["oldid"] for v in found.values() if v["oldid"]}))
    for j, (qrev, v) in enumerate(found.items(), 1):
        if j % 200 == 0:
            log.info("  section parse %d/%d (%.0f s, api %s)", j, len(found), time.time() - t0, client.stats)
        rev, rts = v["oldid"], ts.get(v["oldid"])
        if rts is None or not before_cutoff(rts):
            a = as_of_cutoff(client, v["title"])
            if a is None:
                continue
            rev, rts = a["as_of_revid"], a["as_of_ts"]
        marker = heading_marker(fb_head.get(qrev), fb_ts.get(qrev))
        ok = (RAW / "fallback_dt" / f"{rev}.json").exists() or (
            marker is not None and fetch_section_html(client, rev, marker, RAW / "fallback_sections", qrev))
        fmap[str(qrev)] = {"revid": rev, "timestamp": rts, "page": v["title"], "method": "findcomment",
                           "key": v["name"], "dt_ok": ok}
    log.info("findcomment: %d/%d questions located", len(fmap), len(fb))

    rest = fb[~fb.question_revid.astype(str).isin(fmap)]
    for i, r in enumerate(rest.itertuples(), 1):
        marker = heading_marker(r.heading, r.q_ts)
        if marker is None:
            issues.add("fallback_no_marker", rule="R5.5", action="flagged", ids={"revid": r.question_revid},
                       detail={"heading": r.heading})
            continue
        last, probes = last_revision_with_section(client, r.page, int(r.question_revid), r.q_ts, marker)
        if last is None:
            issues.add("fallback_section_absent", rule="R5.6", action="flagged", ids={"revid": r.question_revid},
                       detail={"page": r.page, "heading": r.heading})
            continue
        ok = (RAW / "fallback_dt" / f"{last['revid']}.json").exists() or fetch_section_html(
            client, last["revid"], marker, RAW / "fallback_sections", int(r.question_revid))
        fmap[str(r.question_revid)] = {"revid": last["revid"], "timestamp": last["timestamp"], "page": r.page,
                                       "method": "bisect", "probes": probes, "dt_ok": ok}
        if i % 50 == 0:
            log.info("  bisect %d/%d (%.0f s, api %s)", i, len(rest), time.time() - t0, client.stats)
    for qrev, v in fmap.items():
        if not v["dt_ok"]:
            marker = heading_marker(fb_head.get(int(qrev)), fb_ts.get(int(qrev)))
            got = extract_section_by_marker(revision_content(client, v["revid"]) or "", marker) if marker else None
            if got:
                (RAW / "fallback_wikitext").mkdir(exist_ok=True)
                write_json_atomic(RAW / "fallback_wikitext" / f"{qrev}.json",
                                  {"revid": v["revid"], "page": v["page"], "heading": got[0], "text": got[1]})
    write_json_atomic(RAW / "fallback_map.json", fmap)
    log.info("fallback: %d/%d questions recovered from history", len(fmap), len(fb))

    # 3. Signature name -> uid
    names_path = out / "names_to_resolve.csv"
    names = pd.read_csv(names_path)["name"].dropna().astype(str).tolist() if names_path.exists() else []
    uids = resolve_names(client, names, issues)
    write_json_atomic(RAW / "replier_uids.json", uids)
    log.info("replier names resolved: %d/%d", sum(v is not None for v in uids.values()), len(names))
    return {"topup_text": len(wt), "fallback_requested": len(fb), "fallback_recovered": len(fmap),
            "names_requested": len(names), "names_resolved": sum(v is not None for v in uids.values())}


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("M2 collection: tag top-up, page moves, DiscussionTools as of the cutoff.")
    ap.add_argument("--phase", choices=["pages", "fallback", "removals", "recheck"], default="pages")
    args = ap.parse_args(argv)
    log = setup_logging(f"m2_collect_{args.phase}")
    t0 = time.time()
    check_upstream("m2_collect", ["m1"])
    check_sources(["eval2_tagged_raw", "eval2_conversations", "tag_scan_checkpoint"])
    issues = IssueLog(f"m2_collect_{args.phase}")
    client = MediaWikiClient(online=args.online)
    counts = {"pages": phase_pages, "fallback": phase_fallback, "removals": phase_removals,
              "recheck": phase_recheck}[args.phase](client, log, issues)
    issues.write()
    log.info("done: %s | api %s", counts, client.stats)
    write_manifest(f"m2_collect_{args.phase}", inputs=[config.EXTERNAL["eval2_conversations"]],
                   outputs=[p for p in RAW.glob("*.jsonl")], params={"online": args.online, "phase": args.phase},
                   counts={**counts, "api": client.stats}, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
