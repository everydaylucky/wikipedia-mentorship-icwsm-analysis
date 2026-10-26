"""
turns.py: turn DiscussionTools parse results into a list of "turns" (single messages), and pull text and links out of each turn's HTML

Module: M2 (PLAN.md M2; CODE_PLAN.md section 5, M2)
Rules: DATA_PROCESSING R4.4 (away notices), R4.5 (links grouped by namespace), R5.1 (splitting into turns), R5.2 (speaker roles)

1. What it does
    1. heading_key(): reads the key DiscussionTools gives each thread, "h-<asker>-<YYYYMMDDHHMMSS>",
       and gets the asker (the signature name at that time) and the question time (to the minute).
    2. flatten_thread(): unfolds a thread's reply tree into a list of turns, in the order they appear.
    3. html_to_text(): turn HTML → plain text. It removes signatures (user links and signature
       timestamps) and removes the system away notice at the start (reading the primary mentor from it).
    4. html_links(): pulls from turn HTML the internal links (target page titles), the number of
       external links, and the number of list items.

2. Why it is designed this way
    DiscussionTools is the official parser behind Wikipedia's "Reply" feature (Extension:DiscussionTools).
    It splits turns by signature and identifies the author and time. This is more reliable than our
    own signature regex (R5.1).
    It returns rendered HTML, so:
        - Links are already normalized page titles (the title attribute of a[rel="mw:WikiLink"]), and the
          namespace can be read from the official table (R4.5). Redirects from pseudo-namespaces (for
          example "H:") are resolved together in M6;
        - Templates are already expanded, so the templates themselves are not visible in the HTML. The
          template count feature is therefore computed from the question's wikitext instead (M6).

3. Inputs / 4. Outputs
    DiscussionTools threaditemshtml (a list); the output is Python objects.

6. Possible errors and how they are handled
    - A turn has no author (DiscussionTools could not recognize the signature): author is None, and the caller marks it role="unknown";
    - A thread key cannot be parsed: heading_key returns None, and the caller logs an issue.

8. How to check
    tests/test_m2.py tests these functions on real pieces of DiscussionTools output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from pipeline.common.timeutil import SIG_TS_RE, parse_iso
from pipeline.common.users import norm_user
from pipeline.common.wikimeta import namespace_table

# Thread keys come in two styles: the new style "h-<author>-20251106224000"; old threads keep the old style
# "h-<author>-2015-10-04T14:01:00.0" or "h-<author>-2015-07-09T19:33:00.000Z". Both are used only to the minute.
_KEY_RE = re.compile(r"^h-(?P<author>.+)-(?P<ts>\d{14}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)$")
# Away notice (R4.4): after HTML is turned into text it looks like "Note: <newcomer>'s mentor <primary mentor> is away."
_AWAY_TEXT_RE = re.compile(r"^\s*Note:\s*(?P<mentee>.+?)'s\s+mentor\s+(?P<primary>.+?)\s+is\s+away(?:\s+until\s+[^.\n]*)?\.?\s*", re.I | re.S)
_SIG_REMNANT_RE = re.compile(r"(?:[-–—~]{1,3}\s*)?\(\s*(?:talk|contribs?|t|c|·|\s|/|\|)*\)\s*")
_WS_RE = re.compile(r"[ \t ]+")
_USER_HREF_RE = re.compile(r"^\./(?:User|User_talk|Special:Contrib(?:ution)?s)[:/]", re.I)


@dataclass
class Turn:
    """One turn (a single message).

    Attributes:
        author: the signature name found by DiscussionTools (the user name at the time of the turn; may be None).
        ts: time of the turn (UTC, to the minute).
        level: reply depth (1 is the top level of the thread).
        comment_id: the DiscussionTools id of the turn.
        html: the turn's HTML.
    """
    author: str | None
    ts: datetime | None
    level: int
    comment_id: str
    html: str
    text: str | None = None      # when the fallback splitter is used, plain text is given directly (no HTML)


@dataclass
class Thread:
    """One thread.

    Attributes:
        key: the DiscussionTools thread key "h-<asker>-<time>".
        key_author: the asker in the key (the signature name at that time, normalized).
        key_ts: the question time in the key (to the minute).
        heading_html: heading HTML (the entry-point check uses the links inside it).
        heading_text: plain text of the heading, for example "Question from X on Article (22:40, 6 November 2025)".
        page: title of the page the thread is on; as_of_revid: the revision id used for parsing.
        turns: list of turns (in the order they appear).
    """
    key: str
    key_author: str | None
    key_ts: datetime | None
    heading_text: str
    page: str
    as_of_revid: int
    heading_html: str = ""
    turns: list[Turn] = field(default_factory=list)
    owner_uid: int | None = None          # mentor who owns the page (filled in by build_questions)
    origin: str = "cutoff"                # "cutoff": the version at the cutoff; "fallback": recovered from revision history
    for_question: int | None = None       # which question a fallback thread was recovered for


def heading_key(name: str | None) -> tuple[str | None, datetime | None]:
    """Parse a thread key: "h-Brayden.frankcom-20251106224000" → ("Brayden.frankcom", 2025-11-06 22:40 UTC).

    Old-style key: "h-Rubbish_computer-2015-10-04T14:01:00.0" → ("Rubbish computer", 2015-10-04 14:01 UTC).
    Notes: user names in keys use underscores instead of spaces; here they are normalized back to spaces.
    """
    m = _KEY_RE.match(name or "")
    if not m:
        return None, None
    raw = m.group("ts")
    if raw.isdigit():
        ts = datetime.strptime(raw, "%Y%m%d%H%M%S")
    else:
        ts = datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S")
    ts = ts.replace(tzinfo=timezone.utc)
    return norm_user(m.group("author")), ts


_BLOCKS = ("p", "li", "dd", "dt", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote")


def _block_text(soup: BeautifulSoup) -> str:
    """Visible text: a line break after block elements and at <br>; inline elements (links, bold) are joined directly."""
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for el in soup.find_all(_BLOCKS):
        el.append("\n")
    return soup.get_text("")


def _text(html: str) -> str:
    """Visible text of the HTML (line breaks kept)."""
    return _block_text(BeautifulSoup(html or "", "lxml"))


def flatten_thread(item: dict, page: str, as_of_revid: int) -> Thread:
    """Unfold one DiscussionTools thread (type="heading") into a Thread.

    The reply tree is unfolded depth first, in the order items appear. This matches the top-to-bottom
    reading order on the talk page.
    """
    author, ts = heading_key(item.get("name"))
    th = Thread(key=item.get("name", ""), key_author=author, key_ts=ts,
                heading_text=re.sub(r"\s+", " ", _text(item.get("html", ""))).strip(),
                page=page, as_of_revid=as_of_revid, heading_html=item.get("html", ""))

    def walk(node: dict) -> None:
        for r in node.get("replies", []):
            if r.get("type") == "heading":
                # A sub-heading is its own thread (threads_from_dt collects it). Do not pull its comments into
                # this one: archive bots often group a month of questions as level-3 sections under one
                # level-2 heading, and that heading can even share the key of its first sub-section (I-56).
                continue
            if r.get("type") == "comment":
                th.turns.append(Turn(author=norm_user(r.get("author")), ts=parse_iso(r.get("timestamp")),
                                     level=int(r.get("level", 0) or 0), comment_id=r.get("id", ""),
                                     html=r.get("html", "")))
            walk(r)
    walk(item)
    return th


def threads_from_dt(items: list[dict], page: str, as_of_revid: int) -> list[Thread]:
    """All threads on one page: every heading, including headings nested under another heading (I-56).

    Each heading becomes one thread holding only the comments directly under it (not those under its
    sub-headings). A heading with no comments of its own but with sub-headings is a pure container and is
    dropped (an empty heading without sub-headings is kept, as before: it can be an unsigned question).
    A heading that DiscussionTools lists twice (once nested, once at the top level) is kept once.
    """
    out, seen = [], set()

    def visit(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "heading":
            th = flatten_thread(node, page, as_of_revid)
            sig = (th.key, tuple(t.comment_id for t in th.turns))
            container = not th.turns and any(isinstance(r, dict) and r.get("type") == "heading"
                                             for r in node.get("replies", []))
            if not container and sig not in seen:
                seen.add(sig)
                out.append(th)
        for r in node.get("replies", []):
            visit(r)

    for it in items or []:
        visit(it)
    return out


def html_to_text(html: str) -> tuple[str, dict | None]:
    """Turn HTML → plain text (R4.4, R4.6).

    Steps:
        1. Take the full text first, detect an away notice at the start, read the newcomer name and the
           primary mentor name from it, then remove the notice;
        2. Remove links that point to users (signatures, pings): User:, User talk:, Special:Contributions/;
        3. Remove signature timestamps and leftover signature bits ("(talk)", "--", and so on), and collapse whitespace.

    Returns:
        (plain text, away notice info {"mentee", "primary_mentor"} or None)
    """
    soup = BeautifulSoup(html or "", "lxml")
    away = None
    full = soup.get_text(" ")
    m = _AWAY_TEXT_RE.match(full)
    if m:
        away = {"mentee": norm_user(m.group("mentee")), "primary_mentor": norm_user(m.group("primary"))}
        # The away notice is italic text in the first paragraph (DiscussionTools renders it as <i> or <p><i>); remove the first block that contains "is away"
        for el in soup.find_all(["i", "p", "dl", "dd"]):
            if "is away" in el.get_text(" ") and "mentor" in el.get_text(" "):
                el.decompose()
                break
    for a in soup.find_all("a"):
        if _USER_HREF_RE.match(a.get("href", "")):
            a.decompose()
    text = _block_text(soup)
    text = re.sub(r"@(?=[\s,.:;!?]|$)", "", text)          # the "@" left over after removing a ping's user link
    text = SIG_TS_RE.sub("", text)
    text = _SIG_REMNANT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"(?m)^:+\s*", "", text)                 # leftover indent colons at line start (the wikitext ":" reply depth)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)\s*[-–—~]+\s*$", "", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    return text.strip(), away


def thread_from_wikitext(section: str, heading: str, page: str, revid: int) -> Thread:
    """Fallback: when DiscussionTools fails (for example, an old version of a very large page times out), split the section wikitext by signatures.

    Method: pipeline.common.signatures.find_signatures finds each signature (a timestamp plus the nearest
    user link before it). The text is cut at the end of each signature, and each piece is one turn;
    level = number of colons or asterisks at line start + 1.
    This is our own rule-of-thumb method (R5.1 only uses it as a comparison), so every question found
    this way is marked turn_source_regex.
    """
    from pipeline.common.signatures import find_signatures
    from pipeline.common.textclean import to_plain_text
    th = Thread(key="", key_author=None, key_ts=None, heading_text=heading, page=page, as_of_revid=revid,
                origin="fallback_regex")
    pos = 0
    for i, sig in enumerate(find_signatures(section)):
        chunk = section[pos:sig.end]
        pos = sig.end
        lead = re.match(r"\s*([:*#]*)", chunk).group(1)
        th.turns.append(Turn(author=norm_user(sig.user), ts=sig.ts, level=len(lead) + 1,
                             comment_id=f"regex-{i}", html="", text=to_plain_text(chunk)))
    if th.turns:
        th.key_author, th.key_ts = th.turns[0].author, th.turns[0].ts
    return th


_MARK_START_RE = re.compile(r'<span[^>]*data-mw-comment-start=""[^>]*id="(?P<id>[^"]+)"[^>]*>')
_CID_RE = re.compile(r"^c-(?P<author>.+?)-(?P<ts>\d{14}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)(?:-(?P<parent>.*))?$")


def thread_from_section_html(html: str, heading: str, page: str, revid: int) -> Thread:
    """Read the DiscussionTools turn markers from the HTML of action=parse&section=N and build a thread.

    Why this works: in the parsed talk page, DiscussionTools wraps each turn in a pair of markers
    <span data-mw-comment-start id="c-<author>-<time>-<parent>"> … <span data-mw-comment-end="<same id>">.
    DiscussionTools creates the id, and the author and time are the signature it recognized (the same
    parser as discussiontoolspageinfo). Parsing only one section is much faster than parsing a whole very
    large talk page, and it does not time out.

    Level: the last part of the id is the parent turn ("<author>-<time>") or the heading. If the parent is
    in this section, level = parent level + 1; otherwise level is 1.
    """
    th = Thread(key="", key_author=None, key_ts=None, heading_text=heading, page=page, as_of_revid=revid,
                origin="fallback_section")
    level_of: dict[str, int] = {}
    for m in _MARK_START_RE.finditer(html or ""):
        cid = m.group("id")
        if cid.startswith("h-"):
            if th.key == "":
                th.key = cid
            continue
        c = _CID_RE.match(cid)
        if not c:
            continue
        end = html.find(f'data-mw-comment-end="{cid}"', m.end())
        body = html[m.end():end if end > 0 else len(html)]
        body = body[:body.rfind("<span")] if end > 0 and "<span" in body else body
        author, ts = heading_key(f"h-{c.group('author')}-{c.group('ts')}")
        me = f"{c.group('author')}-{c.group('ts')}"
        level = level_of.get(c.group("parent") or "", 0) + 1
        level_of[me] = level
        th.turns.append(Turn(author=author, ts=ts, level=level, comment_id=cid, html=body))
    if th.turns:
        th.key_author, th.key_ts = th.turns[0].author, th.turns[0].ts
    return th


def html_links(html: str) -> dict:
    """Pull structural features from turn HTML (R4.5), for use in M6.

    Returns:
        {"wikilinks": [{"title", "ns", "red"}], "n_external": int, "n_list_items": int}
        wikilinks leaves out links to users (signatures). ns is read from the official namespace table;
        redirects from pseudo-namespaces are resolved in M6.
    """
    soup = BeautifulSoup(html or "", "lxml")
    nst = namespace_table()
    links = []
    for a in soup.find_all("a", rel=True):
        rel = " ".join(a.get("rel"))
        href = a.get("href", "")
        if "mw:WikiLink" in rel and not _USER_HREF_RE.match(href):
            title = a.get("title") or href[2:].split("?")[0].replace("_", " ")
            links.append({"title": nst.split(title).full, "ns": nst.split(title).ns,
                          "red": "redlink=1" in href})
    return {"wikilinks": links,
            "n_external": sum(1 for a in soup.find_all("a", rel=True) if "mw:ExtLink" in " ".join(a.get("rel"))),
            "n_list_items": len(soup.find_all("li"))}
