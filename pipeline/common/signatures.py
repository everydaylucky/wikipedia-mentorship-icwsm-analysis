"""
signatures.py: our own signature parser (used only as a cross-check)

Module: common layer (CODE_PLAN.md section 4)
Rules: DATA_PROCESSING R5.1

1. What it does
    Finds every signature in a piece of talk page wikitext. A signature is a UTC
    timestamp plus the nearest user link in front of it. For each signature it
    returns the signer (with any subpage removed and known alt accounts mapped to
    the main account) and the time.

2. Why it is designed this way (and why it is only a cross-check)
    The MediaWiki signature ~~~~ expands to "user page link + talk page link +
    timestamp" (Wikipedia:Signatures). So "the nearest user link before the
    timestamp" is the standard rule of thumb for finding the signer. But it is not
    an authoritative method. Custom signatures, signatures added by templates, and
    quoted signatures of other people can all fool it.

    The official split into individual comments uses Wikipedia's own
    DiscussionTools parser (R5.1, M2). The code here is only used to:
        1. compare with the DiscussionTools result; records where the two disagree
           go into the anomaly list;
        2. act as a fallback when DiscussionTools cannot cover a case (for example,
           when the page revision has been deleted).

3. Inputs
    A wikitext string.

4. Outputs
    A list of Signature objects, sorted by position in the text.

6. Possible errors and how they are handled
    No user link within WINDOW characters before a timestamp: that timestamp is
    not counted as a signature (for example, a time mentioned in the body text).

8. How to check
    tests/test_common.py::test_signatures_* (includes the Yoshi24517 (mobile) and
    Gommeh/guestbook cases).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from pipeline.common.timeutil import SIG_TS_RE, parse_sig_ts
from pipeline.common.users import canonical_person, strip_subpage

#: How far back (in characters) to look for a user link before a timestamp.
#: A signature's user link is usually right next to the timestamp.
WINDOW = 400

#: Links that point to a user: [[User:X]], [[User talk:X]], [[Special:Contributions/X]]
#: (a pipe and an anchor are allowed).
USER_LINK_RE = re.compile(
    r"\[\[\s*:?\s*(?:User(?:[ _]talk)?|Special:Contrib(?:ution)?s)\s*[:/]\s*([^\]|#]+?)\s*(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]",
    re.I,
)


@dataclass(frozen=True)
class Signature:
    """One signature.

    Attributes:
        start, end: position of the timestamp in the text (end is where the
            timestamp ends).
        raw_user: the user name exactly as written in the link (not normalized,
            may include a subpage).
        user: the signer, normalized, with the subpage removed and known alt
            accounts mapped to the main account (R5.1, R5.2).
        ts: signature time (UTC).
    """
    start: int
    end: int
    raw_user: str
    user: str
    ts: datetime


def find_signatures(text: str) -> list[Signature]:
    """Find all signatures in text.

    Args:
        text: talk page wikitext.

    Returns:
        A list of Signature objects, sorted by position.

    Example:
        "...thanks! [[User:Gommeh/guestbook|Gommeh]] 16:52, 12 February 2026 (UTC)"
        -> user == "Gommeh" (the subpage is removed)
    """
    out: list[Signature] = []
    for m in SIG_TS_RE.finditer(text):
        ts = parse_sig_ts(m.group(0))
        if ts is None:
            continue
        window = text[max(0, m.start() - WINDOW):m.start()]
        links = list(USER_LINK_RE.finditer(window))
        if not links:
            continue
        raw = links[-1].group(1)
        user = canonical_person(raw) or strip_subpage(raw) or raw
        out.append(Signature(m.start(), m.end(), raw, user, ts))
    return out


def signer_at(text: str, pos: int) -> Signature | None:
    """Return the first signature after position pos, i.e. the signer of that passage."""
    for s in find_signatures(text):
        if s.start >= pos:
            return s
    return None
