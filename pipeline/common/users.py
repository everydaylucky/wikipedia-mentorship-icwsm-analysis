"""
users.py: normalizing user names, and a time-aware "name -> uid" index

Module: common layer (CODE_PLAN.md section 4)
Rules: DATA_PROCESSING R5.1 (remove subpages), R5.2 (mentor alt accounts),
       R6.1 to R6.3 (uid is the main key)

1. What it does
    1. norm_user(): turns a user name from any source (signatures, log titles,
       rosters, the API) into the standard MediaWiki form. This is the only way to
       check whether two user names are the same.
    2. strip_subpage(): removes the subpage part of "User:Name/Subpage".
    3. UserIndex: stores every name each uid has used, with the time span when
       each name was valid. It answers "at this moment, who was this name?" and
       helps identify mentor alt accounts.

2. Why it is designed this way
    MediaWiki's user name rules (Manual:Page title, $wgCapitalLinks):
        - an underscore is the same as a space;
        - the first letter is capitalized automatically (English Wikipedia has
          $wgCapitalLinks turned on);
        - runs of whitespace count as one space.
    The old code missed the underscore and first-letter rules, and lost mentors
    such as Dr_vulpes and lajmmoore (recorded in the eval2 project notes). So the rules are
    written out in full here, and the whole project uses only this one version.

    Names change, and a name can be registered again by someone else. So
    "name -> uid" must take time into account. For example, User talk:方志維 was
    moved to User talk:楊文祖 on 2025-06-30 (the same person switched to a new
    account with a different uid). When looking up by name, we need to know which
    time span the question falls in.

3. Inputs
    The caller fills UserIndex one record at a time with add() (sources can be the
    M1 roster, M3 profiles, or API results).

4. Outputs
    Function return values. UserIndex can be exported as a table with to_frame().

6. Possible errors and how they are handled
    - One name maps to several uids at the same moment: lookup() returns None and
      adds the name to index.ambiguous. The caller logs the issue
      `ambiguous_username`.
    - Name not found: returns None. The caller decides whether to look it up
      online (R6.3).

8. How to check
    test_norm_user_* and test_user_index_* in tests/test_common.py.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

import config
from pipeline.common.timeutil import parse_iso

# All Unicode "whitespace-like" characters (including the no-break space U+00A0,
# the narrow no-break space U+202F, and so on)
_WS_RE = re.compile(r"[\s   -     　_]+")
# Invisible direction marks and zero-width characters. They sometimes appear in
# signatures, so they are removed before comparing names
_INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


def norm_user(name: str | None) -> str | None:
    """Turn a user name into the standard MediaWiki form (R6.2).

    Args:
        name: a user name from any source; it may start with "User:".

    Returns:
        The normalized user name; None if the input is empty.

    Rules (applied in order):
        0. decode HTML entities ("&amp;" -> "&"), the same as MediaWiki
           Title::newFromText;
        1. remove invisible characters;
        2. remove the "User:" or "User talk:" prefix (case insensitive);
        3. turn underscores and all kinds of whitespace into a single space, and
           trim spaces at both ends;
        4. capitalize the first letter ($wgCapitalLinks). Only the first character
           changes; the rest stays as it is.

    Example:
        norm_user("dr_vulpes") == "Dr vulpes"
        norm_user("User talk:Foo__Bar ") == "Foo Bar"
        norm_user("Dhruv &amp; baby") == "Dhruv & baby"
    """
    if name is None:
        return None
    s = _INVISIBLE_RE.sub("", unicodedata.normalize("NFC", html.unescape(str(name))))
    s = re.sub(r"^\s*(?:user|user talk)\s*:", "", s, flags=re.I)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return None
    first = s[0].upper()
    # Some characters become two characters when capitalized (for example ß -> SS).
    # MediaWiki does not do this, so we keep the original character
    if len(first) != 1:
        first = s[0]
    return first + s[1:]


def strip_subpage(name: str | None) -> str | None:
    """Remove the subpage part: "Gommeh/guestbook" -> "Gommeh" (R5.1).

    Notes:
        Signatures sometimes link to the user's own subpage (a guestbook, an
        archive, and so on). These links still point to the user, so when finding
        the signer we keep only the part before "/".
    """
    if name is None:
        return None
    return norm_user(str(name).split("/", 1)[0])


def canonical_person(name: str | None) -> str | None:
    """Normalize the name, and map known mentor alt accounts to the main account
    (R5.2, config.MENTOR_ALT_ACCOUNTS).

    Example:
        canonical_person("Yoshi24517 (mobile)") == "Yoshi24517"
    """
    n = strip_subpage(name)
    if n is None:
        return None
    alt = {norm_user(k): norm_user(v) for k, v in config.MENTOR_ALT_ACCOUNTS.items()}
    return alt.get(n, n)


@dataclass
class NameSpan:
    """A name that belongs to one uid during a span of time.

    Attributes:
        uid: user ID.
        name: the normalized name.
        start: start time (included); None means "from long ago".
        end: end time (not included); None means "up to now".
        source: where the evidence comes from, for example "m1_roster",
            "api_users", "move_log".
    """
    uid: int
    name: str
    start: datetime | None = None
    end: datetime | None = None
    source: str = ""

    def covers(self, at: datetime | None) -> bool:
        """Whether this time span includes the moment at. If at is None, it counts
        as included."""
        if at is None:
            return True
        if self.start is not None and at < self.start:
            return False
        if self.end is not None and at >= self.end:
            return False
        return True


@dataclass
class UserIndex:
    """A time-aware "name <-> uid" index (R6.1 to R6.3).

    Usage:
        idx = UserIndex()
        idx.add(45329123, "方志維", end="2025-06-30T20:35:19Z", source="move_log")
        idx.add(49604360, "楊文祖", start="2025-04-30T00:04:11Z", source="api_users")
        idx.lookup("方志維", at="2024-07-19T10:36:17Z")  # -> 45329123
    """
    spans: list[NameSpan] = field(default_factory=list)
    ambiguous: set[str] = field(default_factory=set)
    _by_name: dict[str, list[NameSpan]] = field(default_factory=dict, repr=False)

    def add(self, uid: int, name: str, start=None, end=None, source: str = "") -> None:
        """Add one record saying "this name belongs to uid". start/end can be
        strings or datetimes."""
        n = norm_user(name)
        if n is None or uid is None:
            return
        span = NameSpan(int(uid), n, parse_iso(start), parse_iso(end), source)
        # Keep only one copy of a record with the same uid, name and time span
        for s in self._by_name.get(n, []):
            if (s.uid, s.start, s.end) == (span.uid, span.start, span.end):
                return
        self.spans.append(span)
        self._by_name.setdefault(n, []).append(span)

    def lookup(self, name: str | None, at=None) -> int | None:
        """Find which uid the name belonged to at a given moment.

        Args:
            name: user name. It first goes through canonical_person() (subpage
                removed, alt accounts mapped).
            at: the moment; None means any time.

        Returns:
            The uid if exactly one matches. None if nothing matches or several
            match (when several match, the name is added to ambiguous).
        """
        n = canonical_person(name)
        if n is None:
            return None
        at_dt = parse_iso(at)
        hits = {s.uid for s in self._by_name.get(n, []) if s.covers(at_dt)}
        if len(hits) == 1:
            return next(iter(hits))
        if len(hits) > 1:
            self.ambiguous.add(n)
        return None

    def names_of(self, uid: int) -> list[str]:
        """All names a uid has used (normalized, in order of first appearance)."""
        out: list[str] = []
        for s in self.spans:
            if s.uid == uid and s.name not in out:
                out.append(s.name)
        return out

    def to_frame(self) -> pd.DataFrame:
        """Export as a table for checking and saving: uid, name, start, end, source."""
        return pd.DataFrame([s.__dict__ for s in self.spans],
                            columns=["uid", "name", "start", "end", "source"])
