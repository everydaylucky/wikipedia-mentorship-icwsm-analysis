"""
timeutil.py: parsing times and checking the data cutoff

Module: common layer (CODE_PLAN.md section 4)
Rules: PLAN.md section 0.1 (fixed cutoff date); DATA_PROCESSING R5.1 (signature
       times), R7.2 (windows measured in hours)

1. What it does
    Turns the different time formats found in the project into one form: a UTC
    datetime that carries its time zone. It also provides the single cutoff check,
    before_cutoff().

2. Why it is designed this way
    The data has three time formats:
        1. ISO 8601 from the API, for example "2024-03-27T15:55:04Z";
        2. times in talk page signatures, for example "15:55, 27 March 2024 (UTC)".
           This is what the MediaWiki signature ~~~~ expands to
           (Wikipedia:Signatures). It is precise to the minute;
        3. ISO strings without a time zone mark, found in older data.
    The old code dropped the time zone in some places (naive datetimes) and kept it
    in others. Comparing the two kinds can give wrong answers or raise errors.
    Here everything becomes a UTC aware datetime, so all comparisons happen in the
    same time zone.

    The cutoff check is written only once, here. A time counts as "before the
    cutoff" only if it is strictly earlier than DATA_CUTOFF; the cutoff moment
    itself is excluded. Every module must call before_cutoff() and must not write
    its own comparison.

3. Inputs / 4. Outputs
    A string or a datetime; the output is a UTC aware datetime.

6. Possible errors and how they are handled
    A string that cannot be parsed returns None. The caller decides whether to
    log an issue or raise an error. With parse_iso(strict=True), a ValueError is
    raised right away.

8. How to check
    test_time_* in tests/test_common.py.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import config

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

#: Regex for a signature timestamp: HH:MM, D Month YYYY (UTC). Several spaces are
#: allowed after the comma.
SIG_TS_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}),\s*(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+)\s+(?P<y>\d{4})\s*\(UTC\)"
)


def parse_iso(value: str | datetime | None, *, strict: bool = False) -> datetime | None:
    """Turn an ISO 8601 string or a datetime into a UTC aware datetime.

    Args:
        value: for example "2024-03-27T15:55:04Z", "2024-03-27T15:55:04+00:00",
               "2024-03-27T15:55:04" (treated as UTC), or a datetime object.
        strict: if True, raise ValueError when the value cannot be parsed; if
               False, return None.

    Returns:
        A UTC aware datetime; None when value is empty.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        if strict:
            raise
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_sig_ts(text: str) -> datetime | None:
    """Parse a signature time like "15:55, 27 March 2024 (UTC)" into a UTC aware datetime.

    Notes:
        Month names are case insensitive. A month name we do not recognize (for
        example, a localized month name) returns None. Signature times are only
        precise to the minute, so seconds are set to 0.
    """
    m = SIG_TS_RE.search(text)
    if not m:
        return None
    month = _MONTHS.get(m.group("mon").lower())
    if month is None:
        return None
    try:
        return datetime(int(m.group("y")), month, int(m.group("d")),
                        int(m.group("h")), int(m.group("m")), tzinfo=timezone.utc)
    except ValueError:  # for example, 31 February
        return None


def before_cutoff(ts: str | datetime | None, cutoff: datetime | None = None) -> bool:
    """Check whether a time is before the cutoff (PLAN.md section 0.1). This is the
    only cutoff check in the whole project.

    Args:
        ts: the time (string or datetime).
        cutoff: defaults to config.DATA_CUTOFF.

    Returns:
        True if ts is strictly earlier than cutoff. Returns False if ts is empty or
        cannot be parsed: if we do not know when a record happened, we cannot show
        that it is before the cutoff, so it does not enter the analysis.

    Example:
        before_cutoff("2026-04-23T23:59:59Z") is True;
        before_cutoff("2026-04-24T00:00:00Z") is False (the cutoff moment itself
        is excluded).
    """
    dt = parse_iso(ts)
    if dt is None:
        return False
    return dt < (cutoff or config.DATA_CUTOFF)


def hours_between(start: datetime, end: datetime) -> float:
    """Number of hours between two times (end - start). Can be negative. All window
    calculations use this function (R7.2)."""
    return (end - start).total_seconds() / 3600.0


def add_days(ts: datetime, days: float) -> datetime:
    """Add a number of days to ts, counted as exact hours (days * 24 hours), not
    rounded to calendar days."""
    return ts + timedelta(hours=days * 24.0)


def utc_date(ts: datetime) -> date:
    """Get the UTC date. Used to count "active days" (the number of distinct UTC dates)."""
    return ts.astimezone(timezone.utc).date()


def to_iso(ts: datetime | None) -> str | None:
    """Turn a datetime back into a "YYYY-MM-DDTHH:MM:SSZ" string, for writing files."""
    if ts is None:
        return None
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
