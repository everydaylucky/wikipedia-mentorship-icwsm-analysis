"""
_show.py: shared helpers for the check notebooks

Module: checks (CODE_PLAN.md section 7.2)

1. What it does
    Small helpers that every check notebook uses:
        Checks      collects pass/fail checks and prints them as [OK ] / [FAIL] lines;
        compare     puts a value recomputed by hand next to the value stored in the data, one row per field;
        links       URLs to open a revision, diff, contributions page or log entry on Wikipedia;
        pick        draws a random sample and says which seed was used.

2. Why it is designed this way
    The notebooks are meant for a person to verify the data by eye: open the raw record, see the processed
    value, see an independent recomputation, and see whether they agree. Keeping these helpers in one place
    makes every notebook print the same, easy to read format.

3. How to use
    from checks._show import Checks, compare, pick, oldid, diff, contribs, logid
    ck = Checks(); ck.add("primary key is unique", df.id.is_unique); ck.report()
    compare({"n_edits": 3}, {"n_edits": 3})       # prints a table with a "match" column
"""
from __future__ import annotations

from urllib.parse import quote

import numpy as np
import pandas as pd

WIKI = "https://en.wikipedia.org/w/index.php"


def oldid(revid) -> str:
    """Link to one revision as it looked when it was saved."""
    return f"{WIKI}?oldid={int(revid)}"


def diff(revid) -> str:
    """Link to the change made by one revision (compared with the revision before it)."""
    return f"{WIKI}?diff={int(revid)}"


def contribs(name: str, end: str = "2026-04-20") -> str:
    """Link to a user's contributions page, ending at the data cutoff date."""
    return f"https://en.wikipedia.org/wiki/Special:Contributions/{quote(str(name).replace(' ', '_'))}?end={end}"


def logid(i) -> str:
    """Link to one log entry."""
    return f"{WIKI}?title=Special:Log&logid={int(i)}"


def pick(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A random sample of at most n rows. Change the seed at the top of the notebook to draw new rows."""
    return df.sample(min(n, len(df)), random_state=seed) if len(df) else df


def _same(a, b) -> bool:
    """Equality that treats NaN == NaN, numbers within 1e-9, and list-like values element by element."""
    if isinstance(a, (list, tuple, set, np.ndarray)) or isinstance(b, (list, tuple, set, np.ndarray)):
        return list(a) == list(b)
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        na = a is None or (isinstance(a, float) and np.isnan(a))
        nb = b is None or (isinstance(b, float) and np.isnan(b))
        return na and nb
    try:
        return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-9))
    except (TypeError, ValueError):
        return a == b


def compare(hand: dict, data: dict, title: str | None = None) -> bool:
    """Print hand-computed values next to stored values. Returns True when every field matches."""
    rows = [{"field": k, "recomputed": hand[k], "in data": data.get(k), "match": _same(hand[k], data.get(k))}
            for k in hand]
    t = pd.DataFrame(rows)
    if title:
        print(title)
    print(t.to_string(index=False))
    ok = bool(t["match"].all())
    print("  -> all fields match" if ok else f"  -> MISMATCH in: {list(t.loc[~t['match'], 'field'])}")
    return ok


class Checks:
    """Collects named True/False checks and prints them. report() raises if any check failed."""

    def __init__(self):
        self.items: list[tuple[str, bool]] = []

    def add(self, name: str, ok) -> bool:
        self.items.append((name, bool(ok)))
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        return bool(ok)

    def report(self, strict: bool = False) -> None:
        bad = [n for n, ok in self.items if not ok]
        print(f"\n{len(self.items) - len(bad)} / {len(self.items)} checks passed")
        if bad:
            print("failed:", bad)
            if strict:
                raise AssertionError(f"failed checks: {bad}")
