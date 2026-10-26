"""
checks.py: invariant checks

Module: common layer (CODE_PLAN.md sections 4 and 7)

1. What it does
    Provides a set of checks for things that "must be true": the primary key is
    unique, there are no records after the cutoff date, a join is one-to-one,
    there are no constant columns, and no future information leaks in. A module
    calls them before it writes out data; pytest calls the same functions.

2. Why it is designed this way
    If one of these conditions fails, the code or the data has a basic problem.
    Going on with the computation would only give results that look normal but
    are wrong. Problems seen in the old code (for example, aligning rows by
    their order, I-33; a constant column, I-01; a duplicated column, I-02) are
    all caught by these checks automatically.
    When a check fails, it raises InvariantError and lists up to 10 offending
    records in the message, so you can find the problem directly.

6. Possible errors and how they are handled
    A failed check raises InvariantError. Nothing is "fixed automatically".

8. How to check
    test_checks_* in tests/test_common.py.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

import config
from pipeline.common.timeutil import parse_iso

MAX_SHOW = 10


class InvariantError(AssertionError):
    """An invariant does not hold. The message includes example records that break it."""


def _show(df: pd.DataFrame) -> str:
    """Format the first few offending rows as a string to put in the error message."""
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        return "\n" + df.head(MAX_SHOW).to_string()


def assert_unique(df: pd.DataFrame, key: str | Sequence[str], name: str = "") -> None:
    """Check that the primary key is unique.

    Args:
        df: the table.
        key: name of the primary-key column, or several column names that form a composite key.
        name: table name, used in error messages.
    """
    cols = [key] if isinstance(key, str) else list(key)
    dup = df[df.duplicated(cols, keep=False)]
    if len(dup):
        raise InvariantError(f"[{name}] {len(dup)} rows with duplicated key {cols}:{_show(dup[cols])}")
    nulls = df[df[cols].isna().any(axis=1)]
    if len(nulls):
        raise InvariantError(f"[{name}] {len(nulls)} rows with null key {cols}:{_show(nulls[cols])}")


def assert_no_after_cutoff(df: pd.DataFrame, ts_cols: str | Iterable[str], name: str = "") -> None:
    """Check that the time columns have no value at or after the cutoff (PLAN.md section 0.1). Missing values are allowed.

    Args:
        ts_cols: one or more time columns (strings or datetimes both work).
    """
    cols = [ts_cols] if isinstance(ts_cols, str) else list(ts_cols)
    for c in cols:
        if c not in df.columns:
            raise InvariantError(f"[{name}] column {c!r} not found")
        t = pd.to_datetime(df[c], utc=True, errors="coerce")
        bad = df[t >= config.DATA_CUTOFF]
        if len(bad):
            raise InvariantError(f"[{name}] {len(bad)} rows with {c} >= cutoff "
                                 f"{config.DATA_CUTOFF_ISO}:{_show(bad[[c]])}")


def assert_one_to_one(left: pd.DataFrame, right: pd.DataFrame, key: str, name: str = "") -> None:
    """Check that two tables match one-to-one on key: the key is unique on both sides, and both sides have exactly the same set of keys."""
    assert_unique(left, key, f"{name}.left")
    assert_unique(right, key, f"{name}.right")
    lk, rk = set(left[key]), set(right[key])
    if lk != rk:
        raise InvariantError(f"[{name}] key sets differ: only-left={len(lk - rk)} "
                             f"(e.g. {list(lk - rk)[:5]}), only-right={len(rk - lk)} (e.g. {list(rk - lk)[:5]})")


def assert_no_constant_cols(df: pd.DataFrame, cols: Iterable[str] | None = None,
                            name: str = "", allow: Iterable[str] = ()) -> None:
    """Check that there are no constant columns (like I-01, a variable that is all 0) and no two identical columns (I-02).

    Args:
        cols: columns to check; default is all numeric columns.
        allow: columns that are explicitly allowed to stay (the reason must be written in the data dictionary).
    """
    cols = list(cols) if cols is not None else list(df.select_dtypes("number").columns)
    allow = set(allow)
    const = [c for c in cols if c not in allow and df[c].nunique(dropna=False) <= 1]
    if const:
        raise InvariantError(f"[{name}] constant columns: {const}")
    num = df[cols].select_dtypes("number")
    seen: dict[tuple, str] = {}
    dups = []
    for c in num.columns:
        h = tuple(pd.util.hash_pandas_object(num[c], index=False).values[:2000])
        if h in seen and num[c].equals(num[seen[h]]) and c not in allow:
            dups.append((seen[h], c))
        seen.setdefault(h, c)
    if dups:
        raise InvariantError(f"[{name}] identical column pairs: {dups}")


def assert_no_leakage(df: pd.DataFrame, record_ts: str, reference_ts: str, name: str = "") -> None:
    """Check that every record used by a "before the question" variable is earlier than the reference time (R7.2).

    Args:
        df: each row is one raw record used to compute a covariate, with a record time and a reference time.
        record_ts: name of the record-time column (for example, edit time).
        reference_ts: name of the reference-time column (for example, question time).
    """
    r = pd.to_datetime(df[record_ts], utc=True, errors="coerce")
    q = pd.to_datetime(df[reference_ts], utc=True, errors="coerce")
    bad = df[r >= q]
    if len(bad):
        raise InvariantError(f"[{name}] {len(bad)} records not strictly before {reference_ts}:"
                             f"{_show(bad[[record_ts, reference_ts]])}")


def assert_expected_range(n: int, lo: int, hi: int, what: str) -> None:
    """Check that the row count is in the expected range. This guards against a wrong filter that "quietly" drops or duplicates many rows."""
    if not (lo <= n <= hi):
        raise InvariantError(f"{what}: got {n:,}, expected between {lo:,} and {hi:,}")


def is_before_cutoff_series(s: pd.Series) -> pd.Series:
    """For each value in a time column, tell whether it is before the cutoff (vectorized version of timeutil.before_cutoff)."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t < config.DATA_CUTOFF


__all__ = ["InvariantError", "assert_unique", "assert_no_after_cutoff", "assert_one_to_one",
           "assert_no_constant_cols", "assert_no_leakage", "assert_expected_range",
           "is_before_cutoff_series", "parse_iso"]
