"""
outcomes.py: formulas for the outcome variables (PLAN.md M5 table; all outcomes from build_outcomes in the paper
version s12, plus fixes)

Module: M5 (CODE_PLAN.md section 5, M5)
Rules: DATA_PROCESSING R7.2 to R7.5, R7.8; PLAN section 0.1

1. What it does
    For one newcomer and one start time a, it computes every outcome variable used in the paper (key names match
    the oc_* names in the paper version's npz).
    Let E be all of this person's edits before the cutoff. E(lo, hi] is the edits with times in (a + lo, a + hi].
    E0 is the edits among these in namespace 0 (articles). Eeven is the edits in even namespaces (not talk pages).
    days(·) is the number of distinct UTC dates.

        primary                    = 1[ |E0(0, 14d]| ≥ 1 ]                          main outcome Y1
        n_mainspace_edits_14d      = |E0(0, 14d]|                                    Y2
        cross_day_any_14d          = 1[ days(E0(0, 14d]) ≥ 2 ]                       Y3
        unique_ns                  = |{ns(e) : e ∈ Eeven(0, 14d]}|                    Y4
        active_days_30d            = days(Eeven(0, 30d])                             Y5
        reverted_any               = 1[ ∃ e ∈ Eeven(0, 14d] reverted before the cutoff ]  Y6
        primary_constructive       = 1[ at least one edit in E0(0, 14d] was not reverted ]
        cross_day_constructive_14d = 1[ days(non-reverted E0(0, 14d]) ≥ 2 ]
        sec2                       = 1[ |E0(14d, 28d]| ≥ 1 ]
        constructive_edit_15_60d   = 1[ non-reverted E0(14d, 60d] is not empty ]
        active_days_14d            = days(Eeven(0, 14d])
        constructive_days_30d      = days(non-reverted Eeven(0, 30d])
        mainspace_15_30d / n_mainspace_15_30d / active_days_15_30d: the (14d, 30d] window
        mainspace_{7,14,21,28,30,60,180}d = 1[ |E0(0, w]| ≥ 1 ]                       window sensitivity
        th_{1,5}plus_{15_28d,29_60d,61_180d} = 1[ |E0(lo, hi]| ≥ 1 or 5 ]
        eligible_{7,14,21,28,30,60,180}d = 1[ a + w ≤ cutoff ]                      whether follow-up is complete (R7.4)

2. Why it is designed this way
    - Windows are measured in hours (timedelta), not calendar days (R7.2). Only edits strictly after the start
      time are used.
    - "Reverted" only counts reverts that happened before the cutoff (R7.8). For an edit with the mw-reverted tag:
      if we found the revert time, it must be earlier than the cutoff. If we did not find the time (these edits
      are far from both the cutoff and the question time, and at most 4.1% of such reverts cross the cutoff), we
      go by the tag.
    - There are only two differences from the paper version: the timing check for reverts, and the cutoff, which
      is now a single fixed time, 2026-04-20T14:00:00Z.

3. Inputs / 4. Outputs
    EditArrays (one person's edits as numpy arrays sorted by time) and a start time; a dict out.

8. How to check
    tests/test_m5.py uses small hand-computed examples; checks/m5_check.ipynb lists, for a fixed sample of
    newcomers, each edit inside the windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

import config

WINDOWS = (7, 14, 21, 28, 30, 60, 180)
ELIGIBLE_WINDOWS = WINDOWS
OC_KEYS = ("primary", "n_mainspace_edits_14d", "primary_constructive", "sec2", "constructive_edit_15_60d",
           "reverted_any", "active_days_14d", "active_days_30d", "constructive_days_30d", "unique_ns",
           "cross_day_constructive_14d", "cross_day_any_14d", "mainspace_15_30d", "n_mainspace_15_30d",
           "active_days_15_30d", "th_1plus_15_28d", "th_5plus_15_28d", "th_1plus_29_60d", "th_5plus_29_60d",
           "th_1plus_61_180d", "th_5plus_61_180d") + tuple(f"mainspace_{w}d" for w in WINDOWS)


@dataclass
class EditArrays:
    """One person's edits before the cutoff (sorted by time).

    Attributes:
        ts: datetime64[s] (UTC, no time zone).
        ns: namespace.
        reverted_by_cutoff: reverted before the cutoff (R7.8).
        reverted_tag: had the mw-reverted tag when downloaded.
        revert_ts: the revert time we found (NaT means not looked up or not found).
    """
    ts: np.ndarray
    ns: np.ndarray
    reverted_by_cutoff: np.ndarray
    reverted_tag: np.ndarray
    revert_ts: np.ndarray

    def reverted_before(self, t: np.datetime64) -> np.ndarray:
        """Already reverted before time t: has the tag, and (when the revert time is known) the revert time is earlier than t."""
        known = ~np.isnat(self.revert_ts)
        return self.reverted_tag & (~known | (self.revert_ts < t))


def _days(ts: np.ndarray) -> int:
    """Number of distinct UTC dates."""
    return int(np.unique(ts.astype("datetime64[D]")).size)


def build_outcomes(e: EditArrays, anchor: datetime, cutoff: datetime | None = None) -> dict:
    """Compute all outcome variables listed in section 1 above. anchor and cutoff are UTC datetimes (a time zone is allowed)."""
    cutoff = cutoff or config.DATA_CUTOFF
    a = np.datetime64(anchor.replace(tzinfo=None), "s")
    d = lambda days: a + np.timedelta64(int(days * 24 * 3600), "s")       # noqa: E731
    post = e.ts > a
    main = post & (e.ns == 0)
    even = post & (e.ns % 2 == 0)
    ok = ~e.reverted_by_cutoff

    def win(mask, lo, hi):
        return mask & (e.ts > d(lo)) & (e.ts <= d(hi))

    m14, e14, e30 = win(main, 0, 14), win(even, 0, 14), win(even, 0, 30)
    oc = {
        "primary": int(m14.any()),
        "n_mainspace_edits_14d": int(m14.sum()),
        "primary_constructive": int((m14 & ok).any()),
        "sec2": int(win(main, 14, 28).any()),
        "constructive_edit_15_60d": int((win(main, 14, 60) & ok).any()),
        "reverted_any": int((e14 & e.reverted_by_cutoff).any()),
        "active_days_14d": _days(e.ts[e14]),
        "active_days_30d": _days(e.ts[e30]),
        "constructive_days_30d": _days(e.ts[e30 & ok]),
        "unique_ns": int(np.unique(e.ns[e14]).size),
        "cross_day_constructive_14d": int(_days(e.ts[m14 & ok]) >= 2),
        "cross_day_any_14d": int(_days(e.ts[m14]) >= 2),
        "mainspace_15_30d": int(win(main, 14, 30).any()),
        "n_mainspace_15_30d": int(win(main, 14, 30).sum()),
        "active_days_15_30d": _days(e.ts[win(even, 14, 30)]),
    }
    for w in WINDOWS:
        oc[f"mainspace_{w}d"] = int(win(main, 0, w).any())
    for name, lo, hi in (("15_28d", 14, 28), ("29_60d", 28, 60), ("61_180d", 60, 180)):
        n = int(win(main, lo, hi).sum())
        oc[f"th_1plus_{name}"] = int(n >= 1)
        oc[f"th_5plus_{name}"] = int(n >= 5)
    cut = np.datetime64(cutoff.replace(tzinfo=None), "s")
    for w in ELIGIBLE_WINDOWS:
        oc[f"eligible_{w}d"] = int(d(w) <= cut)
    return oc


def edit_arrays(df, revert_lookup: dict[int, np.datetime64] | None = None,
                cutoff: datetime | None = None) -> EditArrays:
    """Turn one person's edit table (columns timestamp, ns, tags, revid) into EditArrays.

    revert_lookup: revid -> revert time (M3 revert_times; NaT when there is none).
    """
    cutoff = cutoff or config.DATA_CUTOFF
    df = df.sort_values("timestamp")
    ts = df.timestamp.to_numpy(dtype="datetime64[s]") if df.timestamp.dtype != object else \
        np.array([np.datetime64(t.replace("Z", ""), "s") for t in df.timestamp])
    tag = np.array(["mw-reverted" in list(t) for t in df.tags], dtype=bool)
    rts = np.array([revert_lookup.get(int(r), np.datetime64("NaT")) if revert_lookup else np.datetime64("NaT")
                    for r in df.revid], dtype="datetime64[s]")
    arr = EditArrays(ts=ts, ns=df.ns.to_numpy(dtype=int), reverted_by_cutoff=tag, reverted_tag=tag, revert_ts=rts)
    arr.reverted_by_cutoff = arr.reverted_before(np.datetime64(cutoff.replace(tzinfo=None), "s"))
    return arr
