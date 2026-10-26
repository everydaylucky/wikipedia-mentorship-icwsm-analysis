"""
test_m5.py: M5 outcome variables and sample (CODE_PLAN.md section 7.1).

The outcome functions are tested on small examples worked out by hand. The other tests read the M5 output and are skipped automatically if M5 has not been run yet.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import config
from pipeline.m5_sample.build_sample import ascii_rule, detect_language
from pipeline.m5_sample.outcomes import EditArrays, build_outcomes

UTC = timezone.utc
M5 = config.INTERIM / "m5"
needs_m5 = pytest.mark.skipif(not (M5 / "sample.parquet").exists(), reason="M5 has not been run")


def _arr(rows):
    """rows: (time, ns, has tag, revert time or None)."""
    ts = np.array([np.datetime64(t, "s") for t, *_ in rows])
    tag = np.array([r[2] for r in rows], dtype=bool)
    rts = np.array([np.datetime64(r[3], "s") if r[3] else np.datetime64("NaT") for r in rows], dtype="datetime64[s]")
    e = EditArrays(ts=ts, ns=np.array([r[1] for r in rows]), reverted_by_cutoff=tag, reverted_tag=tag, revert_ts=rts)
    e.reverted_by_cutoff = e.reverted_before(np.datetime64(config.DATA_CUTOFF.replace(tzinfo=None), "s"))
    return e


def test_outcomes_hand_example():
    """The clock starts at 2025-01-01 00:00.
    Edits: one at exactly 00:00 (does not count, it must be strictly after the start); an article edit on Jan 2;
    a second article edit on the same day, Jan 2 (reverted); a user page edit on Jan 10; an article edit on
    Jan 15 at 00:00:01 (already past 14 days, counted in hours).
    """
    e = _arr([("2025-01-01T00:00:00", 0, False, None), ("2025-01-02T10:00:00", 0, False, None),
              ("2025-01-02T11:00:00", 0, True, "2025-01-02T12:00:00"), ("2025-01-10T00:00:00", 2, False, None),
              ("2025-01-15T00:00:01", 0, False, None)])
    oc = build_outcomes(e, datetime(2025, 1, 1, tzinfo=UTC))
    assert oc["primary"] == 1 and oc["n_mainspace_edits_14d"] == 2
    assert oc["cross_day_any_14d"] == 0                     # the two article edits are on the same day
    assert oc["unique_ns"] == 2 and oc["active_days_14d"] == 2
    assert oc["reverted_any"] == 1 and oc["primary_constructive"] == 1
    assert oc["sec2"] == 1                                  # the day-15 edit falls in (14d, 28d]
    assert oc["eligible_14d"] == 1


def test_revert_after_cutoff_is_not_counted():
    """Has the mw-reverted tag, but the revert happened after the cutoff time (R7.8): not counted as reverted."""
    e = _arr([("2026-04-10T00:00:00", 0, True, "2026-05-01T00:00:00")])
    oc = build_outcomes(e, datetime(2026, 4, 9, tzinfo=UTC))
    assert oc["reverted_any"] == 0 and oc["primary_constructive"] == 1
    assert oc["eligible_14d"] == 0                          # 4/9 plus 14 days is past the cutoff time


def test_language_rules():
    assert detect_language("How do I add a citation to the article I am writing?")[0] == "en"
    assert detect_language("¿Cómo puedo crear un artículo nuevo sobre mi pueblo en Wikipedia?")[0] == "es"
    assert ascii_rule("¿Cómo puedo crear un artículo nuevo sobre mi pueblo?")   # the paper-version rule also counts Spanish as English
    assert detect_language("How do I create a citation?")[0] == "en"      # langdetect says Italian
    assert detect_language("How do I add my music duo?")[0] == "en"       # langdetect says Welsh
    assert detect_language("DO I HAVE A HISTORY?")[0] == "en"             # all caps: convert to lower case first
    assert detect_language("hi how do i create a citetation")[0] == "en"  # low confidence, but en is in the top three
    assert detect_language("hi")[2] == "ascii_fallback_short"


@needs_m5
def test_sample_invariants():
    s = pd.read_parquet(M5 / "sample.parquet")
    s = s[s.in_sample]
    assert s.mentee_uid.is_unique and s.eligible.all() and s.is_english.all()
    anchor = pd.to_datetime(s.anchor, utc=True)
    q = pd.to_datetime(s.q_ts, utc=True)
    assert (anchor >= q.dt.floor("min")).all()
    assert set(s.treat.unique()) <= {0, 1}
