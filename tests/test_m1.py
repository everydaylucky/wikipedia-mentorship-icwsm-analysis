"""
test_m1.py: M1 mentor list (CODE_PLAN.md section 7.1).

Some tests only use the parsing functions (no data needed). The other tests read the M1 output,
and they are skipped automatically if M1 has not been run yet.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
from pipeline.m1_roster.build_roster import (json_entry_in_auto_pool, parse_json_mentors,
                                             parse_wikitext_mentors)

M1 = config.INTERIM / "m1"
needs_m1 = pytest.mark.skipif(not (M1 / "mentor_list.parquet").exists(), reason="M1 has not been run")


# ── Parsing (R1.1) ──────────────────────────────────────────────────────────────
def test_parse_wikitext_ignores_comments_examples_and_subpages():
    text = ("Mentors have to sign up using: <code>[[User:Name]]|Description</code>\n"
            "* [[User:Sungodtemple]]|Welcome!\n"
            "<!-- * [[User:Commented out]]|gone -->\n"
            "* [[User:Gommeh/guestbook|Gommeh]]|hi\n"
            "* [[user:dr_vulpes]]|x\n")
    assert parse_wikitext_mentors(text) == ["Sungodtemple", "Gommeh", "Dr vulpes"]


def test_json_entry_three_eras():
    assert json_entry_in_auto_pool({"automaticallyAssigned": False, "weight": 2}) == (False, True)   # period 2
    assert json_entry_in_auto_pool({"weight": 0}) == (False, True)                                   # period 3
    assert json_entry_in_auto_pool({"weight": 4}) == (True, True)
    assert json_entry_in_auto_pool({"message": "x"}) == (None, False)
    assert parse_json_mentors("not json") is None
    assert parse_json_mentors('{"Mentors": {"32": {"username": "Dreamyshade", "weight": 2}}}') == \
        {"32": {"username": "Dreamyshade", "weight": 2}}


# ── Outputs ────────────────────────────────────────────────────────────────────
@needs_m1
def test_roster_counts_match_eval2():
    ml = pd.read_parquet(M1 / "mentor_list.parquet")
    assert len(ml) == 450 and ml.uid_resolved.all()
    assert ml.is_current.sum() == 291
    assert (ml.currently_in_auto_pool == True).sum() == 210          # noqa: E712


@needs_m1
def test_no_event_after_cutoff():
    ev = pd.read_parquet(M1 / "mentor_events.parquet")
    assert (pd.to_datetime(ev.timestamp, utc=True) < config.DATA_CUTOFF).all()


@needs_m1
def test_manual_list_mentor_xaosflux():
    """Xaosflux (uid 502540) left the automatic list on 2021-06-07, has been on the manual list since 2021-09-20, and has been a mentor ever since."""
    p = pd.read_parquet(M1 / "mentor_periods.parquet")
    x = p[p.mentor_uid == 502540].sort_values("start")
    on_manual = x[(x.source == "wikitext_manual") & (x.in_auto_pool == False)]   # noqa: E712
    assert len(on_manual) >= 1 and on_manual.start.iloc[0].startswith("2021-09-20")


@needs_m1
def test_monthly_auto_pool_matches_public_figure():
    mp = pd.read_csv(config.RESULTS / "m1" / "monthly_pool.csv").set_index("month")
    assert abs(mp.loc["2025-02", "auto_pool"] - 187) <= 5


@needs_m1
def test_names_cover_renames():
    names = pd.read_parquet(M1 / "mentor_names.parquet")
    alext = names[names.mentor_uid == 23085006].name.tolist()      # Rubbish computer → Alextejthompson
    assert "Alextejthompson" in alext and "Rubbish computer" in alext


@needs_m1
def test_display_name_is_latest():
    ml = pd.read_parquet(M1 / "mentor_list.parquet")
    assert ml.loc[ml.mentor_uid == 23085006, "username"].iloc[0] == "Alextejthompson"
