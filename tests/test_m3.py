"""
test_m3.py: M3 newcomer activity data (CODE_PLAN.md section 7.1).

The parsing functions need no data. The other tests read the M3 output and are skipped automatically
if M3 has not been run yet. Real accounts in the fixed sample: AlphaBetaGamma (uid 45433808, created
automatically from another wiki, 2023-02-23T04:19:02Z) and Couscous1812 (uid 50301938, created by the user).
"""
from __future__ import annotations

import json

import pandas as pd
import pyarrow.dataset as ds
import pytest

import config
from pipeline.common.issues import IssueLog
from pipeline.m3_activity.index_s6 import _HEAD_RE
from pipeline.m3_activity.merge_activity import creation_events

M3 = config.INTERIM / "m3"
needs_m3 = pytest.mark.skipif(not (M3 / "mentee_profile.parquet").exists(), reason="M3 has not been run")


def test_head_regex_reads_escaped_names():
    line = b'{"username": "\\"Mdubeyvns\\"", "role": "mentee", "edits": []}'
    m = _HEAD_RE.match(line)
    assert json.loads(b'"' + m.group(1) + b'"') == '"Mdubeyvns"'
    assert _HEAD_RE.match(b'{"role": "x"}') is None


def test_creation_event_must_belong_to_the_user():
    """Logs that target the user's own user page may contain newusers records of other people (for example, accounts this user created for others). Keep only the record whose userid is the user's own."""
    logs = pd.DataFrame([
        {"uid": 1, "logid": 10, "type": "newusers", "action": "create2", "timestamp": "2024-01-02T00:00:00Z",
         "params": json.dumps({"userid": 2}), "performer": "A"},
        {"uid": 1, "logid": 9, "type": "newusers", "action": "create", "timestamp": "2024-01-01T00:00:00Z",
         "params": json.dumps({"userid": 1}), "performer": "A"},
    ])
    got = creation_events(logs, IssueLog("test"))
    assert got[1]["creation_action"] == "create" and got[1]["creation_logid"] == 9


@needs_m3
def test_every_asker_has_one_profile_row():
    prof = pd.read_parquet(M3 / "mentee_profile.parquet")
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["mentee_uid"])
    assert prof.mentee_uid.is_unique
    assert set(prof.mentee_uid) == {int(u) for u in q.mentee_uid.dropna()}


@needs_m3
def test_no_activity_after_cutoff():
    e = ds.dataset(M3 / "mentee_edits").to_table(columns=["timestamp"]).to_pandas()
    assert (pd.to_datetime(e.timestamp, utc=True) < config.DATA_CUTOFF).all()
    for name in ("mentee_logs", "mentee_abuse"):
        t = pd.read_parquet(M3 / f"{name}.parquet", columns=["timestamp"])
        assert (pd.to_datetime(t.timestamp, utc=True) < config.DATA_CUTOFF).all()


@needs_m3
def test_edits_unique_by_revid():
    e = ds.dataset(M3 / "mentee_edits").to_table(columns=["revid"]).to_pandas()
    assert e.revid.is_unique


@needs_m3
def test_creation_actions_of_tracer_accounts():
    prof = pd.read_parquet(M3 / "mentee_profile.parquet").set_index("mentee_uid")
    assert prof.loc[45433808, "creation_action"] == "autocreate"
    assert prof.loc[50301938, "creation_action"] == "create"
