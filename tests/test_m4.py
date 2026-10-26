"""
test_m4.py: M4 assignment eligibility (CODE_PLAN.md section 7.1).

The rule functions are tested on real claim-log comments. The other tests read the M4 output and are skipped automatically if M4 has not been run yet.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
from pipeline.m4_eligibility.build_eligibility import classify_claim, decide, is_alt_or_test

M4 = config.INTERIM / "m4"
needs_m4 = pytest.mark.skipif(not (M4 / "eligibility.parquet").exists(), reason="M4 has not been run")


def test_bulk_templates():
    assert classify_claim("Asilvering removed SimpleObjects-9ei from mentorship") == "bulk"
    assert classify_claim("Kqol retired from mentorship.") == "bulk"
    assert classify_claim("I am teaching them how to edit Wikipedia in real life.") == "manual"
    assert classify_claim("Requested mentorship") == "manual"


def test_self_transfer_between_own_accounts():
    assert classify_claim("Moving mentees from main account to alt account specifically for mentoring.",
                          "LuniZunie-Mentor", "LuniZunie") == "self_transfer"


@pytest.mark.parametrize("comment,mentee,claimer,expected", [
    ("This is my alt-account", "NM Demo 2", "Nick Moyes", True),          # fixed sample, logid 127195674
    ("alt account", "LuniZunie-Mentor", "LuniZunie", True),
    ("Test (can I claim myself?)", "Tol", "Tol", True),
    ("Presumed alt acct for Tamaraaraui.", "Tamara Malik", "Mathglot", False),   # someone else's alt account: an ordinary manual claim
    ("test", "Jimbo Wales", "LuniZunie", False),
    ("I met and helped recruit these users at edit-a-thons that I co-taught", "X", "Voidxor", False),
])
def test_alt_or_test_requires_ownership(comment, mentee, claimer, expected):
    assert is_alt_or_test(comment, mentee, claimer) is expected


def test_decide_order_and_unrestricted():
    base = {"current_name": "A", "mentee_name": "A", "creation_action": "create",
            "registration": "2024-01-01T00:00:00Z", "alt_or_test": False, "manual_claim_before_q": False}
    assert decide(base) == (None, [], True)
    r, rs, unr = decide({**base, "creation_action": "byemail"})
    assert r == "created_by_other" and unr is True                    # D10: the reason does not affect unrestricted
    r, rs, unr = decide({**base, "manual_claim_before_q": True, "registration": "2020-01-01T00:00:00Z"})
    assert rs == ["registered_before_mentorship", "manual_claim_before_question"] and unr is False
    assert decide({**base, "current_name": "~2024-12345"})[0] == "temporary_account"


@needs_m4
def test_one_row_per_asker_and_reasons():
    e = pd.read_parquet(M4 / "eligibility.parquet")
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["mentee_uid"])
    assert e.mentee_uid.is_unique and set(e.mentee_uid) == {int(u) for u in q.mentee_uid.dropna()}
    assert (e.eligible == e.exclude_reason.isna()).all()


@needs_m4
def test_tracer_accounts():
    e = pd.read_parquet(M4 / "eligibility.parquet").set_index("mentee_uid")
    assert e.loc[45433808, "exclude_reason"] == "autocreated_account"     # AlphaBetaGamma
    assert bool(e.loc[50301938, "eligible"])                              # Couscous1812
