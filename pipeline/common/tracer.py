"""
tracer.py: the fixed set of tracer samples

Module: common layer (PLAN.md section 12; CODE_PLAN.md section 4)

1. What it does
    Defines a locked set of tracer samples, and provides trace() to pull the rows
    for these samples out of any table. Every module's check notebook shows the
    results for this same set of samples. This lets us follow the same people
    through the whole pipeline and check them step by step.

2. Why it is designed this way
    The samples come from three sources:
        1. the 4 mentors and 11 mentees locked in s1_collect_mentors_check.ipynb
           (the regular samples);
        2. the real cases in section 10 of DATA_PROCESSING.md;
        3. one example for each type of "possible processing error" listed for
           each module.
    Every sample is identified by uid or revid, not by user name (R6.1). User names
    are only display labels.

    Once locked, this list does not change. If a new edge case is needed, it can
    only be added at the end, with a written reason.

6. Possible errors and how they are handled
    If trace() cannot find a sample, it does not raise an error. Instead it marks
    False in the "found" column of the result. A missing sample is itself a check
    finding that needs an explanation (for example, the person was removed by the
    sample filters).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Tracer:
    """One tracer sample.

    Attributes:
        label: display name (user name or a short description).
        kind: "mentor", "mentee", "question", or "log".
        uid: user ID (mentor or mentee).
        revid: revision ID (the question, or the evidence revision).
        logid: log ID (claims and similar events).
        why: why this sample was chosen, with the matching rule or issue ID.
    """
    label: str
    kind: str
    uid: int | None = None
    revid: int | None = None
    logid: int | None = None
    why: str = ""


TRACER: tuple[Tracer, ...] = (
    # ── Regular samples: 4 mentors (s1 check notebook)
    Tracer("Sungodtemple", "mentor", uid=40917049, why="wikitext period, active"),
    Tracer("Usedtobecool", "mentor", uid=26021257, why="wikitext period, has left"),
    Tracer("Mujinga", "mentor", uid=1027855, why="JSON period, active"),
    Tracer("Liu1126", "mentor", uid=37823666, why="JSON period, has left"),
    # ── Regular samples: 11 mentees (s1 check notebook)
    Tracer("AlphaBetaGamma (old name AlphaBetaGammsh)", "mentee", uid=45433808, revid=1174241276, why="rename caused the wrong first question to be picked, and signatures under the old name were taken as a replier R6.4, R5.2"),
    Tracer("Couscous1812", "mentee", uid=50301938, revid=1313486198, why="third-party reply R5.3"),
    Tracer("Salicia7", "mentee", uid=49603274, why="no reply"),
    Tracer("JohnDVandevert", "mentee", uid=41541096, why="mentor reply"),
    Tracer("LokiOttawa", "mentee", uid=42457869, why="mentor reply"),
    Tracer("Cjs1", "mentee", uid=42508293, why="same reply sits under two sections"),
    Tracer("SpacedFarmer", "mentee", uid=46980392, why="mentor reply, several rounds"),
    Tracer("Abdulae2011", "mentee", uid=49892101, why="several questions, deciding which is the first question"),
    Tracer("Shadow. 547", "mentee", uid=48497246, why="mentor reply"),
    Tracer("Ssgrayfox", "mentee", uid=47109702, why="mentor reply, long question"),
    Tracer("Saheed Aderemi Adewale Yussuf", "mentee", uid=50375772, why="several questions on the same day, no reply"),
    # ── Real cases (DATA_PROCESSING section 10)
    Tracer("SashankInAnitquity", "question", revid=1343488016, why="primary mentor away, question passed to a backup mentor R4.2, R4.4"),
    Tracer("revdel", "question", revid=1300898069, why="revision deleted, cannot be recovered R5.6"),
    Tracer("Rubbish computer → Alextejthompson", "mentor", uid=23085006, why="mentor renamed R6.3"),
    Tracer("Xaosflux (manual list)", "mentor", uid=502540, why="left automatic assignment and became a manual mentor R1.1, I-49"),
    Tracer("bulk claim (Asilvering)", "log", logid=179458284, why="bulk transfer logged as a claim R3.1"),
    Tracer("edit-a-thon claim (Voidxor)", "log", logid=145507520, why="manual claim R3.1"),
    Tracer("email claim (SignedInteger)", "log", logid=176026789, why="manual claim R3.1"),
    Tracer("adoption claim (CFA)", "log", logid=166371855, why="manual claim R3.1"),
    Tracer("alt account (NM Demo 2)", "log", logid=127195674, why="mentor alt account R3.4"),
    # ── Examples of processing errors (DATA_PROCESSING sections 4 and 5)
    Tracer("方志維 → 楊文祖 page move", "mentor", uid=45329123, why="which mentor owns the questions after a talk page move I-45"),
    Tracer("Modelex (wrong section)", "question", revid=1215863591, why="wrong section located R5.5"),
    Tracer("Deadriafp (away note)", "question", revid=1309525227, why="away note taken as a separate conversation I-39"),
    Tracer("Yoshi24517 (mobile)", "question", revid=1242646265, why="mentor replied from an alt account R5.2"),
    Tracer("Gommeh/guestbook", "question", revid=1337996005, why="subpage inside a signature R5.1"),
    Tracer("36Flames", "mentee", uid=49563912, revid=1320017346, why="possible_forcementor diagnosis"),
)


def tracer_frame() -> pd.DataFrame:
    """Turn the tracer samples into a table."""
    return pd.DataFrame([t.__dict__ for t in TRACER])


def trace(df: pd.DataFrame, *, uid_col: str | None = None, revid_col: str | None = None,
          logid_col: str | None = None) -> pd.DataFrame:
    """Pull the rows for the tracer samples out of table df.

    Args:
        df: an output table from any module.
        uid_col / revid_col / logid_col: the matching column names in df. Any one
            not given is not used for matching.

    Returns:
        A left join of the tracer table with df. The "found" column says whether
        the sample was found in df.
    """
    tf = tracer_frame()
    parts = []
    for key, col in (("uid", uid_col), ("revid", revid_col), ("logid", logid_col)):
        if col is None or col not in df.columns:
            continue
        sub = tf[tf[key].notna()].copy()
        sub[key] = sub[key].astype("int64")
        right = df.copy()
        right[col] = pd.to_numeric(right[col], errors="coerce").astype("Int64")
        merged = sub.merge(right, left_on=key, right_on=col, how="left", indicator=True)
        merged["found"] = merged.pop("_merge") == "both"
        merged["matched_by"] = key
        parts.append(merged)
    if not parts:
        raise ValueError("trace(): give at least one of uid_col / revid_col / logid_col present in df")
    return pd.concat(parts, ignore_index=True)
