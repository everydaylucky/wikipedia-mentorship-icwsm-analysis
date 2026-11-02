"""
paper_counts.py: counts and shares that the paper quotes directly from the sample (sample construction, question
types, who replied first, covariate blocks)

Module: M8 (camera-ready; added 2026-11-02 so that these paper numbers come from a result file)

1. What it does
    Writes one table (name, value) with:
        sample construction (Table sample): tagged first questions, the newcomers removed at each step (in the
            paper's grouping: hidden and empty question text together; test, alternative and forced local accounts
            together), and the final sample;
        question types (Table qtypes): share of the 34,073 newcomers with each of Q1 to Q5;
        who replied first (Table mentor_reply, main sample of 33,518): (a) the mentor of the page where the question
            was asked replied first, (b) someone else replied first and the assigned mentor later, (c) only others
            replied, (d) no reply;
        covariates (Section on covariates): the number of columns in each block of the PS model.

2. Why it is designed this way
    These numbers were computed by hand for the paper; a result file makes them checkable by the paper number check.
    (a) uses M2's first_replier_role ("page_mentor"); (b) is every other newcomer whose assigned mentor replied
    (mentor_reply_versions counts a + b = 24,563).

3. Inputs
    data/interim/m5/sample.parquet; data/interim/m6/question_features.parquet; the analytic npz (through load).

4. Outputs
    results/m8/paper_counts.csv

7. How to run
    python -m pipeline.m8_analysis.paper_counts

8. How to check
    a + b must equal the Treated count of the assigned-mentor row of mentor_reply_versions.csv (asserted); the
    sample-construction steps must add up to the final sample (asserted).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import config
from pipeline.common.manifest import write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import legacy_parity as lp
from pipeline.m8_analysis.run_analysis import add_followup_masks, load, subset

OUT = config.RESULTS / "m8"
#: paper row of Table sample -> M5 exclude_step values
FUNNEL = [("Account created by another user", ["m4:created_by_other"]),
          ("Automatically created account", ["m4:autocreated_account"]),
          ("Registered before mentorship began", ["m4:registered_before_mentorship"]),
          ("No account creation record", ["m4:no_creation_log"]),
          ("Claimed by a mentor before asking", ["m4:manual_claim_before_question"]),
          ("Other account exclusions (test, forced local)", ["m4:alt_or_test_account", "m4:forced_local_account"]),
          ("Conversation not found", ["thread_not_found"]),
          ("Reply recorded before the question", ["reply_before_question"]),
          ("Question removed in the next edit", ["question_removed_immediately"]),
          ("Question text hidden or empty", ["question_text_hidden", "question_text_empty"]),
          ("Not in English", ["not_english"])]


def main(argv: list[str] | None = None) -> None:
    base_parser("Counts and shares quoted in the paper.").parse_args(argv)
    log = setup_logging("m8_paper_counts")
    t0 = time.time()
    rows = []
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet")
    steps = s.exclude_step.fillna("in sample").value_counts()
    rows.append(("sample: tagged first questions", len(s)))
    used = {"in sample"}
    for label, keys in FUNNEL:
        rows.append((f"sample: removed: {label}", int(sum(steps.get(k, 0) for k in keys))))
        used |= set(keys)
    assert set(steps.index) <= used, f"exclusion steps not in the table: {set(steps.index) - used}"
    rows.append(("sample: final", int(steps["in sample"])))
    assert len(s) - sum(v for k, v in rows if k.startswith("sample: removed")) == steps["in sample"]

    Dall = add_followup_masks(load())
    D = subset(Dall, Dall["in_sample"])
    S = s.drop_duplicates("first_question_revid").set_index("first_question_revid").loc[D["cid_order"].astype(np.int64)]
    qt = pd.read_parquet(config.INTERIM / "m6" / "question_features.parquet").set_index("first_question_revid")
    for c in ("q_substantive", "q_referent", "q_appraisal", "q_normative", "q_own_work"):
        rows.append((f"qtype share: {c}", float(qt.loc[S.index, c].mean())))

    ms = D["keep__primary"].astype(bool)
    treat = S.treat.to_numpy() == 1
    mentor = S.has_mentor_reply.fillna(False).to_numpy().astype(bool)
    a = ms & treat & (S.first_replier_role.to_numpy() == "page_mentor")
    b = ms & treat & mentor & ~a
    rows += [("who replied: (a) page mentor first", int(a.sum())), ("who replied: (b) other first, mentor later", int(b.sum())),
             ("who replied: (c) only others", int((ms & treat & ~mentor).sum())),
             ("who replied: (d) no reply", int((ms & ~treat).sum()))]
    mrv = pd.read_csv(OUT / "mentor_reply_versions.csv")
    ab = mrv.loc[mrv.Spec.str.startswith("Assigned mentor replied vs no reply (only others dropped)"), "Treated"].iloc[0]
    assert a.sum() + b.sum() == ab, f"(a) + (b) = {a.sum() + b.sum()}, mentor_reply_versions {ab}"
    assert not (a & ~mentor).any()

    B = lp.blocks(D)
    for k in ("E", "Qtext", "Qpersp", "Qtype", "emb20", "temporal"):
        rows.append((f"covariates: {k}", int(B[k].shape[1])))
    rows.append(("covariates: total", int(sum(B[k].shape[1] for k in ("E", "Qtext", "Qpersp", "Qtype", "emb20", "temporal")))))
    rows.append(("covariates: year-month indicators", int(sum(str(c).startswith("ym_") for c in D["temporal_cols"]))))

    out = pd.DataFrame(rows, columns=["name", "value"])
    out.to_csv(OUT / "paper_counts.csv", index=False)
    for r in out.itertuples():
        log.info("%-60s %s", r.name, r.value)
    write_manifest("m8_paper_counts", inputs=[config.INTERIM / "m5" / "sample.parquet",
                                              config.INTERIM / "m6" / "question_features.parquet"],
                   outputs=[OUT / "paper_counts.csv"], params={}, counts={"rows": len(out)}, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
