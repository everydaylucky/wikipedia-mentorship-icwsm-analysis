"""
descriptive.py: descriptive analyses for the camera-ready (long-term trajectories, conversation depth, reply rates)

Module: M8 (analyses added for the ICWSM 2027 camera-ready, answering the meta-review and reviewers 1 to 3)

1. What it does
    Three descriptive analyses. None of them is a causal estimate.
    a. Long-term trajectories (meta-review, reviewer 3). For newcomers with at least 365 days between the anchor and
       the data cutoff, activity in the year after the anchor:
         - any edit in days 181 to 365;
         - an "active editor" period: at least 5 edits in one of the twelve 30-day periods after the anchor
           (the Wikimedia activity level of 5 edits a month), and a "very active" period (at least 100 edits);
         - reached 500 edits in total by day 365 (one of the requirements for extended-confirmed rights);
         - where the edits of days 15 to 365 went, as the share of each person's edits by namespace group, and the
           person's main type (articles, drafts, user and talk pages, other, or no further edits).
       Shown for treated and control newcomers, for control newcomers reweighted to the treated newcomers'
       propensity-score strata (so the two groups are comparable on everything measured before the question),
       and by whether the newcomer edited an article in the first 14 days.
    b. Conversation depth and repeated questions (reviewers 1 and 2). Among treated newcomers: whether the newcomer
       wrote back after the first reply, whether a mentor replied more than once, the number of turns, how long the
       thread stayed active, how many later questions the newcomer asked and to whom. The share who continued
       editing within 14 days is compared between newcomers who wrote back and those who did not, within
       propensity-score strata. Writing back happens after the reply, so this is an association only.
    c. Reply rates (reviewer 2): the share of questions and of sample newcomers without any reply, overall and by year.

2. Why it is designed this way
    The reviewers asked what kind of editors the newcomers become and how the conversation itself relates to
    continued editing. Both questions involve events after the reply, which the causal design cannot cover, so the
    answers are descriptive. Activity levels use Wikimedia's own thresholds rather than thresholds chosen here.
    Only people with a full year of follow-up enter part a, so nobody is counted as inactive just because the
    data end.

3. Inputs
    data/interim/m5/sample.parquet; data/interim/m3/mentee_edits/; data/interim/m2/questions.parquet and turns.jsonl;
    data/analytic/psm_dataset_legacy_layout.npz (for the propensity-score strata).

4. Outputs (results/m8/)
    descriptive_trajectories.csv, descriptive_editor_type.csv, descriptive_conversation.csv,
    descriptive_followup_association.csv, descriptive_reply_rates.csv

5. Key parameters
    FOLLOW_DAYS = 365; ACTIVE = 5 and VERY_ACTIVE = 100 edits per 30-day period; ESTABLISHED = 500 edits.

6. Possible errors and how they are handled
    Newcomers without the needed follow-up are left out of part a and counted in the output.

7. How to run
    python -m pipeline.m8_analysis.descriptive

8. How to check
    checks/m8_check.ipynb (section 7).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import orjson
import pandas as pd
import pyarrow.dataset as ds

import config
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m8_analysis import estimators as est
from pipeline.m8_analysis.run_analysis import covariates, load, subset

FOLLOW_DAYS = 365
ACTIVE, VERY_ACTIVE, ESTABLISHED = 5, 100, 500
OUT = config.RESULTS / "m8"


def ns_group(ns: np.ndarray) -> np.ndarray:
    """Namespace group of each edit: articles (0), drafts (118), user and talk pages (2 and odd numbers), other."""
    return np.select([ns == 0, ns == 118, (ns == 2) | (ns % 2 == 1)], ["articles", "drafts", "user_and_talk"], "other")


def strata_weights(rows: pd.DataFrame, D: dict) -> pd.Series:
    """Weights that make control newcomers look like treated newcomers on the propensity score (ATT weights).

    The propensity score is fitted with the main design (5-fold cross-fitting, all covariates) on the rows given;
    within each of the 10 strata, a treated newcomer has weight 1 and a control newcomer n_treated / n_control.
    """
    S = subset(D, np.isin(D["cid_order"].astype(np.int64), rows.first_question_revid.to_numpy()))
    X = covariates(S)
    d = S["y_treat"].astype(float)
    st = est.make_strata(est.fit_propensity(X[:, X.std(axis=0) > 0], d).e_hat, config.N_STRATA)
    f = pd.DataFrame({"first_question_revid": S["cid_order"].astype(np.int64), "d": d, "st": st})
    n = f.groupby(["st", "d"]).size().unstack(fill_value=0)
    ratio = (n[1.0] / n[0.0].replace(0, np.nan)).fillna(0)
    f["w"] = np.where(f.d == 1, 1.0, f.st.map(ratio))
    return rows.first_question_revid.map(f.set_index("first_question_revid").w)


def wmean(x: pd.Series, w: pd.Series) -> float:
    """Weighted mean, ignoring missing values."""
    ok = x.notna() & w.notna()
    return float(np.average(x[ok], weights=w[ok])) if ok.any() and w[ok].sum() > 0 else float("nan")


def trajectories(s: pd.DataFrame, D: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Part a: activity in the year after the anchor."""
    anchor = pd.to_datetime(s.anchor, utc=True)
    s = s[anchor + pd.Timedelta(days=FOLLOW_DAYS) <= pd.Timestamp(config.DATA_CUTOFF)].copy()
    log.info("[trajectories] %d newcomers with %d days of follow-up", len(s), FOLLOW_DAYS)
    e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "timestamp", "ns"], filter=ds.field("uid").isin([int(u) for u in s.mentee_uid])).to_pandas()
    e["ts"] = pd.to_datetime(e.timestamp, utc=True)
    e = e.merge(s[["mentee_uid", "anchor"]].assign(a=lambda x: pd.to_datetime(x.anchor, utc=True)),
                left_on="uid", right_on="mentee_uid")
    e["day"] = (e.ts - e.a).dt.total_seconds() / 86400
    pre = e[e.day <= 0].groupby("uid").size()
    post = e[(e.day > 0) & (e.day <= FOLLOW_DAYS)].copy()
    post["period"] = np.ceil(post.day / 30).clip(upper=12).astype(int)
    per = post.groupby(["uid", "period"]).size().unstack(fill_value=0)
    late = post[post.day > 180].groupby("uid").size()
    later = post[post.day > 14].assign(g=lambda x: ns_group(x.ns.to_numpy()))
    shares = later.groupby(["uid", "g"]).size().unstack(fill_value=0)
    shares = shares.div(shares.sum(axis=1), axis=0)
    uid = s.mentee_uid.astype(int)
    s["any_edit_days_181_365"] = uid.map(late).fillna(0).gt(0).astype(float).to_numpy()
    s["active_period"] = uid.map(per.max(axis=1)).fillna(0).ge(ACTIVE).astype(float).to_numpy()
    s["very_active_period"] = uid.map(per.max(axis=1)).fillna(0).ge(VERY_ACTIVE).astype(float).to_numpy()
    s["reached_500_edits"] = (uid.map(pre).fillna(0) + uid.map(post.groupby("uid").size()).fillna(0)).ge(ESTABLISHED).astype(float).to_numpy()
    for g in ("articles", "drafts", "user_and_talk", "other"):
        s[f"share_{g}"] = uid.map(shares[g] if g in shares else pd.Series(dtype=float)).to_numpy()
    main = shares.idxmax(axis=1).where(shares.max(axis=1) >= 0.5, "mixed")
    s["editor_type"] = uid.map(main).fillna("no edits after day 14").to_numpy()
    s["w"] = strata_weights(s, D).to_numpy()
    metrics = ["any_edit_days_181_365", "active_period", "very_active_period", "reached_500_edits",
               "share_articles", "share_drafts", "share_user_and_talk", "share_other"]
    groups = [("treated", s.treat == 1, False), ("control", s.treat == 0, False),
              ("control, reweighted to treated", s.treat == 0, True),
              ("edited an article in days 1-14", s.oc_primary == 1, False),
              ("no article edit in days 1-14", s.oc_primary == 0, False)]
    rows = []
    for name, m, weighted in groups:
        g = s[m]
        w = g.w if weighted else pd.Series(1.0, index=g.index)
        rows.append({"group": name, "N": int(len(g)), **{k: wmean(g[k], w) for k in metrics}})
    traj = pd.DataFrame(rows)
    types = []
    for name, m, weighted in groups:
        g = s[m]
        w = g.w if weighted else pd.Series(1.0, index=g.index)
        t = w.groupby(g.editor_type).sum() / w.sum()
        types.append({"group": name, **t.to_dict()})
    log.info("[trajectories]\n%s", traj.round(3).to_string(index=False))
    return traj, pd.DataFrame(types).fillna(0.0)


def conversation(s: pd.DataFrame, D: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Part b: conversation depth and repeated questions among treated newcomers."""
    t = s[s.treat == 1].copy()
    want = set(t.first_question_revid)
    info = {}
    for line in open(config.INTERIM / "m2" / "turns.jsonl", "rb"):
        r = orjson.loads(line)
        if r["question_revid"] not in want:
            continue
        ts = [x for x in r["turns"] if x.get("ts")]
        info[r["question_revid"]] = ts
    fr = pd.to_datetime(t.first_reply_ts, utc=True)
    rows = []
    for rv, f in zip(t.first_question_revid, fr):
        ts = info.get(rv, [])
        after = [x for x in ts if pd.Timestamp(x["ts"]) > f]
        rows.append({"first_question_revid": rv, "n_turns": len(ts),
                     "mentee_wrote_back": any(x["role"] == "mentee" for x in after),
                     "mentor_replied_again": sum(x["role"] in ("page_mentor", "primary_mentor", "other_mentor") for x in ts) > 1,
                     "thread_days": (max(pd.Timestamp(x["ts"]) for x in ts) - min(pd.Timestamp(x["ts"]) for x in ts)).total_seconds() / 86400 if ts else 0.0})
    c = t.merge(pd.DataFrame(rows), on="first_question_revid", how="left")
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["mentee_uid", "q_ts", "page_mentor_uid"])
    q = q.dropna(subset=["mentee_uid"]).assign(mentee_uid=lambda x: x.mentee_uid.astype(int))
    first = c.set_index("mentee_uid")[["q_ts", "page_mentor_uid"]]
    q = q[q.mentee_uid.isin(first.index)]
    later = q[pd.to_datetime(q.q_ts, utc=True) > q.mentee_uid.map(pd.to_datetime(first.q_ts, utc=True))]
    c["n_later_questions"] = c.mentee_uid.map(later.groupby("mentee_uid").size()).fillna(0).to_numpy()
    same = later[later.page_mentor_uid == later.mentee_uid.map(first.page_mentor_uid)]
    c["n_later_questions_same_mentor"] = c.mentee_uid.map(same.groupby("mentee_uid").size()).fillna(0).to_numpy()
    summary = pd.DataFrame([{
        "treated newcomers": len(c), "mentee wrote back after the first reply": c.mentee_wrote_back.mean(),
        "a mentor replied more than once": c.mentor_replied_again.mean(), "median turns in the thread": c.n_turns.median(),
        "median days the thread was active": c.thread_days.median(), "asked at least one later question": (c.n_later_questions > 0).mean(),
        "later question to the same mentor's page": (c.n_later_questions_same_mentor > 0).mean()}]).T.reset_index()
    summary.columns = ["measure", "value"]
    # association: continued editing by whether the newcomer wrote back, compared within propensity-score strata
    S = subset(D, np.isin(D["cid_order"].astype(np.int64), c.first_question_revid.to_numpy()))
    X = covariates(S)
    order = pd.Series(range(len(S["cid_order"])), index=S["cid_order"].astype(np.int64))
    c = c.set_index("first_question_revid").loc[order.index].reset_index()
    wrote = c.mentee_wrote_back.astype(float).to_numpy()
    st = est.make_strata(est.fit_propensity(X[:, X.std(axis=0) > 0], wrote).e_hat, config.N_STRATA)
    y = c.oc_primary.astype(float).to_numpy()
    diff = est.att_stratified(y, wrote, st).att
    assoc = pd.DataFrame([{"comparison": "treated newcomers who wrote back vs those who did not",
                           "N": len(c), "wrote_back": int(wrote.sum()), "share_continued_wrote_back": y[wrote == 1].mean(),
                           "share_continued_no": y[wrote == 0].mean(), "raw_difference": y[wrote == 1].mean() - y[wrote == 0].mean(),
                           "difference_within_strata": diff,
                           "note": "association only: writing back happens after the reply and may reflect engagement"}])
    log.info("[conversation]\n%s\n%s", summary.to_string(index=False), assoc.round(4).T.to_string())
    return summary, assoc


def reply_rates(s_all: pd.DataFrame, log) -> pd.DataFrame:
    """Part c: share without any reply, for all tagged questions and for the analysis sample, overall and by year."""
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["q_ts", "has_any_reply"])
    q["year"] = pd.to_datetime(q.q_ts, utc=True).dt.year
    s = s_all[s_all.in_sample].assign(year=lambda x: pd.to_datetime(x.q_ts, utc=True).dt.year)
    rows = []
    for name, df, col in (("all tagged questions", q, "has_any_reply"), ("analysis sample (first questions)", s, "treat")):
        for year, g in [("all", df)] + list(df.groupby("year")):
            rows.append({"set": name, "year": year, "N": len(g), "share_without_reply": 1 - g[col].astype(float).mean()})
    out = pd.DataFrame(rows)
    log.info("[reply rates]\n%s", out[out.year == "all"].round(4).to_string(index=False))
    return out


def main(argv: list[str] | None = None) -> None:
    base_parser("M8 descriptive analyses for the camera-ready.").parse_args(argv)
    log = setup_logging("m8_descriptive")
    t0 = time.time()
    check_upstream("m8_descriptive", ["m7"])
    OUT.mkdir(parents=True, exist_ok=True)
    s_all = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet")
    s = s_all[s_all.in_sample].copy()
    s["mentee_uid"] = s.mentee_uid.astype(int)
    Dall = load()
    D = subset(Dall, Dall["in_sample"])
    traj, types = trajectories(s, D, log)
    summary, assoc = conversation(s, D, log)
    res = {"descriptive_trajectories": traj, "descriptive_editor_type": types, "descriptive_conversation": summary,
           "descriptive_followup_association": assoc, "descriptive_reply_rates": reply_rates(s_all, log)}
    for k, v in res.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    write_manifest("m8_descriptive", inputs=[config.INTERIM / "m5" / "sample.parquet"],
                   outputs=[OUT / f"{k}.csv" for k in res], params={"follow_days": FOLLOW_DAYS},
                   counts={k: len(v) for k, v in res.items()}, started=t0)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
