"""
revert_times.py: M3 part 4: for edits that carry the mw-reverted tag, find out "when they were reverted"

Module: M3 (PLAN.md M3; CODE_PLAN.md section 5, M3)
Rules: DATA_PROCESSING R7.2 (variables measured before the question use only records from before the question), R7.5 (definition of a reverted edit), R7.8; PLAN section 0.1

1. What it does
    1. candidates(): picks the edits whose revert time we need to look up. They carry the mw-reverted tag and
       are less than window_days days before "the person's first question" or "the cutoff time"
       (for edits further back, the revert does not cross either of these moments, see section 2).
    2. revert_time(): for one edit r, searches the history of its page for "the first revision that restores the
       page content to some version from before r":
       - take the sha1 of r and of up to 15 revisions before it (MediaWiki looks back at most 15 revisions
         when it decides whether something is a revert, $wgRevertedTagMaxDepth);
       - walk forward from r one revision at a time; the first one whose sha1 equals some version before r is
         the revert, and its time is the revert time;
       - when an older edit is undone, the page content does not always return to an old version. In that case we
         use an undo or revert revision whose edit summary mentions r (it carries the mw-undo, mw-rollback or
         mw-manual-revert tag, and its summary contains the revision ID of r);
       - if the page was deleted or the revision was hidden: record page_deleted; the revert time is unknown.
    3. outcome_candidates() (phase "outcomes", added 2026-10-26 for the 48-hour revert rule): every edit with the
       mw-reverted tag made by a main-sample newcomer between their first question and 30 days after their
       outcome start time (M5 anchor). The outcome "reverted within 48 hours of the edit" needs the revert time of
       each of these edits, wherever it lies.
    4. build(): writes a table with the revert time of every candidate edit, for use in M7:
       outcome variables count only reverts that happened before the cutoff time; covariates measured before the
       question count only reverts that happened before the question.

2. Why it is designed this way
    The mw-reverted tag is added to the reverted edit by a background job only after the revert has happened
    (MediaWiki 1.36, RevertedTagUpdate). The tag itself does not store the revert time. So the same edit can have
    different tags depending on when we fetched it. Using today's tags would mix in reverts that happened after the
    cutoff time, or after the question (for example, an edit made before the question that the mentor reverted the
    next day).
    MediaWiki itself detects a "manual revert" by comparing sha1: the new revision's content is exactly the same as
    some earlier version (ManualRevert). We use the same method here to work out which revision did the revert,
    and when.
    Window: in the sample phase we measure, on a random sample, how long it takes from an edit to its revert.
    We pick a number of days that covers the great majority of reverts. Edits outside the window are treated as
    "the revert happened before the boundary", and the share of errors this causes goes into the report.

3. Inputs
    data/interim/m3/mentee_edits/, data/interim/m2/questions.parquet.

4. Outputs
    data/raw/m3/revert_times.jsonl: one line per edit {revid, uid, method, revert_revid, revert_ts, lag_hours, fetched_at}
    data/interim/m3/revert_times.parquet: same content (build phase)

6. Possible errors and how they are handled
    - Page deleted or revision hidden: method="page_deleted", revert_ts is empty. M7 keeps the tag as fetched for
      these edits and counts them separately;
    - No revert found after looking at MAX_FORWARD later revisions: method="not_found" (the tag may come from a
      revert that sha1 cannot detect). The tag is kept here too.

7. How to run
    python -m pipeline.m3_activity.revert_times --online --phase sample   # first, measure the lag distribution
    python -m pipeline.m3_activity.revert_times --online --phase all --window-days N
    python -m pipeline.m3_activity.revert_times --online --phase outcomes   # after M5: edits in the outcome windows
    python -m pipeline.m3_activity.revert_times --phase build

8. How to check
    The "revert times" section of checks/m3_check.ipynb.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pyarrow.dataset as ds

import config
from pipeline.common.api import ApiError, MediaWikiClient
from pipeline.common.io import iter_jsonl, write_table
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import parse_iso

RAW = config.RAW / "m3"
OUT = config.INTERIM / "m3"
MAX_DEPTH = 15          # $wgRevertedTagMaxDepth
MAX_FORWARD = 500       # at most this many later revisions are checked
REVERT_TAGS = {"mw-rollback", "mw-undo", "mw-manual-revert"}
PROPS = "ids|timestamp|sha1|tags|comment|user"


def candidates(window_days: float | None) -> pd.DataFrame:
    """Edits with the mw-reverted tag that are less than window_days days before the person's first question or the cutoff time (None means all of them)."""
    e = ds.dataset(OUT / "mentee_edits").to_table(columns=["uid", "revid", "title", "timestamp", "tags"]).to_pandas()
    r = e[e.tags.map(lambda t: "mw-reverted" in list(t))].copy()
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["mentee_uid", "q_ts"]).dropna()
    fq = pd.to_datetime(q.groupby("mentee_uid").q_ts.min(), utc=True)
    r["ts"] = pd.to_datetime(r.timestamp, utc=True)
    r["first_q"] = r.uid.map(fq)
    if window_days is None:
        return r
    w = pd.Timedelta(days=window_days)
    near_q = (r.ts < r.first_q) & (r.ts >= r.first_q - w)
    near_c = r.ts >= config.DATA_CUTOFF - w
    return r[near_q | near_c]


def outcome_candidates(days: float = 30) -> pd.DataFrame:
    """Edits with the mw-reverted tag made by main-sample newcomers after their first question and at most `days`
    days after their outcome start time (M5 sample.parquet, column anchor)."""
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet", columns=["in_sample", "mentee_uid", "q_ts", "anchor"])
    s = s[s.in_sample]
    lo = pd.to_datetime(s.q_ts, utc=True)
    hi = pd.concat([lo, pd.to_datetime(s.anchor, utc=True)], axis=1).max(axis=1) + pd.Timedelta(days=days)
    win = pd.DataFrame({"lo": lo.to_numpy(), "hi": hi.to_numpy()}, index=s.mentee_uid.astype("int64").to_numpy())
    e = ds.dataset(OUT / "mentee_edits").to_table(columns=["uid", "revid", "title", "timestamp", "tags"],
                                                  filter=ds.field("uid").isin(list(win.index))).to_pandas()
    r = e[e.tags.map(lambda t: "mw-reverted" in list(t))].copy()
    ts = pd.to_datetime(r.timestamp, utc=True)
    w = win.loc[r.uid.astype("int64")]
    return r[(ts.to_numpy() > w.lo.to_numpy()) & (ts.to_numpy() <= w.hi.to_numpy())]


def _revisions(client, params) -> list[dict]:
    d = client.get({"action": "query", "prop": "revisions", "rvslots": "main", "rvprop": PROPS, **params})
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return []
    return pages[0].get("revisions", [])


def revert_time(client, revid: int) -> dict:
    """The reverting revision and revert time for one edit (see section 1, point 2)."""
    d = client.get({"action": "query", "prop": "revisions", "revids": revid, "rvprop": "ids|timestamp"})
    q = d.get("query", {})
    if q.get("badrevids") or not q.get("pages") or q["pages"][0].get("missing"):
        return {"method": "page_deleted"}
    pageid = q["pages"][0]["pageid"]
    older = _revisions(client, {"pageids": pageid, "rvstartid": revid, "rvdir": "older", "rvlimit": MAX_DEPTH + 1})
    if not older or older[0]["revid"] != revid:
        return {"method": "page_deleted"}
    t0 = parse_iso(older[0]["timestamp"])
    prior = {x.get("sha1") for x in older[1:] if x.get("sha1")}
    seen, cont = 0, None
    while seen < MAX_FORWARD:
        params = {"pageids": pageid, "rvstartid": revid, "rvdir": "newer", "rvlimit": 50}
        if cont:
            params["rvcontinue"] = cont
        d = client.get({"action": "query", "prop": "revisions", "rvslots": "main", "rvprop": PROPS, **params})
        revs = (d.get("query", {}).get("pages") or [{}])[0].get("revisions", [])
        for x in revs:
            if x["revid"] == revid:
                continue
            seen += 1
            undo_ref = set(x.get("tags") or []) & REVERT_TAGS and str(revid) in (x.get("comment") or "")
            if (x.get("sha1") and x["sha1"] in prior) or undo_ref:
                ts = parse_iso(x["timestamp"])
                return {"method": "sha1" if x.get("sha1") in prior else "undo_comment", "revert_revid": x["revid"],
                        "revert_ts": x["timestamp"], "lag_hours": round((ts - t0).total_seconds() / 3600, 3)}
        cont = d.get("continue", {}).get("rvcontinue")
        if not cont:
            break
    return {"method": "not_found", "scanned": seen}


def page_ids(client, revids: list[int]) -> dict[int, int | None]:
    """Revision ID -> page ID (50 at a time). None if the page was deleted or the revision was hidden."""
    out = {}
    for i in range(0, len(revids), 50):
        d = client.get({"action": "query", "prop": "revisions", "revids": "|".join(map(str, revids[i:i + 50])),
                        "rvprop": "ids"})
        for pg in d.get("query", {}).get("pages", []):
            for r in pg.get("revisions", []):
                out[int(r["revid"])] = pg.get("pageid")
    return {r: out.get(r) for r in revids}


def revert_times_on_page(client, pageid: int, revids: list[int]) -> dict[int, dict]:
    """Looks up several candidate edits on the same page together, so the page history is fetched only once (the method is the same as in revert_time).

    We take the MAX_DEPTH revisions before the earliest candidate, then read forward from it until every candidate
    has a revert, or until we have looked at MAX_FORWARD revisions after the last candidate.
    """
    revids = sorted(revids)
    older = _revisions(client, {"pageids": pageid, "rvstartid": revids[0], "rvdir": "older", "rvlimit": MAX_DEPTH + 1})
    hist = list(reversed(older[1:]))          # revisions before the earliest candidate, oldest first
    pending, out, cont, after_last = set(revids), {}, None, 0
    pos: dict[int, int] = {}
    while pending and after_last < MAX_FORWARD:
        params = {"pageids": pageid, "rvstartid": revids[0], "rvdir": "newer", "rvlimit": "max"}
        if cont:
            params["rvcontinue"] = cont
        d = client.get({"action": "query", "prop": "revisions", "rvslots": "main", "rvprop": PROPS, **params})
        revs = (d.get("query", {}).get("pages") or [{}])[0].get("revisions", [])
        for x in revs:
            hist.append(x)
            k = len(hist) - 1
            if x["revid"] in pending:
                pos[x["revid"]] = k
            if x["revid"] > revids[-1]:
                after_last += 1
            for r in list(pending):
                if r not in pos or pos[r] >= k:
                    continue
                i = pos[r]
                prior = {h.get("sha1") for h in hist[max(0, i - MAX_DEPTH):i] if h.get("sha1")}
                undo_ref = set(x.get("tags") or []) & REVERT_TAGS and str(r) in (x.get("comment") or "")
                if (x.get("sha1") and x["sha1"] in prior) or undo_ref:
                    t0, t1 = parse_iso(hist[i]["timestamp"]), parse_iso(x["timestamp"])
                    out[r] = {"method": "sha1" if x.get("sha1") in prior else "undo_comment", "revert_revid": x["revid"],
                              "revert_ts": x["timestamp"], "lag_hours": round((t1 - t0).total_seconds() / 3600, 3)}
                    pending.discard(r)
        cont = d.get("continue", {}).get("rvcontinue")
        if not cont:
            break
    for r in pending:
        out[r] = {"method": "not_found" if r in pos else "page_deleted"}
    return out


def collect_grouped(client, cand: pd.DataFrame, log, out_name: str) -> None:
    """Looks up revert times grouped by page (used in the full phase). Appends to out_name and can resume after a stop."""
    path = RAW / out_name
    done = {r["revid"] for r in iter_jsonl(path)} if path.exists() else set()
    todo = cand[~cand.revid.isin(done)]
    log.info("revert-time lookups (grouped by page): %d edits (%d already done)", len(todo), len(cand) - len(todo))
    pids = page_ids(client, [int(r) for r in todo.revid])
    todo = todo.assign(pageid=todo.revid.map(pids))
    t0, n = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        def write(row, res):
            f.write(json.dumps({"revid": int(row.revid), "uid": int(row.uid), "ts": row.timestamp,
                                "fetched_at": datetime.now(timezone.utc).isoformat(), **res}) + "\n")
        for r in todo[todo.pageid.isna()].itertuples():
            write(r, {"method": "page_deleted"})
        groups = list(todo.dropna(subset=["pageid"]).groupby("pageid"))
        for j, (pid, g) in enumerate(groups, 1):
            try:
                res = revert_times_on_page(client, int(pid), [int(x) for x in g.revid])
            except ApiError as e:
                res = {int(x): {"method": "api_error", "error": str(e)} for x in g.revid}
            for r in g.itertuples():
                write(r, res[int(r.revid)])
            f.flush()
            n += len(g)
            if j % 200 == 0:
                log.info("  pages %d/%d, edits %d (%.0f s, api %s)", j, len(groups), n, time.time() - t0, client.stats)


def collect(client, cand: pd.DataFrame, log, out_name: str) -> None:
    """Looks up revert times one edit at a time and appends to out_name (to resume after a stop, revids already written are skipped)."""
    path = RAW / out_name
    done = {r["revid"] for r in iter_jsonl(path)} if path.exists() else set()
    todo = cand[~cand.revid.isin(done)].sort_values(["title", "timestamp"])
    log.info("revert-time lookups: %d (%d already done)", len(todo), len(cand) - len(todo))
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo.itertuples(), 1):
            try:
                res = revert_time(client, int(r.revid))
            except ApiError as e:
                res = {"method": "api_error", "error": str(e)}
            f.write(json.dumps({"revid": int(r.revid), "uid": int(r.uid), "ts": r.timestamp,
                                "fetched_at": datetime.now(timezone.utc).isoformat(), **res}) + "\n")
            f.flush()
            if i % 200 == 0:
                log.info("  %d/%d (%.0f s, api %s)", i, len(todo), time.time() - t0, client.stats)


def lag_report(path) -> dict:
    """Distribution of the time from edit to revert: quantiles (in hours), and the share of reverts that took more than 1, 3, 7, 14 and 30 days."""
    df = pd.DataFrame(list(iter_jsonl(path)))
    lag = df.lag_hours.dropna()
    return {"n": len(df), "methods": df.method.value_counts().to_dict(),
            "lag_hours_quantiles": {str(k): round(float(v), 2) for k, v in lag.quantile([.5, .9, .95, .99, .995, .999]).items()},
            "share_over_days": {d: round(float((lag > 24 * d).mean()), 5) for d in (1, 3, 7, 14, 30)}}


def build(log) -> None:
    """Combines the sample results and the full results, and writes revert_times.parquet."""
    rows = []
    for name in ("revert_times_sample.jsonl", "revert_times.jsonl"):
        p = RAW / name
        if p.exists():
            rows += list(iter_jsonl(p))
    df = pd.DataFrame(rows).drop_duplicates("revid", keep="last")
    write_table(OUT / "revert_times.parquet", df)
    log.info("revert times: %d edits; %s", len(df), df.method.value_counts().to_dict())


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("M3 revert times for mw-reverted edits near the first question or the cutoff.")
    ap.add_argument("--phase", choices=["sample", "all", "outcomes", "build"], required=True)
    ap.add_argument("--window-days", type=float, default=30)
    ap.add_argument("--sample-size", type=int, default=1500)
    args = ap.parse_args(argv)
    log = setup_logging(f"m3_revert_{args.phase}")
    if args.phase == "build":
        build(log)
        return
    client = MediaWikiClient(online=args.online)
    if args.phase == "sample":
        cand = candidates(None)
        s = cand.sample(min(args.sample_size, len(cand)), random_state=config.SEED)
        collect(client, s, log, "revert_times_sample.jsonl")
        rep = lag_report(RAW / "revert_times_sample.jsonl")
        (config.RESULTS / "m3_revert_lag_sample.json").write_text(json.dumps(rep, indent=1))
        log.info("lag report: %s", rep)
    elif args.phase == "outcomes":
        collect_grouped(client, outcome_candidates(), log, "revert_times.jsonl")
    else:
        collect_grouped(client, candidates(args.window_days), log, "revert_times.jsonl")


if __name__ == "__main__":
    main(sys.argv[1:])
