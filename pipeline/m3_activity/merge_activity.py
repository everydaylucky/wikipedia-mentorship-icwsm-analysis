"""
merge_activity.py: M3 part 3 (offline): merges the s6 data with the top-up data, giving each asker's profile, edits, logs and abuse filter records

Module: M3 (PLAN.md M3; CODE_PLAN.md section 5, M3)
Rules: DATA_PROCESSING R2.2 (how the account was created), R6.1 (identity by uid), R7.1; PLAN section 0.1 (data frozen at the cutoff date)

1. What it does
    1. Profile table mentee_profile: one row per asker, key mentee_uid.
       Fields: source (s6 / topup), late_fetch, name at fetch time, current name, registration time, edit count,
       block status, user groups, the account creation log (action, time, performer, logid, source), and the
       number of edits, logs and abuse filter records.
    2. Edits mentee_edits/shard=NN.parquet: s6 and top-up merged, duplicates removed by revid. For people whose
       s6 record was incomplete and who were fetched again, we take the union of both sides (source="s6+topup"):
       s6 keeps edits that were deleted later, and the top-up adds edits made after s6.
    3. Logs mentee_logs.parquet and abuse filter records mentee_abuse.parquet: merged and de-duplicated the same way.
    4. A final cutoff check: if any table contains a record on or after the cutoff date, stop with an error.

2. Why it is designed this way
    - How the account was created (needed for D10) comes from the newusers log, and we require that the userid in
      the log parameters is this same person:
      the s6 logs were queried both as "performed by the person" and as "targeting the person's user page", so
      accounts created by someone else are also recorded;
      for people not in s6, we use the entry found in the top-up phase by registration time (collect_topup.creation_log).
    - People fetched in the top-up get late_fetch=True: the top-up happened after the cutoff date, so edits deleted
      after the cutoff and tags added after the cutoff cannot be recovered.

3. Inputs
    data/interim/m3/s6/ (index_s6); data/raw/m3/topup.jsonl, creation.jsonl (collect_topup);
    data/interim/m2/questions.parquet; config.RAW_FILES["registration"] (current names).

4. Outputs (data/interim/m3/)
    mentee_profile.parquet, mentee_edits/shard=NN.parquet, mentee_logs.parquet, mentee_abuse.parquet

6. Possible errors and how they are handled
    - An asker is neither in s6 nor in the top-up: source="none", logged as issue `asker_without_activity`
      (M5 drops these people and lists them in the sample construction table);
    - The same uid has several newusers logs of its own: take the earliest, log issue `multiple_creation_logs`;
    - Records on or after the cutoff date: stop with an error (FatalIssue). This means an earlier cutoff filter has a gap.

7. How to run
    python -m pipeline.m3_activity.merge_activity     # about 1 minute

8. How to check
    checks/m3_check.ipynb; tests/test_m3.py.
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from pipeline.common.checks import assert_no_after_cutoff, assert_unique
from pipeline.common.io import iter_jsonl, write_table
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import before_cutoff
from pipeline.m3_activity.index_s6 import EDIT_SCHEMA, N_SHARDS, askers

S6 = config.INTERIM / "m3" / "s6"
RAW = config.RAW / "m3"
OUT = config.INTERIM / "m3"
CREATION_ACTIONS = ("create", "create2", "byemail", "autocreate", "forcecreatelocal", "newusers")


def load_topup() -> dict[int, dict]:
    """Top-up results: uid -> record (from topup.jsonl and topup_stale.jsonl; if the same uid was written more than once, the last one wins)."""
    out = {}
    for name in ("topup.jsonl", "topup_stale.jsonl"):
        p = RAW / name
        if p.exists():
            out.update({r["uid"]: r for r in iter_jsonl(p)})
    return out


def topup_edits(rec: dict) -> list[dict]:
    """Converts top-up edits to the same fields as s6."""
    out = []
    for e in rec.get("contribs") or []:
        if not before_cutoff(e.get("timestamp")):
            continue
        out.append({"uid": rec["uid"], "revid": e.get("revid"), "parentid": e.get("parentid"), "ns": e.get("ns"),
                    "title": e.get("title"), "timestamp": e.get("timestamp"), "size": e.get("size"),
                    "sizediff": e.get("sizediff"), "tags": list(e.get("tags") or []), "minor": bool(e.get("minor")),
                    "new": bool(e.get("new")), "comment": e.get("comment")})
    return out


def merge_edits(topup: dict[int, dict], log) -> dict[int, int]:
    """Merges edits one shard at a time: takes the s6 shard, adds the top-up edits, and removes duplicates by revid. Returns the number of edits per person."""
    (OUT / "mentee_edits").mkdir(parents=True, exist_ok=True)
    by_shard: dict[int, list[dict]] = {}
    for uid, rec in topup.items():
        by_shard.setdefault(uid % N_SHARDS, []).extend(topup_edits(rec))
    counts: dict[int, int] = {}
    for shard in range(N_SHARDS):
        src = S6 / "edits" / f"shard={shard:02d}.parquet"
        parts = []
        if src.exists():
            t = pq.read_table(src)
            # If a top-up person is also in s6 (the s6 record was incomplete), take the union of both: s6 has edits deleted after the cutoff, the top-up has edits made after s6
            parts.append(t.append_column("source", pa.array(["s6"] * t.num_rows)))
        if by_shard.get(shard):
            t2 = pa.Table.from_pylist(by_shard[shard], schema=EDIT_SCHEMA)
            parts.append(t2.append_column("source", pa.array(["topup"] * t2.num_rows)))
        if not parts:
            continue
        df = pa.concat_tables(parts).to_pandas().drop_duplicates("revid", keep="first")
        assert_no_after_cutoff(df, "timestamp")
        for uid, n in df.groupby("uid").size().items():
            counts[int(uid)] = int(n)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT / "mentee_edits" / f"shard={shard:02d}.parquet")
    log.info("edits merged: %d users, %d edits", len(counts), sum(counts.values()))
    return counts


def merge_logs(topup: dict[int, dict]) -> pd.DataFrame:
    """Merges logs: s6 (without the top-up people) + top-up, duplicates removed by (uid, logid)."""
    s6 = pd.read_parquet(S6 / "logs.parquet").assign(source="s6")
    rows = [{"uid": uid, "logid": e.get("logid"), "type": e.get("type"), "action": e.get("action"),
             "timestamp": e.get("timestamp"), "ns": e.get("ns"), "title": e.get("title"), "performer": e.get("user"),
             "params": json.dumps(e.get("params") or {}, ensure_ascii=False), "comment": e.get("comment"),
             "source": "topup"}
            for uid, rec in topup.items() for e in rec.get("logs") or [] if before_cutoff(e.get("timestamp"))]
    df = pd.concat([s6, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["uid", "logid"])
    df["logid"] = df.logid.astype("Int64")
    return df


def merge_abuse(topup: dict[int, dict]) -> pd.DataFrame:
    """Merges abuse filter records: s6 (without the top-up people) + top-up, duplicates removed by (uid, abuse_id)."""
    s6 = pd.read_parquet(S6 / "abuse.parquet").assign(source="s6")
    rows = [{"uid": uid, "abuse_id": e.get("id"), "filter_id": str(e.get("filter_id") or ""), "filter": e.get("filter"),
             "action": e.get("action"), "result": e.get("result"), "timestamp": e.get("timestamp"), "ns": e.get("ns"),
             "title": e.get("title"), "source": "topup"}
            for uid, rec in topup.items() for e in rec.get("abuse") or [] if before_cutoff(e.get("timestamp"))]
    df = pd.concat([s6, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["uid", "abuse_id"])
    df["abuse_id"] = df.abuse_id.astype("Int64")
    return df


def creation_events(logs: pd.DataFrame, issues: IssueLog) -> dict[int, dict]:
    """The account creation log of each asker: the newusers log entry whose params.userid equals the person's own uid (R2.2).

    Order of sources: first the s6 and top-up logs (source="log"); if none is found there, creation.jsonl
    (found by registration time, source="lookup").
    """
    out: dict[int, dict] = {}
    nu = logs[(logs.type == "newusers") & logs.action.isin(CREATION_ACTIONS)]
    for uid, g in nu.groupby("uid"):
        own = [r for r in g.itertuples() if str(json.loads(r.params or "{}").get("userid")) == str(uid)]
        if not own:
            continue
        own.sort(key=lambda r: r.timestamp)
        if len(own) > 1:
            issues.add("multiple_creation_logs", rule="R2.2", action="flagged", ids={"uid": int(uid)},
                       detail={"logids": [int(r.logid) for r in own]})
        r = own[0]
        out[int(uid)] = {"creation_action": r.action, "creation_ts": r.timestamp, "creation_performer": r.performer,
                         "creation_logid": int(r.logid), "creation_source": "log"}
    p = RAW / "creation.jsonl"
    for r in iter_jsonl(p) if p.exists() else []:
        if r["uid"] in out or not r.get("event"):
            continue
        e = r["event"]
        out[r["uid"]] = {"creation_action": e.get("action"), "creation_ts": e.get("timestamp"),
                         "creation_performer": e.get("user"), "creation_logid": e.get("logid"),
                         "creation_source": "lookup"}
    return out


def build_profile(ask: set[int], topup: dict[int, dict], n_edits: dict, logs: pd.DataFrame, abuse: pd.DataFrame,
                  creation: dict, issues: IssueLog) -> pd.DataFrame:
    """Profile table: one row per asker."""
    s6 = pd.read_parquet(S6 / "name_to_uid.parquet").set_index("uid")
    reg = json.loads(config.RAW_FILES["registration"].read_text())
    n_logs = logs.groupby("uid").size().to_dict()
    n_abuse = abuse.groupby("uid").size().to_dict()
    rows = []
    for uid in sorted(ask):
        now = reg.get(str(uid)) or {}
        if uid in topup:
            u = topup[uid]["user"]
            base = {"source": "s6+topup" if uid in s6.index else "topup", "late_fetch": True, "fetched_at": topup[uid]["fetched_at"],
                    "name_at_fetch": u.get("name"), "registration": u.get("registration") or now.get("registration"),
                    "editcount": u.get("editcount"), "blocked": "blockid" in u, "groups": list(u.get("groups") or []),
                    "hidden_or_missing": bool(u.get("missing") or u.get("hidden"))}
        elif uid in s6.index:
            p = s6.loc[uid]
            base = {"source": "s6", "late_fetch": False, "fetched_at": "2026-04-23",
                    "name_at_fetch": p["name"], "registration": p["registration"] or now.get("registration"),
                    "editcount": p["editcount"], "blocked": bool(p["blocked"]), "groups": list(p["groups"]),
                    "hidden_or_missing": False}
        else:
            issues.add("asker_without_activity", rule="R6.1", action="flagged", ids={"uid": uid}, detail=now)
            base = {"source": "none", "late_fetch": None, "fetched_at": None, "name_at_fetch": None,
                    "registration": now.get("registration"), "editcount": now.get("editcount"),
                    "blocked": now.get("blocked"), "groups": [], "hidden_or_missing": None}
        c = creation.get(uid) or {"creation_action": None, "creation_ts": None, "creation_performer": None,
                                  "creation_logid": None, "creation_source": "none"}
        rows.append({"mentee_uid": uid, **base, "current_name": now.get("name"), **c,
                     "n_edits": n_edits.get(uid, 0), "n_logs": n_logs.get(uid, 0), "n_abuse": n_abuse.get(uid, 0)})
    df = pd.DataFrame(rows)
    df["editcount"] = pd.to_numeric(df.editcount, errors="coerce").astype("Int64")
    df["creation_logid"] = pd.to_numeric(df.creation_logid, errors="coerce").astype("Int64")
    return df


def main(argv: list[str] | None = None) -> None:
    base_parser("M3 merge: profile, edits, logs and abuse log per asker.").parse_args(argv)
    log = setup_logging("m3")
    t0 = time.time()
    check_upstream("m3", ["m3_index"])
    issues = IssueLog("m3")
    ask = askers()
    topup = load_topup()
    # Some people fetched in the top-up under the old cutoff date (4/24) only asked after the new cutoff time, so they are no longer askers. Drop them
    extra = set(topup) - ask
    topup = {u: r for u, r in topup.items() if u in ask}
    log.info("askers %d; top-up records %d (dropped %d no longer askers)", len(ask), len(topup), len(extra))
    n_edits = merge_edits(topup, log)
    logs = merge_logs(topup)
    abuse = merge_abuse(topup)
    for df, col in ((logs, "timestamp"), (abuse, "timestamp")):
        assert_no_after_cutoff(df, col)
    creation = creation_events(logs, issues)
    prof = build_profile(ask, topup, n_edits, logs, abuse, creation, issues)
    assert_unique(prof, "mentee_uid")
    write_table(OUT / "mentee_profile.parquet", prof)
    write_table(OUT / "mentee_logs.parquet", logs)
    write_table(OUT / "mentee_abuse.parquet", abuse)
    issues.write()
    counts = {"askers": len(prof), "source": prof.source.value_counts().to_dict(),
              "creation_action": prof.creation_action.value_counts(dropna=False).to_dict(),
              "creation_source": prof.creation_source.value_counts().to_dict(),
              "edits": int(sum(n_edits.values())), "logs": len(logs), "abuse": len(abuse),
              "askers_without_edits": int((prof.n_edits == 0).sum())}
    log.info("done: %s", counts)
    write_manifest("m3", inputs=[S6 / "coverage.json", RAW / "topup.jsonl", RAW / "creation.jsonl"],
                   outputs=[OUT / "mentee_profile.parquet", OUT / "mentee_logs.parquet", OUT / "mentee_abuse.parquet"],
                   params={}, counts=counts, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
