"""
index_s6.py: M3 part 1 (offline): s6 stores its data by user name. This re-indexes it by uid and keeps only askers

Module: M3 (PLAN.md M3; CODE_PLAN.md section 5, M3)
Rules: DATA_PROCESSING R6.1 to R6.3 (identity by uid), R7.1 (edit records); PLAN section 0.1 (data frozen at the cutoff date)

1. What it does
    1. build_name_to_uid(): from the s6 profile table (fetched 2026-04-23 to 24), builds "user name at fetch time -> uid".
       The 596 rows without a uid (the user name is no longer valid) are logged as issue `s6_profile_without_uid`.
    2. stream_contribs(): reads the 3.4 GB edits file line by line, keeps only M2 askers, converts them to uid,
       drops edits on or after the cutoff date, and writes parquet split into 16 shards by uid.
    3. stream_logs(), stream_abuselog(): do the same for the logs (111 MB) and the abuse filter records (5.1 GB).
       The details field of abuse filter records holds the wikitext of the whole page, so we keep only the summary fields.
    4. sanity_check_identity(): each person's registration time should be earlier than their earliest edit. If not,
       the "user name at fetch time" may point to a different person. We log issue `name_maps_to_other_user` and
       let the top-up fetch that person again by uid.

2. Why it is designed this way
    - s6 was fetched on April 23 to 24, so it already stops before the cutoff date (the latest log is 2026-04-23T19:23).
      We reuse it directly and do not fetch it again. It is stored by user name; the profile table comes from the
      same fetch, so "user name -> uid" is not affected by later renames.
    - The files are very large. We read only the user name at the start of each line to decide whether the line
      belongs to an asker, and only then parse the whole line (orjson). Only one line is held in memory at a time.
    - The s6 logs combine two queries: "performed by the person" and "targeting the person's user page". The same
      log entry can appear twice, so we remove duplicates by logid. Because of this, the account creation log
      (newusers) also covers accounts created by someone else (create2, byemail).

3. Inputs
    config.EXTERNAL["s6_profiles", "s6_contribs", "s6_logevents", "s6_abuselog"];
    data/interim/m2/questions.parquet (asker uids).

4. Outputs (data/interim/m3/s6/)
    name_to_uid.parquet          user name at fetch time -> uid (with profile fields)
    edits/shard=NN.parquet       edits: uid, revid, parentid, ns, title, timestamp, size, sizediff, tags, minor, new, comment
    logs.parquet                 logs: uid, logid, type, action, timestamp, ns, title, performer, params, comment
    abuse.parquet                abuse filter: uid, abuse_id, filter_id, filter, action, result, timestamp, ns, title
    coverage.json                for each asker, whether s6 has a profile, edits, logs and abuse filter records

5. Key parameters
    N_SHARDS = 16: shards are split by uid % 16.

6. Possible errors and how they are handled
    - A broken JSON line: stop with an error that shows the file name and line number (we do not skip it, so
      no one is silently lost);
    - Records on or after the cutoff date: dropped, and the count is written to the manifest (s6 was fetched the
      day before the cutoff, so we expect 0 or very few);
    - Registration time in the profile table is later than the earliest edit: log an issue and leave it to the top-up.

7. How to run
    python -m pipeline.m3_activity.index_s6          # about 10 to 20 minutes (reads 8.6 GB)

8. How to check
    checks/m3_check.ipynb; tests/test_m3.py.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from pipeline.common.io import iter_jsonl, write_json_atomic, write_table
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_sources, check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import before_cutoff, parse_iso
from pipeline.common.users import norm_user

OUT = config.INTERIM / "m3" / "s6"
N_SHARDS = 16
FLUSH_ROWS = 200_000
# Every line starts with {"username": "<name>", ...}. We read only this part to decide whether to parse the whole line
_HEAD_RE = re.compile(rb'^\{"username":\s*"((?:[^"\\]|\\.)*)"')

EDIT_SCHEMA = pa.schema([("uid", pa.int64()), ("revid", pa.int64()), ("parentid", pa.int64()), ("ns", pa.int32()),
                         ("title", pa.string()), ("timestamp", pa.string()), ("size", pa.int64()),
                         ("sizediff", pa.int64()), ("tags", pa.list_(pa.string())), ("minor", pa.bool_()),
                         ("new", pa.bool_()), ("comment", pa.string())])


def askers() -> set[int]:
    """The uids of all M2 askers."""
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["mentee_uid"])
    return {int(u) for u in q.mentee_uid.dropna()}


def build_name_to_uid(issues: IssueLog) -> pd.DataFrame:
    """s6 profile table: user name at fetch time -> uid, plus registration time, edit count and block status."""
    rows = []
    for p in iter_jsonl(config.EXTERNAL["s6_profiles"]):
        if p.get("userid") in (None, "", "None"):
            issues.add("s6_profile_without_uid", rule="R6.1", action="flagged", ids={"name": p.get("username")},
                       detail={"missing": p.get("missing")})
            continue
        rows.append({"name": p["username"], "uid": int(p["userid"]), "registration": p.get("registration"),
                     "editcount": p.get("editcount"), "blocked": bool(p.get("blocked")),
                     "groups": list(p.get("groups") or []), "role": list(p.get("role") or [])})
    df = pd.DataFrame(rows)
    dup = df.name.duplicated(keep=False)
    if dup.any():
        raise RuntimeError(f"s6 profiles map one name to several uids: {df[dup].head().to_dict('records')}")
    return df


def _lines(path: Path, wanted: dict[str, int]):
    """Reads a large file line by line and yields only lines whose user name is in wanted: (uid, parsed record)."""
    with open(path, "rb") as f:
        for lineno, line in enumerate(f, 1):
            m = _HEAD_RE.match(line)
            if not m:
                if line.strip():
                    raise ValueError(f"{path}:{lineno}: line does not start with a username field")
                continue
            name = json.loads(b'"' + m.group(1) + b'"')
            uid = wanted.get(name)
            if uid is None:
                continue
            try:
                yield uid, orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e


def _int(x):
    """Converts to int; returns None for empty values."""
    return None if x in (None, "") else int(x)


def stream_contribs(wanted: dict[str, int], log) -> dict:
    """Edits file -> edits/shard=NN.parquet (keeps only askers and edits before the cutoff date)."""
    (OUT / "edits").mkdir(parents=True, exist_ok=True)
    writers, buf = {}, defaultdict(list)
    counts = {"users": 0, "edits": 0, "dropped_after_cutoff": 0, "edit_count_mismatch": 0}
    seen_users: dict[int, int] = {}

    def flush(shard):
        if not buf[shard]:
            return
        tbl = pa.Table.from_pylist(buf[shard], schema=EDIT_SCHEMA)
        if shard not in writers:
            writers[shard] = pq.ParquetWriter(OUT / "edits" / f"shard={shard:02d}.parquet", EDIT_SCHEMA)
        writers[shard].write_table(tbl)
        buf[shard].clear()

    t0 = time.time()
    for uid, rec in _lines(config.EXTERNAL["s6_contribs"], wanted):
        counts["users"] += 1
        edits = rec.get("edits") or []
        seen_users[uid] = len(edits)
        if rec.get("edit_count") not in (None, "") and int(rec["edit_count"]) != len(edits):
            counts["edit_count_mismatch"] += 1
        shard = uid % N_SHARDS
        for e in edits:
            if not before_cutoff(e.get("timestamp")):
                counts["dropped_after_cutoff"] += 1
                continue
            buf[shard].append({"uid": uid, "revid": _int(e.get("revid")), "parentid": _int(e.get("parentid")),
                               "ns": _int(e.get("ns")), "title": e.get("title"), "timestamp": e.get("timestamp"),
                               "size": _int(e.get("size")), "sizediff": _int(e.get("sizediff")),
                               "tags": list(e.get("tags") or []), "minor": bool(e.get("minor")),
                               "new": bool(e.get("new")), "comment": e.get("comment")})
            counts["edits"] += 1
        if len(buf[shard]) >= FLUSH_ROWS:
            flush(shard)
        if counts["users"] % 5000 == 0:
            log.info("  contribs: %d users, %d edits (%.0f s)", counts["users"], counts["edits"], time.time() - t0)
    for s in list(buf):
        flush(s)
    for w in writers.values():
        w.close()
    counts["users_with_edits"] = sum(1 for v in seen_users.values() if v)
    return counts, seen_users


def stream_logs(wanted: dict[str, int]) -> tuple[pd.DataFrame, dict]:
    """Logs file -> one row per log entry, duplicates removed by (uid, logid), only entries before the cutoff date."""
    rows, seen, dropped = [], set(), 0
    users = set()
    for uid, rec in _lines(config.EXTERNAL["s6_logevents"], wanted):
        users.add(uid)
        for e in rec.get("events") or []:
            key = (uid, e.get("logid"))
            if key in seen:
                continue
            seen.add(key)
            if not before_cutoff(e.get("timestamp")):
                dropped += 1
                continue
            rows.append({"uid": uid, "logid": _int(e.get("logid")), "type": e.get("type"), "action": e.get("action"),
                         "timestamp": e.get("timestamp"), "ns": _int(e.get("ns")), "title": e.get("title"),
                         "performer": e.get("user"), "params": json.dumps(e.get("params") or {}, ensure_ascii=False),
                         "comment": e.get("comment")})
    return pd.DataFrame(rows), {"users": len(users), "events": len(rows), "dropped_after_cutoff": dropped}


def stream_abuselog(wanted: dict[str, int], log) -> tuple[pd.DataFrame, dict]:
    """Abuse filter file -> one row per record (without the whole-page wikitext in details), only records before the cutoff date."""
    rows, dropped, users = [], 0, set()
    t0 = time.time()
    for i, (uid, rec) in enumerate(_lines(config.EXTERNAL["s6_abuselog"], wanted), 1):
        users.add(uid)
        for e in rec.get("entries") or []:
            if not before_cutoff(e.get("timestamp")):
                dropped += 1
                continue
            rows.append({"uid": uid, "abuse_id": _int(e.get("id")), "filter_id": str(e.get("filter_id") or ""),
                         "filter": e.get("filter"), "action": e.get("action"), "result": e.get("result"),
                         "timestamp": e.get("timestamp"), "ns": _int(e.get("ns")), "title": e.get("title")})
        if i % 5000 == 0:
            log.info("  abuselog: %d users (%.0f s)", i, time.time() - t0)
    df = pd.DataFrame(rows).drop_duplicates(["uid", "abuse_id"]) if rows else pd.DataFrame(rows)
    return df, {"users": len(users), "entries": len(df), "dropped_after_cutoff": dropped}


def sanity_check_identity(prof: pd.DataFrame, issues: IssueLog) -> set[int]:
    """Registration time should be earlier than (or equal to) the earliest edit. uids that fail are logged as issues and returned.

    The earliest edit is taken from all shards. Accounts with an empty registration time (old accounts from
    before 2005) are not checked.
    """
    first = {}
    for f in sorted((OUT / "edits").glob("shard=*.parquet")):
        t = pq.read_table(f, columns=["uid", "timestamp"]).to_pandas()
        for uid, ts in t.groupby("uid").timestamp.min().items():
            first[int(uid)] = ts
    bad = set()
    for r in prof.itertuples():
        reg, fe = r.registration, first.get(r.uid)
        if reg and fe and parse_iso(fe) < parse_iso(reg):
            bad.add(r.uid)
            issues.add("name_maps_to_other_user", rule="R6.1", action="flagged", ids={"uid": r.uid, "name": r.name},
                       detail={"registration": reg, "first_edit": fe})
    return bad


def main(argv: list[str] | None = None) -> None:
    base_parser("M3 index: rebuild s6 by uid for askers only.").parse_args(argv)
    log = setup_logging("m3_index")
    t0 = time.time()
    check_upstream("m3_index", ["m2"])
    paths = check_sources(["s6_profiles", "s6_contribs", "s6_logevents", "s6_abuselog"])
    issues = IssueLog("m3_index")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in (OUT / "edits").glob("shard=*.parquet") if (OUT / "edits").exists() else []:
        old.unlink()

    ask = askers()
    n2u = build_name_to_uid(issues)
    prof = n2u[n2u.uid.isin(ask)].copy()
    wanted = dict(zip(prof.name, prof.uid))
    log.info("askers %d; in s6 profiles %d; s6 profiles total %d", len(ask), len(prof), len(n2u))
    write_table(OUT / "name_to_uid.parquet", n2u)

    c_edits, seen = stream_contribs(wanted, log)
    log.info("contribs: %s", c_edits)
    logs, c_logs = stream_logs(wanted)
    write_table(OUT / "logs.parquet", logs)
    log.info("logs: %s", c_logs)
    abuse, c_abuse = stream_abuselog(wanted, log)
    write_table(OUT / "abuse.parquet", abuse)
    log.info("abuselog: %s", c_abuse)

    bad = sanity_check_identity(prof, issues)
    log.info("identity check failed (registration after first edit): %d", len(bad))
    cov = {"askers": len(ask), "in_profiles": len(prof), "with_contribs_line": len(seen),
           "with_logs": c_logs["users"], "with_abuse_line": c_abuse["users"], "identity_failed": sorted(bad),
           "missing_uids": sorted(ask - set(prof.uid))}
    write_json_atomic(OUT / "coverage.json", cov)
    issues.write()
    write_manifest("m3_index", inputs=list(paths.values()), outputs=[OUT / "logs.parquet", OUT / "abuse.parquet"],
                   params={"n_shards": N_SHARDS},
                   counts={"askers": len(ask), "in_profiles": len(prof), "missing": len(cov["missing_uids"]),
                           "identity_failed": len(bad), "edits": c_edits, "logs": c_logs, "abuse": c_abuse},
                   started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
