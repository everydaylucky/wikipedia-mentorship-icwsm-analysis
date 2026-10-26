"""
collect_topup.py: M3 part 2 (online, run only once): fetches askers that s6 does not have, and fills in account creation logs

Module: M3 (PLAN.md M3; CODE_PLAN.md section 5, M3)
Rules: DATA_PROCESSING R2.2 (how the account was created), R6.1 to R6.3 (identity by uid), R7.1; Issues: I-48

1. What it does
    0. find_stale(): not all s6 edit records were fetched in April. In batches, we check whether each s6 asker has
       any edits "after their last s6 edit and before the cutoff date". People who do, and people whose s6 record
       stops at exactly 500 edits, count as incomplete, and the whole person is fetched again.
    1. missing_uids(): askers in M2 that s6 does not have (about 1,520 people), plus people who failed the identity
       check in index_s6, plus the incomplete people found in step 0.
    2. For each person (by uid):
       - profile: list=users&ususerids=... (current name, registration time, edit count, block, user groups), 50 people per request;
       - edits: list=usercontribs&ucuserids=uid&ucstart=cutoff&ucdir=older, all pages. Queried by uid, so renames do not matter;
       - logs: list=logevents&leuser=current name&lestart=cutoff&ledir=older, and also letitle=User:current name;
       - abuse filter: list=abuselog&afluser=current name&aflstart=cutoff&afldir=older.
    3. creation_log(): the account creation log. For top-up people, and for askers whose s6 logs have no creation
       record of their own, we search by registration time: list=logevents&letype=newusers, with a window from
       2 seconds before to 10 seconds after registration, and take the entry whose params.userid equals this uid.

2. Why it is designed this way
    - Edits are queried by uid (ucuserids), so renames do not matter. The logs and abuse filter APIs accept only
      user names, so we first use the uid to find the current name.
    - Why the creation log is searched by registration time: the target page of a newusers log entry is the name at
      account creation, and it is not updated after a rename. For accounts created by someone else (create2,
      byemail), the performer is the person who created it. So searching by current name, as target page or as
      performer, can miss the entry. The registration time is the time of the creation log (written in the same
      step), and the log parameters carry the new account's userid. So "the newusers entry in the second of
      registration with the same userid" is not affected by renames or by creation by others, and it is unique.
    - The top-up happened after the cutoff date (2026-10). Edits deleted after the cutoff and tags added after the
      cutoff cannot be recovered. These people get late_fetch=True in mentee_profile, and M8 runs a sensitivity
      analysis that excludes them.
    - Resuming after a stop: one line is written to topup.jsonl as soon as each person is done; all requests are
      cached on disk.

3. Inputs
    data/interim/m3/s6/coverage.json, logs.parquet (index_s6); data/interim/m2/questions.parquet;
    config.RAW_FILES["registration"] (registration times fetched by uid on 2026-10-07, used to locate the creation log).

4. Outputs (data/raw/m3/)
    stale.json       people whose s6 record is incomplete: uid -> {reason, n_new}
    stale_edits.jsonl  edits newly found by find_stale (all fields): uid, fetched_at, edits
    topup.jsonl      one line per person: uid, fetched_at, user (profile), contribs, logs, abuse
    topup_stale.jsonl  people with incomplete records: same as above; contribs is only the part added by find_stale, partial=True
    creation.jsonl   one line per person: uid, fetched_at, registration, event (the creation log, null if not found)

6. Possible errors and how they are handled
    - The user name is hidden or the account does not exist (list=users returns missing/hidden): log issue
      `user_hidden_or_missing`. Edits are still fetched by uid (usercontribs returns nothing for hidden users);
    - No creation log with the same userid in the second of registration (old accounts from before 2005 have no
      creation log): event=null, log issue `creation_log_not_found`. M4 treats these as "creation method unknown".

7. How to run
    python -m pipeline.m3_activity.collect_topup --online --phase stale   # only step 0
    python -m pipeline.m3_activity.collect_topup --online                 # everything (finished work is skipped)

8. How to check
    The "top-up" section of checks/m3_check.ipynb.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd

import config
from pipeline.common.api import MediaWikiClient
from pipeline.common.io import iter_jsonl
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import before_cutoff

RAW = config.RAW / "m3"
S6 = config.INTERIM / "m3" / "s6"
CUTOFF = config.DATA_CUTOFF_ISO
# The creation log time is 0 to 1 second later than the registration time (in the same request, the user table is
# written first and the log after; for example AlphaBetaGamma registered at 04:19:01, log at 04:19:02). We search
# from 2 seconds before to 10 seconds after registration, then match the userid exactly.
CREATION_WINDOW_BEFORE = 2
CREATION_WINDOW_AFTER = 10


def missing_uids() -> list[int]:
    """uids to fetch in the top-up: askers not in s6 + people who failed the identity check + people whose s6 record is incomplete (stale.json, see find_stale)."""
    cov = json.loads((S6 / "coverage.json").read_text())
    st = json.loads((RAW / "stale.json").read_text()) if (RAW / "stale.json").exists() else {}
    capped = {int(u) for u, v in st.items() if v["reason"] == "capped_500"}
    return sorted(set(cov["missing_uids"]) | set(cov["identity_failed"]) | capped)


def stale_uids() -> list[int]:
    """People whose s6 edits stop too early but whose new edits find_stale already added: only their logs and abuse filter records still need fetching."""
    st = json.loads((RAW / "stale.json").read_text()) if (RAW / "stale.json").exists() else {}
    return sorted(int(u) for u, v in st.items() if v["reason"] == "newer_edits")


def s6_last_edits() -> pd.DataFrame:
    """For each asker in s6: the number of edits and the time of the last edit."""
    import pyarrow.dataset as ds
    e = ds.dataset(S6 / "edits").to_table(columns=["uid", "timestamp"]).to_pandas()
    return e.groupby("uid").timestamp.agg(n="size", last="max").reset_index()


def find_stale(client, log) -> dict[int, dict]:
    """Finds askers whose s6 edit record is incomplete (not all s6 edit records were fetched in April).

    How we found this: for 104 askers, their own question revision was not among their s6 edits. Checking against
    the API showed that their s6 edits stop at an earlier time (for example IQEES stops at 2026-01-19, while the
    April profile table already shows 61 edits). So part of the s6 edit records was fetched earlier and never updated.

    Method: for each s6 asker, check whether there are edits "after the last s6 edit and before the cutoff date":
        list=usercontribs&ucuserids=up to 50 people&ucstart=cutoff&ucend=the earliest "last edit time" in this batch.
    We sort by "last edit time" before splitting into batches, so people in one batch have similar times and few
    edits are returned twice.
    Also, people with exactly 500 s6 edits but a higher edit count in the profile table (only the first page was
    fetched) count as incomplete.

    Newly found edits (all fields) are also written to stale_edits.jsonl. Later, these people only need their logs
    and abuse filter records fetched.

    Returns:
        uid -> {"reason": "newer_edits" or "capped_500", "n_new": number of newly found edits}
    """
    new_edits: dict[int, list[dict]] = {}
    last = s6_last_edits().sort_values("last")
    prof = pd.read_parquet(S6 / "name_to_uid.parquet").set_index("uid")
    out: dict[int, dict] = {}
    known_last = dict(zip(last.uid, last["last"]))
    for r in last.itertuples():
        ec = pd.to_numeric(prof.editcount.get(r.uid), errors="coerce")
        if r.n == 500 and pd.notna(ec) and ec > 500:
            out[int(r.uid)] = {"reason": "capped_500", "n_new": None}
    uids = list(last.uid)
    for i in range(0, len(uids), 50):
        batch = [int(u) for u in uids[i:i + 50]]
        end = min(known_last[u] for u in batch)
        for e in client.query_all({"action": "query", "list": "usercontribs", "ucuserids": "|".join(map(str, batch)),
                                   "ucstart": CUTOFF, "ucend": end, "ucdir": "older", "uclimit": "max",
                                   "ucprop": "ids|title|timestamp|comment|size|sizediff|flags|tags"},
                                  ("query", "usercontribs")):
            u = int(e["userid"])
            if e["timestamp"] > known_last[u] and before_cutoff(e["timestamp"]):
                d = out.setdefault(u, {"reason": "newer_edits", "n_new": 0})
                if d["reason"] == "newer_edits":
                    d["n_new"] += 1
                new_edits.setdefault(u, []).append(e)
        if (i // 50) % 50 == 0:
            log.info("  stale check %d/%d users (stale so far %d, api %s)", i + len(batch), len(uids), len(out), client.stats)
    now = datetime.now(timezone.utc).isoformat()
    with open(RAW / "stale_edits.jsonl", "w", encoding="utf-8") as f:
        for u, es in new_edits.items():
            f.write(json.dumps({"uid": u, "fetched_at": now, "edits": es}, ensure_ascii=False) + "\n")
    return out


def fetch_users(client, uids: list[int]) -> dict[int, dict]:
    """Fetches profiles by uid (50 people per request)."""
    out = {}
    for i in range(0, len(uids), 50):
        batch = uids[i:i + 50]
        d = client.get({"action": "query", "list": "users", "ususerids": "|".join(map(str, batch)),
                        "usprop": "registration|editcount|blockinfo|groups"})
        for u in d["query"]["users"]:
            if u.get("userid"):
                out[int(u["userid"])] = u
    for u in uids:
        out.setdefault(u, {"userid": u, "missing": True})
    return out


def fetch_contribs(client, uid: int) -> list[dict]:
    """All edits before the cutoff date (by uid)."""
    out = []
    for e in client.query_all({"action": "query", "list": "usercontribs", "ucuserids": uid, "ucstart": CUTOFF,
                               "ucdir": "older", "uclimit": "max",
                               "ucprop": "ids|title|timestamp|comment|size|sizediff|flags|tags"},
                              ("query", "usercontribs")):
        if before_cutoff(e.get("timestamp")):
            out.append(e)
    return out


def fetch_logs(client, name: str) -> list[dict]:
    """Logs performed by the person and logs targeting the person's user page (the same two queries as s6), before the cutoff date."""
    out, seen = [], set()
    base = {"action": "query", "list": "logevents", "lestart": CUTOFF, "ledir": "older", "lelimit": "max",
            "leprop": "ids|title|type|user|userid|timestamp|comment|details"}
    for extra in ({"leuser": name}, {"letitle": f"User:{name}"}):
        for e in client.query_all({**base, **extra}, ("query", "logevents")):
            if e.get("logid") not in seen and before_cutoff(e.get("timestamp")):
                seen.add(e.get("logid"))
                out.append(e)
    return out


def fetch_abuselog(client, name: str) -> list[dict]:
    """Abuse filter records (by name), before the cutoff date."""
    out = []
    for e in client.query_all({"action": "query", "list": "abuselog", "afluser": name, "aflstart": CUTOFF,
                               "afldir": "older", "afllimit": "max",
                               "aflprop": "ids|filter|user|title|action|result|timestamp|hidden"},
                              ("query", "abuselog")):
        if before_cutoff(e.get("timestamp")):
            out.append(e)
    return out


def creation_log(client, uid: int, registration: str | None) -> dict | None:
    """Account creation log: the newusers entry near the registration time whose params.userid equals uid (see section 2)."""
    if not registration:
        return None
    from pipeline.common.timeutil import parse_iso, to_iso
    from datetime import timedelta
    t = parse_iso(registration)
    d = client.get({"action": "query", "list": "logevents", "letype": "newusers",
                    "lestart": to_iso(t + timedelta(seconds=CREATION_WINDOW_AFTER)),
                    "leend": to_iso(t - timedelta(seconds=CREATION_WINDOW_BEFORE)), "ledir": "older", "lelimit": "max",
                    "leprop": "ids|title|type|user|userid|timestamp|comment|details"})
    for e in d.get("query", {}).get("logevents", []):
        if int((e.get("params") or {}).get("userid", -1)) == uid:
            return e
    return None


def creation_needed(topup: list[int]) -> list[int]:
    """uids whose creation log must be looked up: top-up people + askers with no newusers record of their own in the s6 logs."""
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["mentee_uid"])
    ask = {int(u) for u in q.mentee_uid.dropna()}
    logs = pd.read_parquet(S6 / "logs.parquet")
    nu = logs[logs.type == "newusers"]
    have = {int(r.uid) for r in nu.itertuples()
            if json.loads(r.params or "{}").get("userid") in (r.uid, str(r.uid))}
    return sorted((ask - have) | set(topup))


def done_uids(path) -> set[int]:
    """For resuming after a stop: the uids already written to path."""
    return {r["uid"] for r in iter_jsonl(path)} if path.exists() else set()


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("M3 top-up: askers missing from s6, stale s6 histories, account creation logs.")
    ap.add_argument("--phase", choices=["all", "stale", "stale_topup"], default="all",
                    help="stale: only the stale-history check; stale_topup: only logs and abuse log for stale users")
    args = ap.parse_args(argv)
    log = setup_logging("m3_collect")
    t0 = time.time()
    check_upstream("m3_collect", ["m3_index"])
    issues = IssueLog("m3_collect")
    client = MediaWikiClient(online=args.online)
    RAW.mkdir(parents=True, exist_ok=True)
    reg = json.loads(config.RAW_FILES["registration"].read_text())

    # 0. Askers whose s6 edit record is incomplete
    if args.phase == "stale" or not (RAW / "stale.json").exists():
        from pipeline.common.io import write_json_atomic
        stale = find_stale(client, log)
        write_json_atomic(RAW / "stale.json", {str(k): v for k, v in stale.items()})
        log.info("stale s6 histories: %d (%s)", len(stale),
                 pd.Series([v["reason"] for v in stale.values()]).value_counts().to_dict() if stale else {})
    if args.phase == "stale":
        return

    # 1. Top-up (skipped in the stale_topup phase)
    todo = [] if args.phase == "stale_topup" else missing_uids()
    if args.limit:
        todo = todo[:args.limit]
    users = fetch_users(client, todo) if todo else {}
    done = done_uids(RAW / "topup.jsonl")
    log.info("top-up: %d uids (%d already done)", len(todo), len(done & set(todo)))
    with open(RAW / "topup.jsonl", "a", encoding="utf-8") as f:
        for i, uid in enumerate(todo, 1):
            if uid in done:
                continue
            u = users[uid]
            name = u.get("name")
            if u.get("missing") or u.get("hidden") or not name:
                issues.add("user_hidden_or_missing", rule="R6.1", action="flagged", ids={"uid": uid}, detail=u)
            rec = {"uid": uid, "fetched_at": datetime.now(timezone.utc).isoformat(), "user": u,
                   "contribs": fetch_contribs(client, uid),
                   "logs": fetch_logs(client, name) if name else [],
                   "abuse": fetch_abuselog(client, name) if name else []}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                log.info("  top-up %d/%d (%.0f s, api %s)", i, len(todo), time.time() - t0, client.stats)

    # 1b. People whose s6 edits stop too early: find_stale already got the new edits; here we fetch logs and abuse filter records (all of them; duplicates with s6 are removed by (uid, logid))
    st_new = {r["uid"]: r for r in iter_jsonl(RAW / "stale_edits.jsonl")} if (RAW / "stale_edits.jsonl").exists() else {}
    st_path = RAW / "topup_stale.jsonl"
    already = done_uids(st_path) | done_uids(RAW / "topup.jsonl")
    st_todo = [u for u in stale_uids() if u not in already]
    st_users = fetch_users(client, st_todo) if st_todo else {}
    log.info("stale s6 users needing logs and abuse log: %d", len(st_todo))
    with open(st_path, "a", encoding="utf-8") as f:
        for i, uid in enumerate(st_todo, 1):
            u = st_users[uid]
            name = u.get("name")
            rec = {"uid": uid, "fetched_at": datetime.now(timezone.utc).isoformat(), "user": u, "partial": True,
                   "contribs": (st_new.get(uid) or {}).get("edits", []),
                   "logs": fetch_logs(client, name) if name else [],
                   "abuse": fetch_abuselog(client, name) if name else []}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                log.info("  stale top-up %d/%d (%.0f s)", i, len(st_todo), time.time() - t0)
    if args.phase == "stale_topup":
        return

    # 2. Account creation logs
    need = creation_needed(todo)
    done_c = done_uids(RAW / "creation.jsonl")
    log.info("creation logs to look up: %d (%d already done)", len(need), len(done_c & set(need)))
    n_found = 0
    with open(RAW / "creation.jsonl", "a", encoding="utf-8") as f:
        for i, uid in enumerate(need, 1):
            if uid in done_c:
                continue
            r = (users.get(uid) or {}).get("registration") or (reg.get(str(uid)) or {}).get("registration")
            ev = creation_log(client, uid, r)
            n_found += ev is not None
            if ev is None:
                issues.add("creation_log_not_found", rule="R2.2", action="flagged", ids={"uid": uid},
                           detail={"registration": r})
            f.write(json.dumps({"uid": uid, "fetched_at": datetime.now(timezone.utc).isoformat(),
                                "registration": r, "event": ev}, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                log.info("  creation %d/%d (%.0f s)", i, len(need), time.time() - t0)
    issues.write()
    log.info("done | api %s", client.stats)
    write_manifest("m3_collect", inputs=[S6 / "coverage.json", config.RAW_FILES["registration"]],
                   outputs=[RAW / "topup.jsonl", RAW / "creation.jsonl"], params={"online": args.online},
                   counts={"topup": len(todo), "creation_lookups": len(need), "creation_found_now": n_found,
                           "api": client.stats}, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
