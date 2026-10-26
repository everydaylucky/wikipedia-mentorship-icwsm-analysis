"""
collect_roster.py: the data-collection part of M1 (needs the internet; run it only once)

Module: M1 (PLAN.md M1; CODE_PLAN.md section 5, M1)
Rules: DATA_PROCESSING R1.1 to R1.3, R6.3

1. What it does
    1. Checks whether the roster page revisions we already have (PLAN section 1, item 1) reach the
       cutoff date. If they do, it skips them and does not download them again.
    2. Downloads every revision before the cutoff of the "manual mentor list" page from the
       wikitext period: Wikipedia:Growth Team features/Mentor list/Manual.
    3. For names that appear on that list but are not yet in the uid lookup table, it finds the uid
       through the API (first with list=users; if that fails, it follows the rename log to the
       user's current name, at most 8 steps).

2. Why it is designed this way
    In T419002 the Growth team explained that before September 2022 there were two unstructured
    lists. One listed mentors who were assigned automatically. The other listed "manual" mentors,
    who could only be claimed and were not part of automatic assignment. Where these list pages
    live is set by two settings in MediaWiki:GrowthExperimentsConfig.json:
        GEHomepageMentorsList                 = Wikipedia:Growth Team features/Mentor list
        GEHomepageManualAssignmentMentorsList = Wikipedia:Growth Team features/Mentor list/Manual
                                                (from 2021-09-20, config page revision 1045445204)
    The paper version and eval2 only parsed the first list, so manual mentors from the wikitext
    period were never recorded. With them added, M4 can tell whether a newcomer's mentor was in the
    automatic pool at registration, and M2 can recognize replies from manual mentors.

3. Inputs
    roster_revisions_wikitext, roster_revisions_json, and roster_uid_map from config.EXTERNAL.

4. Outputs (data/raw/m1_roster/)
    manual_list_revisions.jsonl: all revisions of the manual list page before the cutoff (with content);
    uid_map_extra.json: newly resolved "name -> uid" pairs, plus the lookup path for each name (as evidence);
    config_revisions.jsonl: a record of changes to the two list settings in GrowthExperimentsConfig.json.

6. Possible errors and how they are handled
    - The existing revisions do not reach the cutoff: the script stops with an error and says the
      data must be re-fetched (currently they reach 2026-06-17).
    - A name cannot be resolved to a uid: it is written to "unresolved" in uid_map_extra.json and
      logged as the issue `unresolved_mentor_name`. build_roster.py keeps these names under a NAME:
      key, so they are not lost.

7. How to run
    python -m pipeline.m1_roster.collect_roster --online     # about 1 minute; results are cached in data/cache/api/
    On later runs everything comes from the cache, so --online can be left out.

8. How to check
    The "manual list" section of checks/m1_check.ipynb.
"""
from __future__ import annotations

import json
import sys
import time

import config
from pipeline.common.api import MediaWikiClient
from pipeline.common.io import iter_jsonl, write_json_atomic, write_jsonl_atomic
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_sources, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import before_cutoff
from pipeline.common.users import norm_user

RAW_DIR = config.RAW / "m1_roster"
MANUAL_PAGE = "Wikipedia:Growth Team features/Mentor list/Manual"
CONFIG_PAGE = "MediaWiki:GrowthExperimentsConfig.json"
MAX_RENAME_HOPS = 8


def check_revision_coverage(paths: dict) -> dict:
    """Confirm that the roster revisions we have reach the cutoff (the last revision is after the cutoff).

    Returns:
        {"wikitext_last": ..., "json_last": ..., "json_pre_cutoff": number of revisions before the cutoff}
    Raises:
        RuntimeError: the last JSON roster revision is before the cutoff. This means some revisions
        before the cutoff may not have been downloaded yet.
    """
    last_w = max(r["timestamp"] for r in iter_jsonl(paths["roster_revisions_wikitext"]))
    js = [r["timestamp"] for r in iter_jsonl(paths["roster_revisions_json"])]
    out = {"wikitext_last": last_w, "json_last": max(js),
           "json_pre_cutoff": sum(before_cutoff(t) for t in js)}
    if before_cutoff(out["json_last"]):
        raise RuntimeError(f"JSON roster revisions end at {out['json_last']}, before the cutoff; re-fetch needed")
    return out


def fetch_revisions(client: MediaWikiClient, title: str, *, until: str | None = None) -> list[dict]:
    """Get all revisions of a page (with content), from the earliest up to `until` (not included).

    Args:
        title: page title.
        until: cutoff time (ISO). Only revisions earlier than this are kept.
    """
    revs = []
    params = {"action": "query", "prop": "revisions", "titles": title, "rvslots": "main",
              "rvprop": "ids|timestamp|user|userid|comment|content", "rvlimit": 50, "rvdir": "newer"}
    for page in client.query_all(params, ("query", "pages")):
        for r in page.get("revisions", []):
            if until is None or r["timestamp"] < until:
                revs.append(r)
    revs.sort(key=lambda r: r["timestamp"])
    return revs


def config_list_history(client: MediaWikiClient) -> list[dict]:
    """How the two list settings in GrowthExperimentsConfig.json changed over time (only the points where they change)."""
    out, last = [], None
    for r in fetch_revisions(client, CONFIG_PAGE, until=config.ROSTER_SOURCE_SWITCH.isoformat().replace("+00:00", "Z")):
        try:
            cfg = json.loads(r["slots"]["main"]["content"])
        except (KeyError, json.JSONDecodeError):
            continue
        key = (cfg.get("GEHomepageMentorsList"), cfg.get("GEHomepageManualAssignmentMentorsList"))
        if key != last:
            out.append({"timestamp": r["timestamp"], "revid": r["revid"],
                        "auto_list": key[0], "manual_list": key[1]})
            last = key
    return out


def resolve_name(client: MediaWikiClient, name: str) -> tuple[int | None, list[str]]:
    """Find the uid for a user name (R6.3). First try list=users; if that fails, follow the rename log.

    Returns:
        (uid or None, lookup path). The path lists every name visited along the way and is saved as evidence.
    """
    path, current = [name], name
    for _ in range(MAX_RENAME_HOPS + 1):
        users = client.get({"action": "query", "list": "users", "ususers": current})["query"]["users"]
        if users and users[0].get("userid"):
            return int(users[0]["userid"]), path
        log = client.get({"action": "query", "list": "logevents", "letype": "renameuser",
                          "letitle": f"User:{current}", "lelimit": 5})["query"]["logevents"]
        new = None
        for e in log:
            p = e.get("params", {})
            new = p.get("newuser") or p.get("5::newuser")
            if new:
                break
        if not new or norm_user(new) in path:
            return None, path
        current = norm_user(new)
        path.append(current)
    return None, path


def main(argv: list[str] | None = None) -> None:
    args = base_parser("M1 collection: verify roster revisions; fetch the manual mentor list; resolve uids.").parse_args(argv)
    log = setup_logging("m1_collect")
    t0 = time.time()
    issues = IssueLog("m1_collect")
    paths = check_sources(["roster_revisions_wikitext", "roster_revisions_json", "roster_uid_map"])
    client = MediaWikiClient(online=args.online)

    cov = check_revision_coverage(paths)
    log.info("existing roster revisions cover the cutoff: %s (skip re-fetch)", cov)

    cfg = config_list_history(client)
    for c in cfg:
        log.info("config %s rev %s: auto=%r manual=%r", c["timestamp"], c["revid"], c["auto_list"], c["manual_list"])

    switch = config.ROSTER_SOURCE_SWITCH.isoformat().replace("+00:00", "Z")
    manual = fetch_revisions(client, MANUAL_PAGE, until=switch)
    log.info("manual list page: %d revisions before the JSON switch (%s)", len(manual), switch)

    # Names that appear on the manual list (same parsing as build_roster, which explains it in detail)
    from pipeline.m1_roster.build_roster import parse_wikitext_mentors
    names = sorted({n for r in manual for n in parse_wikitext_mentors(r["slots"]["main"].get("content", ""))})
    known = json.loads(paths["roster_uid_map"].read_text())
    need = [n for n in names if known.get(n) is None]
    log.info("manual-list names: %d, already in uid map: %d, to resolve: %d", len(names), len(names) - len(need), len(need))
    extra = {}
    for n in need:
        uid, path = resolve_name(client, n)
        extra[n] = {"uid": uid, "path": path}
        if uid is None:
            issues.add("unresolved_mentor_name", rule="R6.3", action="flagged", ids={"name": n},
                       detail={"path": path})
        log.info("  %-30s -> %s  via %s", n, uid, " → ".join(path))

    # Some entries in the JSON roster have no user name (only a uid). Look up the name by uid;
    # otherwise M2 cannot recognize that mentor's signature.
    uids_with_name, all_uids = set(), set()
    for rev in iter_jsonl(paths["roster_revisions_json"]):
        if not before_cutoff(rev["timestamp"]):
            continue
        from pipeline.m1_roster.build_roster import get_content, parse_json_mentors
        for uid, e in (parse_json_mentors(get_content(rev)) or {}).items():
            all_uids.add(int(uid))
            if norm_user(e.get("username")):
                uids_with_name.add(int(uid))
    nameless = sorted(all_uids - uids_with_name - {v["uid"] for v in extra.values() if v.get("uid")}
                      - {int(v) for v in known.values() if v is not None})
    uid_names = {}
    for i in range(0, len(nameless), 50):
        batch = nameless[i:i + 50]
        for u in client.get({"action": "query", "list": "users", "ususerids": "|".join(map(str, batch))})["query"]["users"]:
            if u.get("userid"):
                uid_names[str(u["userid"])] = u.get("name")
    log.info("uids without a username in the JSON roster: %d, resolved by uid: %d %s", len(nameless), len(uid_names), uid_names)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(RAW_DIR / "uid_names.json", uid_names)
    write_jsonl_atomic(RAW_DIR / "manual_list_revisions.jsonl", manual)
    write_jsonl_atomic(RAW_DIR / "config_revisions.jsonl", cfg)
    write_json_atomic(RAW_DIR / "uid_map_extra.json", extra)
    issues.write()
    log.info("API stats: %s", client.stats)
    write_manifest("m1_collect", inputs=list(paths.values()),
                   outputs=[RAW_DIR / "manual_list_revisions.jsonl", RAW_DIR / "uid_map_extra.json",
                            RAW_DIR / "config_revisions.jsonl"],
                   params={"online": args.online}, counts={"manual_revisions": len(manual),
                                                           "manual_names": len(names), "resolved_now": len(need)},
                   started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
