"""
build_roster.py: M1 mentor roster (offline)

Module: M1 (PLAN.md M1; CODE_PLAN.md section 5, M1)
Rules: DATA_PROCESSING R1.1 (roster formats in the three periods), R1.2 (cutoff date), R1.3 and R6.1 to R6.3 (uid is the primary key)

1. What it does
    Replays every revision of the mentor roster pages before the cutoff date and rebuilds, for each
    mentor at each moment, "is this mentor in the automatic assignment pool?". It also collects every
    name each mentor has used. It does not use the internet (the online part is in collect_roster.py).

2. Why it is designed this way
    The roster went through three periods (DATA_PROCESSING section 1):
        1. Wikitext period (2021-05-04 to 2022-10-26): two pages.
           People listed on Wikipedia:Growth Team features/Mentor list          are in the automatic pool;
           people listed on Wikipedia:Growth Team features/Mentor list/Manual   can only be claimed by hand (in effect from 2021-09-20).
           These pages have only user names, with no uid and no weight.
        2. JSON period with an automaticallyAssigned field (2022-10-26 to 2023-11-07);
        3. JSON period with only weight; weight = 0 means not in the automatic pool (T347157).
    The boundary between periods 2 and 3 is not a fixed date. Each entry is checked on its own:
    if it has automaticallyAssigned, use that; otherwise look at weight.

    Compared with s1_collect_mentors.py in eval2:
        - Adds the manual list from the wikitext period (eval2 did not parse it);
        - The cutoff is always "strictly before" DATA_CUTOFF (eval2 used "not later than");
        - HTML comments are removed before parsing, and the subpage part is removed from names (common.users.strip_subpage);
        - Output is now parquet, with a new table of status periods and a new table of name history.
    The rest of the logic (snapshots, event differences, continuity across periods) is the same as
    in eval2, and the results are compared with eval2 mentor by mentor.

3. Inputs
    config.EXTERNAL: roster_revisions_wikitext, roster_revisions_json, roster_uid_map, eval2_mentor_list;
    data/raw/m1_roster/: manual_list_revisions.jsonl, config_revisions.jsonl, uid_map_extra.json
    (made by collect_roster.py).

4. Outputs (data/interim/m1/)
    mentor_list.parquet     one row per mentor: mentor_uid, username, uid_resolved, first_joined, last_left,
                            is_current, currently_in_auto_pool, ever_in_auto_pool, ever_manual_only,
                            join_count, leave_count, sources
    mentor_periods.parquet  one row per status period of each mentor: mentor_uid, start, end, in_auto_pool,
                            weight_known, source (an empty end means the status still holds at the cutoff)
    mentor_events.parquet   status change events: timestamp, mentor_uid, username, event, in_auto_pool,
                            source, editor, comment, revid
    mentor_names.parquet    names each mentor has used: mentor_uid, name, first_seen, last_seen, source
    monthly_pool.csv (results/m1/) number of mentors on the roster, and in the automatic pool, at the end of each month

5. Processing steps
    load_inputs() → build_uid_map() → build_snapshots() → diff_snapshots_to_events()
    → build_mentor_list() / build_periods() / build_names() → compare_with_eval2() → write

6. Possible errors and how they are handled
    - A name cannot be resolved to a uid: it is kept with the key "NAME:<name>" and not dropped; logged as issue `unresolved_mentor_name`;
    - A JSON revision cannot be parsed: that revision is skipped and logged as issue `json_parse_error` (revisions before and after are not affected);
    - A status that lasts less than 1 hour (back-and-forth edits while someone edits the roster): kept, and logged as issue `transient_pool_state` for review;
    - Differences from the eval2 roster: logged as issue `differs_from_eval2`, with the reason for each one (mostly from the manual list).

7. How to run
    python -m pipeline.m1_roster.collect_roster --online   # needed the first time (about 1 minute)
    python -m pipeline.m1_roster.build_roster              # offline, a few tens of seconds
    python run_all.py --from m1 --to m1

8. How to check
    tests/test_m1.py; checks/m1_check.ipynb.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict

import pandas as pd

import config
from pipeline.common.checks import assert_no_after_cutoff, assert_unique
from pipeline.common.io import iter_jsonl, read_jsonl, write_table
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_sources, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import before_cutoff, parse_iso
from pipeline.common.users import norm_user, strip_subpage

OUT = config.interim("m1")
RAW_DIR = config.RAW / "m1_roster"
SWITCH = "2022-10-26T13:19:10Z"            # first revision of the JSON roster; before it we read wikitext, after it we read JSON
TRANSIENT_HOURS = 1.0

_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_USER_RE = re.compile(r"\[\[\s*User\s*:\s*([^\]|#]+)", re.I)


# ════════════════════════════════════════════════════════════════════════════
# Parsing
# ════════════════════════════════════════════════════════════════════════════


def get_content(rev: dict) -> str:
    """Get the content of a revision. Works with both API formats: slots.main["*"] (formatversion=1) and slots.main.content (=2)."""
    main = rev.get("slots", {}).get("main", {})
    return main.get("content", main.get("*", rev.get("*", ""))) or ""


def parse_wikitext_mentors(text: str) -> list[str]:
    """Wikitext roster: each line looks like "* [[User:<name>]]|<intro>" (R1.1).

    Steps:
        1. Remove HTML comments (entries that are commented out are not on the roster);
        2. Only look at lines that start with * (the example lines in the page instructions do not start with *);
        3. Take the name from the first [[User:...]], normalize it, and remove any subpage part.

    Returns:
        A list of names (in page order, without duplicates).
    """
    out: list[str] = []
    for line in _COMMENT_RE.sub("", text or "").split("\n"):
        line = line.strip()
        if not line.startswith("*"):
            continue
        m = _USER_RE.search(line)
        if m:
            n = strip_subpage(m.group(1))
            if n and n not in out:
                out.append(n)
    return out


def parse_json_mentors(text: str) -> dict[str, dict] | None:
    """JSON roster: {"Mentors": {uid: {username, message, weight[, automaticallyAssigned]}}}.

    Returns:
        {uid string: entry}; None if the content cannot be parsed (the caller logs an issue).
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    md = data.get("Mentors", data) if isinstance(data, dict) else {}
    return {k: v for k, v in md.items() if isinstance(v, dict)} if isinstance(md, dict) else {}


def json_entry_in_auto_pool(entry: dict) -> tuple[bool | None, bool]:
    """Is one JSON entry in the automatic pool (R1.1)? Use automaticallyAssigned if present; otherwise check weight != 0.

    Returns:
        (in_auto_pool, weight_known). If neither field is present: (None, False).
    """
    if "automaticallyAssigned" in entry:
        return bool(entry["automaticallyAssigned"]), True
    if entry.get("weight") is not None:
        return entry["weight"] != 0, True
    return None, False


# ════════════════════════════════════════════════════════════════════════════
# Inputs
# ════════════════════════════════════════════════════════════════════════════


def load_inputs() -> dict:
    """Read the roster revisions, manual list revisions, config changes, and uid lookup table. Keep only revisions before the cutoff (R1.2)."""
    src = check_sources(["roster_revisions_wikitext", "roster_revisions_json", "roster_uid_map",
                         "eval2_mentor_list"])
    manual_path = RAW_DIR / "manual_list_revisions.jsonl"
    if not manual_path.exists():
        raise FileNotFoundError(f"{manual_path} missing: run `python -m pipeline.m1_roster.collect_roster --online` first")
    wiki = [r for r in iter_jsonl(src["roster_revisions_wikitext"]) if before_cutoff(r["timestamp"])]
    js = [r for r in iter_jsonl(src["roster_revisions_json"]) if before_cutoff(r["timestamp"])]
    manual = read_jsonl(manual_path)
    cfg = read_jsonl(RAW_DIR / "config_revisions.jsonl")
    uid_map = {norm_user(k): v for k, v in json.loads(src["roster_uid_map"].read_text()).items()}
    extra = json.loads((RAW_DIR / "uid_map_extra.json").read_text())
    return {"src": src, "wiki": wiki, "json": js, "manual": manual, "cfg": cfg,
            "uid_map": uid_map, "uid_extra": extra}


def build_uid_map(inp: dict) -> dict[str, int]:
    """The "name → uid" lookup for the wikitext period (R6.3).

    Sources (highest priority first):
        1. The JSON roster itself: in the JSON period every entry has both a uid and a name, so this is the most reliable match;
        2. The existing uid lookup table (eval2 resolved it with the API and the rename log; all 118 names have a uid);
        3. Manual list names newly resolved by collect_roster.py.
    """
    m: dict[str, int] = {}
    for k, v in inp["uid_extra"].items():
        if v.get("uid") is not None:
            m[norm_user(k)] = int(v["uid"])
    for k, v in inp["uid_map"].items():
        if v is not None:
            m[k] = int(v)
    for rev in inp["json"]:
        for uid, e in (parse_json_mentors(get_content(rev)) or {}).items():
            n = norm_user(e.get("username"))
            if n:
                try:
                    m[n] = int(uid)
                except ValueError:
                    pass
    return m


# ════════════════════════════════════════════════════════════════════════════
# Snapshots and events
# ════════════════════════════════════════════════════════════════════════════


def manual_list_window(cfg: list[dict]) -> list[tuple[str, str | None]]:
    """When the manual list was in effect: the time spans when the setting GEHomepageManualAssignmentMentorsList pointed to that page."""
    spans, start = [], None
    for c in cfg:
        on = bool(c.get("manual_list"))
        if on and start is None:
            start = c["timestamp"]
        elif not on and start is not None:
            spans.append((start, c["timestamp"]))
            start = None
    if start is not None:
        spans.append((start, None))
    return spans


def build_snapshots(inp: dict, uid_map: dict[str, int], issues: IssueLog) -> list[dict]:
    """Merge revisions from the three sources into one list of snapshots sorted by time.

    Each snapshot: {ts, source, revid, editor, comment, state},
    state = {key: {"in_auto_pool", "weight_known", "username"}},
    key is the uid (int); a name with no resolved uid is kept as "NAME:<name>" (str).

    Wikitext period: whenever either the automatic list page or the manual list page gets a revision,
    one snapshot is built from the "current content" of both pages. If a person is on both pages, the
    automatic list wins (in_auto_pool=True). The manual list only counts during the time spans when the
    config setting was in effect.
    """
    snaps: list[dict] = []
    windows = manual_list_window(inp["cfg"])

    def manual_active(ts: str) -> bool:
        return any(s <= ts and (e is None or ts < e) for s, e in windows)

    reported: set[str] = set()

    def key_of(name: str):
        uid = uid_map.get(name)
        if uid is None:
            if name not in reported:      # log each name only once
                reported.add(name)
                issues.add("unresolved_mentor_name", rule="R6.3", action="flagged", ids={"name": name})
            return f"NAME:{name}"
        return uid

    # ── Wikitext period (ends at the first revision of the JSON roster)
    seq = [("auto", r) for r in inp["wiki"] if r["timestamp"] < SWITCH] + \
          [("manual", r) for r in inp["manual"] if r["timestamp"] < SWITCH]
    seq.sort(key=lambda x: (x[1]["timestamp"], x[0]))
    cur = {"auto": [], "manual": []}
    for which, rev in seq:
        cur[which] = parse_wikitext_mentors(get_content(rev))
        ts = rev["timestamp"]
        state: dict = {}
        if manual_active(ts):
            for n in cur["manual"]:
                state[key_of(n)] = {"in_auto_pool": False, "weight_known": False, "username": n}
        for n in cur["auto"]:
            state[key_of(n)] = {"in_auto_pool": True, "weight_known": False, "username": n}
        snaps.append({"ts": ts, "source": f"wikitext_{which}", "revid": rev.get("revid"),
                      "editor": rev.get("user", ""), "comment": rev.get("comment", ""), "state": state})

    # ── JSON period
    for rev in inp["json"]:
        if rev["timestamp"] < SWITCH:
            continue
        entries = parse_json_mentors(get_content(rev))
        if entries is None:
            issues.add("json_parse_error", rule="R1.1", action="flagged", ids={"revid": rev.get("revid")},
                       evidence=get_content(rev)[:300])
            continue
        if not entries:
            continue
        state = {}
        for uid, e in entries.items():
            try:
                k = int(uid)
            except ValueError:
                continue
            pool, wk = json_entry_in_auto_pool(e)
            state[k] = {"in_auto_pool": pool, "weight_known": wk, "username": norm_user(e.get("username")) or ""}
        snaps.append({"ts": rev["timestamp"], "source": "json", "revid": rev.get("revid"),
                      "editor": rev.get("user", ""), "comment": rev.get("comment", ""), "state": state})
    snaps.sort(key=lambda s: s["ts"])
    return snaps


def diff_snapshots_to_events(snaps: list[dict]) -> tuple[list[dict], dict]:
    """Compare each snapshot with the one before it, keeping a running state, and produce joined / left / auto_pool_change events.

    Continuity across sources: if a person is on the roster in both the wikitext period and the JSON
    period, the switch does not create a false "left, then joined again", because both sides use the
    same uid key (same as eval2).

    Returns:
        (list of events, final state at the cutoff)
    """
    events, prev = [], {}
    for s in snaps:
        cur = s["state"]
        base = {"timestamp": s["ts"], "source": s["source"], "editor": s["editor"],
                "comment": s["comment"], "revid": s["revid"]}
        for k in cur.keys() - prev.keys():
            events.append({**base, "key": k, "username": cur[k]["username"], "event": "joined",
                           "in_auto_pool": cur[k]["in_auto_pool"], "weight_known": cur[k]["weight_known"]})
        for k in prev.keys() - cur.keys():
            events.append({**base, "key": k, "username": prev[k]["username"], "event": "left",
                           "in_auto_pool": None, "weight_known": prev[k]["weight_known"]})
        for k in cur.keys() & prev.keys():
            if cur[k]["in_auto_pool"] != prev[k]["in_auto_pool"]:
                events.append({**base, "key": k, "username": cur[k]["username"], "event": "auto_pool_change",
                               "in_auto_pool": cur[k]["in_auto_pool"], "weight_known": cur[k]["weight_known"],
                               "in_auto_pool_from": prev[k]["in_auto_pool"]})
        prev = cur
    return events, prev


# ════════════════════════════════════════════════════════════════════════════
# Output tables
# ════════════════════════════════════════════════════════════════════════════


def _uid(k):
    """Turn a key into the mentor_uid used in the output (None if the uid was not resolved)."""
    return k if isinstance(k, int) else None


def build_mentor_list(events: list[dict], final: dict) -> pd.DataFrame:
    """Summary table with one row per mentor."""
    by = defaultdict(list)
    for e in events:
        by[e["key"]].append(e)
    rows = []
    for k, ev in by.items():
        pools = [e["in_auto_pool"] for e in ev if e["event"] != "left" and e["in_auto_pool"] is not None]
        rows.append({
            "mentor_uid": _uid(k), "key": str(k),
            "username": next((e["username"] for e in reversed(ev) if e["username"]), str(k)),
            "uid_resolved": isinstance(k, int),
            "first_joined": min(e["timestamp"] for e in ev if e["event"] == "joined"),
            "last_left": max((e["timestamp"] for e in ev if e["event"] == "left"), default=None),
            "is_current": k in final,
            "currently_in_auto_pool": final[k]["in_auto_pool"] if k in final else None,
            "ever_in_auto_pool": any(p is True for p in pools),
            "ever_manual_only": any(p is False for p in pools),
            "join_count": sum(e["event"] == "joined" for e in ev),
            "leave_count": sum(e["event"] == "left" for e in ev),
            "sources": ",".join(sorted({e["source"] for e in ev})),
        })
    return pd.DataFrame(rows).sort_values("first_joined").reset_index(drop=True)


def build_periods(events: list[dict]) -> pd.DataFrame:
    """Turn events into status periods: within each period [start, end) the status does not change.

    Rules:
        joined / auto_pool_change starts a new period (and ends the previous one); left ends the
        current period and does not start a new one.
        The last period has an empty end, meaning the status still holds at the cutoff.
    M4 uses this to ask "was the mentor in the automatic pool when the newcomer registered?": it finds
    the period that contains the registration time.
    """
    by = defaultdict(list)
    for e in sorted(events, key=lambda e: e["timestamp"]):
        by[e["key"]].append(e)
    rows = []
    for k, ev in by.items():
        open_seg = None
        for e in ev:
            if open_seg is not None:
                open_seg["end"] = e["timestamp"]
                rows.append(open_seg)
                open_seg = None
            if e["event"] != "left":
                open_seg = {"mentor_uid": _uid(k), "key": str(k), "start": e["timestamp"], "end": None,
                            "in_auto_pool": e["in_auto_pool"], "weight_known": e["weight_known"],
                            "source": e["source"]}
        if open_seg is not None:
            rows.append(open_seg)
    df = pd.DataFrame(rows)
    df["hours"] = (pd.to_datetime(df["end"], utc=True) - pd.to_datetime(df["start"], utc=True)).dt.total_seconds() / 3600
    return df


def build_names(snaps: list[dict], uid_map: dict[str, int]) -> pd.DataFrame:
    """Every name each mentor has used, and the time range when it appears (M2 uses it to identify who signed a message; M4 uses it to match logs).

    Sources:
        roster: names seen in roster snapshots (the username in JSON entries is updated after a rename);
        uid_map: the wikitext period name-to-uid lookup (includes old names from before a rename);
        alt_account: mentors' alternate accounts registered in config.MENTOR_ALT_ACCOUNTS.
    """
    seen: dict[tuple, list[str]] = {}
    for s in snaps:
        for k, info in s["state"].items():
            if isinstance(k, int) and info["username"]:
                seen.setdefault((k, info["username"]), []).append(s["ts"])
    rows = [{"mentor_uid": k, "name": n, "first_seen": min(t), "last_seen": max(t), "source": "roster"}
            for (k, n), t in seen.items()]
    have = {(r["mentor_uid"], r["name"]) for r in rows}
    for n, uid in uid_map.items():
        if (uid, n) not in have:
            rows.append({"mentor_uid": uid, "name": n, "first_seen": None, "last_seen": None, "source": "uid_map"})
    # Mentors with no user name in the JSON roster: use the name collect_roster.py found by uid (source api_users)
    p = RAW_DIR / "uid_names.json"
    if p.exists():
        for uid, n in json.loads(p.read_text()).items():
            if n and (int(uid), norm_user(n)) not in have:
                rows.append({"mentor_uid": int(uid), "name": norm_user(n), "first_seen": None, "last_seen": None,
                             "source": "api_users"})
    name_to_uid = {r["name"]: r["mentor_uid"] for r in rows}
    for alt, main in config.MENTOR_ALT_ACCOUNTS.items():
        uid = name_to_uid.get(norm_user(main))
        if uid is not None:
            rows.append({"mentor_uid": uid, "name": norm_user(alt), "first_seen": None, "last_seen": None,
                         "source": "alt_account"})
    return pd.DataFrame(rows).drop_duplicates(["mentor_uid", "name"]).sort_values(["mentor_uid", "name"])


def monthly_pool(periods: pd.DataFrame) -> pd.DataFrame:
    """At the last moment of each month: how many mentors are on the roster, and how many of them are in the automatic pool. Used to compare with published numbers."""
    months = pd.date_range("2021-05-31", config.DATA_CUTOFF.strftime("%Y-%m-%d"), freq="ME", tz="UTC")
    s = pd.to_datetime(periods["start"], utc=True)
    e = pd.to_datetime(periods["end"], utc=True)
    rows = []
    for m in months:
        t = m + pd.Timedelta(hours=23, minutes=59)
        on = (s <= t) & (e.isna() | (e > t))
        rows.append({"month": m.strftime("%Y-%m"), "listed": int(on.sum()),
                     "auto_pool": int((on & (periods["in_auto_pool"] == True)).sum()),  # noqa: E712
                     "manual_only": int((on & (periods["in_auto_pool"] == False)).sum())})  # noqa: E712
    return pd.DataFrame(rows)


def compare_with_eval2(ml: pd.DataFrame, eval2_path, issues: IssueLog) -> pd.DataFrame:
    """Compare with eval2's frozen roster mentor by mentor: present in both, first join time, last leave time, currently active.

    Expected differences:
        - People only in this version: manual list mentors from the wikitext period (eval2 did not parse that list);
        - Time differences: eval2 includes revisions exactly at the cutoff moment (this version uses strictly
          before), and status changes that come from the manual list.
    """
    e2 = pd.DataFrame(read_jsonl(eval2_path)).rename(columns={"user_id": "mentor_uid"})
    a = ml[ml.uid_resolved].set_index("mentor_uid")
    b = e2.set_index("mentor_uid")
    rows = []
    for uid in sorted(set(a.index) | set(b.index)):
        ra, rb = (a.loc[uid] if uid in a.index else None), (b.loc[uid] if uid in b.index else None)
        if ra is None or rb is None:
            kind = "only_new" if rb is None else "only_eval2"
        else:
            diffs = [f for f, fb in (("first_joined", "first_joined"), ("last_left", "last_left"),
                                     ("is_current", "is_current")) if str(ra[f]) != str(rb[fb])]
            kind = ",".join(diffs) if diffs else "same"
        if kind != "same":
            issues.add("differs_from_eval2", rule="R1.1", action="flagged",
                       ids={"uid": int(uid), "name": (ra if ra is not None else rb)["username"]},
                       detail={"kind": kind,
                               "new": None if ra is None else {k: str(ra[k]) for k in ("first_joined", "last_left", "is_current", "sources")},
                               "eval2": None if rb is None else {k: str(rb[k]) for k in ("first_joined", "last_left", "is_current", "sources")}})
        rows.append({"mentor_uid": uid, "kind": kind})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Main program
# ════════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    base_parser("M1: rebuild the mentor roster (offline).").parse_args(argv)
    log = setup_logging("m1")
    t0 = time.time()
    issues = IssueLog("m1")
    inp = load_inputs()
    log.info("revisions before cutoff: wikitext %d, json %d, manual-list %d",
             len(inp["wiki"]), len(inp["json"]), len(inp["manual"]))
    uid_map = build_uid_map(inp)
    snaps = build_snapshots(inp, uid_map, issues)
    events, final = diff_snapshots_to_events(snaps)
    ml = build_mentor_list(events, final)
    periods = build_periods(events)
    names = build_names(snaps, uid_map)
    # The display user name is the name seen last on the roster. A rename with no status change does
    # not create an event, so we cannot use "the name in the last event" (example: Rubbish computer → Alextejthompson)
    latest = (names[names.source.isin(["roster", "api_users"])].sort_values("last_seen", na_position="first")
              .groupby("mentor_uid").name.last())
    ml["username"] = [latest.get(u, n) if u is not None else n for u, n in zip(ml.mentor_uid, ml.username)]
    ev = pd.DataFrame([{**{k: v for k, v in e.items() if k != "key"}, "mentor_uid": _uid(e["key"]), "key": str(e["key"])}
                       for e in events])

    for r in periods[periods["hours"] < TRANSIENT_HOURS].itertuples():
        issues.add("transient_pool_state", rule="R1.1", action="flagged",
                   ids={"uid": r.mentor_uid, "key": r.key},
                   detail={"start": r.start, "end": r.end, "in_auto_pool": r.in_auto_pool, "hours": round(r.hours, 3)})
    cmp = compare_with_eval2(ml, inp["src"]["eval2_mentor_list"], issues)

    # Invariants
    assert_unique(ml, "key", "mentor_list")
    assert_no_after_cutoff(ml, ["first_joined"], "mentor_list")
    assert_no_after_cutoff(ev, ["timestamp"], "mentor_events")
    assert_unique(names, ["mentor_uid", "name"], "mentor_names")

    write_table(OUT / "mentor_list.parquet", ml)
    write_table(OUT / "mentor_periods.parquet", periods)
    write_table(OUT / "mentor_events.parquet", ev)
    write_table(OUT / "mentor_names.parquet", names)
    res = config.RESULTS / "m1"
    res.mkdir(parents=True, exist_ok=True)
    mp = monthly_pool(periods)
    write_table(res / "monthly_pool.csv", mp)
    write_table(res / "compare_with_eval2.csv", cmp)
    issues.write()

    counts = {"mentors": len(ml), "uid_unresolved": int((~ml.uid_resolved).sum()),
              "current": int(ml.is_current.sum()),
              "current_auto": int((ml.currently_in_auto_pool == True).sum()),  # noqa: E712
              "ever_manual_only": int(ml.ever_manual_only.sum()),
              "only_in_wikitext_manual": int(ml.sources.eq("wikitext_manual").sum()),
              "events": len(ev), "periods": len(periods), "names": len(names),
              "eval2_compare": cmp["kind"].value_counts().to_dict(), "issues": {t: issues.count(t) for t in issues.records}}
    for k, v in counts.items():
        log.info("%-24s %s", k, v)
    for line in issues.summary_lines():
        log.info(line)
    write_manifest("m1", inputs=list(inp["src"].values()) + [RAW_DIR / "manual_list_revisions.jsonl",
                   RAW_DIR / "config_revisions.jsonl", RAW_DIR / "uid_map_extra.json"],
                   outputs=[OUT / f for f in ("mentor_list.parquet", "mentor_periods.parquet",
                                              "mentor_events.parquet", "mentor_names.parquet")]
                   + [res / "monthly_pool.csv", res / "compare_with_eval2.csv"],
                   params={"cutoff": config.DATA_CUTOFF_ISO, "source_switch": SWITCH}, counts=counts, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
