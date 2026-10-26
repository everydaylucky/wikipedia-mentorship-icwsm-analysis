"""
build_m1_check.py: writes checks/m1_check.ipynb, the check notebook for M1 (the mentor roster)

The notebook follows the fixed 8-section structure (PLAN.md section 12):
    1 module description  2 files and manifest  3 invariants  4 comparison with the previous version (eval2)
    5 tracer samples and random samples (raw roster revisions next to processed values)
    6 edge cases  7 distributions  8 summary (results/m1_report.json)

The key idea: the notebook replays the raw roster revisions with its own, much simpler code
(section 3.2), and then compares every derived field with the M1 output, for all mentors and for
random mentors shown in detail.

How to run:
    python3 checks/build_m1_check.py          # write the notebook
    python3 checks/build_m1_check.py --run    # write it and run it from top to bottom
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks._nbgen import code, md, run_as_script, write_notebook  # noqa: E402

CELLS = [
md("""
# M1 mentor roster: check notebook

## 1. What this module does

**Code**: `pipeline/m1_roster/` (`collect_roster.py` downloads the raw data, `build_roster.py` rebuilds the roster offline).
**Rules**: DATA_PROCESSING R1.1 to R1.3 (roster formats, cutoff, uid as key) and R6.1 to R6.3 (names and uids).

Growth mentors sign up on a roster page. The roster had three formats over time:

1. **Wikitext period** (2021-05-04 to 2022-10-26). Two pages. A line `* [[User:Name]]|intro` on
   `Wikipedia:Growth Team features/Mentor list` means the mentor is in the **automatic pool** (new
   accounts are assigned to them at random). The same line on `.../Mentor list/Manual` means the mentor
   can only be claimed by hand. The manual page only counts while the wiki setting points to it
   (from 2021-09-20). If someone is on both pages, the automatic page wins.
2. **JSON period with `automaticallyAssigned`** (from 2022-10-26): `{"Mentors": {uid: {username, weight, automaticallyAssigned}}}`.
3. **JSON period with only `weight`**: weight 0 means "not in the automatic pool".

M1 replays every revision of these pages before the cutoff (2026-04-20T14:00:00Z) and writes four tables:
`mentor_list` (one row per mentor), `mentor_periods` (time spans with a constant status),
`mentor_events` (joined / left / auto_pool_change) and `mentor_names` (every name a mentor used).

**Paper numbers that depend on it**: whether the assigned mentor was in the automatic pool when the newcomer
registered (M4 sample eligibility, D10); who counts as a mentor when M2 labels a reply; the number of mentors (Sec 3.1).

## How to use this notebook

* Run all cells from the top (about 1 minute). Every check prints `[OK ]` or `[FAIL]`.
* Section 3 replays the raw revisions with **separate, simple code written in this notebook** and compares
  the result with M1 for all mentors. Section 5 shows random mentors in detail, with links to the
  exact revisions where their status changed, so you can open them on Wikipedia.
* Change `SEED` in the next cell to draw a new random sample.
"""),
code(r'''
# ── setup ────────────────────────────────────────────────────────────────────
SEED = 0          # change this number to draw a fresh random sample in section 5
N_RANDOM = 5      # how many random mentors / names to show

import json, re, sys, html as _html
from collections import defaultdict
from pathlib import Path
import pandas as pd
ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.common.io import read_jsonl, iter_jsonl
from pipeline.common.tracer import tracer_frame
from checks._show import Checks, compare, pick, oldid, diff, contribs, logid
pd.set_option("display.max_colwidth", 100); pd.set_option("display.width", 200)
M1 = config.INTERIM / "m1"
RAW1 = config.RAW / "m1_roster"
CUTOFF = config.DATA_CUTOFF_ISO
ml = pd.read_parquet(M1 / "mentor_list.parquet")
periods = pd.read_parquet(M1 / "mentor_periods.parquet")
events = pd.read_parquet(M1 / "mentor_events.parquet")
names = pd.read_parquet(M1 / "mentor_names.parquet")
mp = pd.read_csv(config.RESULTS / "m1" / "monthly_pool.csv").set_index("month")
print(f"mentors {len(ml)} | periods {len(periods)} | events {len(events)} | names {len(names)} | months {len(mp)}")
'''),
md("## 2. Files and manifest"),
code(r'''
man = json.loads((M1 / "manifest.json").read_text())
print("finished:", man["finished_at"], "| cutoff:", man["data_cutoff"], "| elapsed:", man["elapsed_sec"], "s")
display(pd.DataFrame(man["inputs"])[["path", "exists", "bytes", "mtime"]])
display(pd.DataFrame(man["outputs"])[["path", "bytes"]])
print(json.dumps(man["counts"], ensure_ascii=False, indent=1))
'''),
md("""
### Data dictionary

Every output column, what it means, and where this notebook checks it.
"Replay" means the independent replay in section 3.2; "5.R" means the random side-by-side examples in section 5.
"""),
code(r'''
DD = {
 "mentor_list": [
  ("mentor_uid", "Wikipedia user id of the mentor (primary key, R1.3)", "3.1 unique; 3.3 replay; 5.R"),
  ("key", "same as mentor_uid as text, or NAME:<name> if no uid was found", "3.1 key == str(uid)"),
  ("username", "the name the mentor had the last time they were seen on the roster", "3.3 replay (last name seen); 5.R"),
  ("uid_resolved", "True if the name could be turned into a uid", "3.1 all True"),
  ("first_joined", "time of the first roster revision that lists the mentor", "3.3 replay; 5.R"),
  ("last_left", "time of the last roster revision that removed the mentor (empty if never removed)", "3.3 replay; 5.R"),
  ("is_current", "still on the roster in the last revision before the cutoff", "3.3 replay; 5.R"),
  ("currently_in_auto_pool", "in the automatic pool in the last revision before the cutoff (empty if not on the roster)", "3.3 replay; 5.R"),
  ("ever_in_auto_pool", "was in the automatic pool at some time", "3.3 replay; 5.R"),
  ("ever_manual_only", "was on the roster but outside the automatic pool at some time", "3.3 replay; 5.R"),
  ("join_count", "number of times the mentor was added to the roster", "3.3 replay; 5.R"),
  ("leave_count", "number of times the mentor was removed from the roster", "3.3 replay; 5.R"),
  ("sources", "which roster formats produced events for this mentor", "3.3 replay; 5.R"),
 ],
 "mentor_periods": [
  ("mentor_uid / key", "the mentor", "3.1 every period belongs to a mentor in mentor_list"),
  ("start", "time the status began (a roster revision time)", "3.3 replay, period by period; 5.R"),
  ("end", "time the status ended (empty: still holds at the cutoff)", "3.3 replay; 3.1 start < end"),
  ("in_auto_pool", "True = automatic pool, False = on the roster but manual only", "3.3 replay; 5.R"),
  ("weight_known", "True if the JSON entry had automaticallyAssigned or weight (always False in the wikitext period)", "3.3 replay"),
  ("source", "roster format of the revision that started the period", "3.3 replay"),
  ("hours", "end minus start in hours", "3.1 recomputed from start and end"),
 ],
 "mentor_events": [
  ("timestamp / revid", "the roster revision where the change happened", "3.1 revid exists in the raw revisions with this timestamp; 5.R links"),
  ("editor / comment", "who saved that revision and their edit summary", "3.1 equal to the raw revision"),
  ("event", "joined, left or auto_pool_change", "3.3 replay (number of events per mentor); 5.R"),
  ("in_auto_pool / in_auto_pool_from", "status after (and before, for auto_pool_change)", "3.1 consistency with periods; 5.R"),
  ("username / mentor_uid / key / source / weight_known", "as above", "3.1, 3.3"),
 ],
 "mentor_names": [
  ("mentor_uid", "the mentor", "3.1 every uid is in mentor_list"),
  ("name", "one normalized name the mentor used (on the roster, in the uid map, or an alt account)", "3.3 replay; 5.N random names"),
  ("first_seen / last_seen", "first and last roster revision that shows this name for this uid (empty for non-roster sources)", "3.3 replay; 5.N"),
  ("source", "roster, uid_map, api_users or alt_account", "5.N shows the raw evidence for each source"),
 ],
 "monthly_pool.csv": [
  ("month / listed / auto_pool / manual_only", "number of mentors on the roster at the last minute of each month, split by pool", "3.3 replay; section 7 public figures"),
 ],
}
for t, rows in DD.items():
    print(f"\n## {t}")
    display(pd.DataFrame(rows, columns=["column", "meaning", "checked in"]))
'''),
md("""
## 3. Invariants

### 3.1 Structure of the output tables
"""),
code(r'''
ck = Checks()
ck.add("mentor_list has 450 mentors (number at the time this notebook was written)", len(ml) == 450)
ck.add("every name was resolved to a uid", ml.uid_resolved.all())
ck.add("key is unique", ml.key.is_unique)
ck.add("key == str(mentor_uid)", (ml.key == ml.mentor_uid.astype(str)).all())
ck.add("no event at or after the cutoff", (pd.to_datetime(events.timestamp, utc=True) < config.DATA_CUTOFF).all())
ck.add("no mentor joined at or after the cutoff", (pd.to_datetime(ml.first_joined, utc=True) < config.DATA_CUTOFF).all())
ck.add("every period starts before it ends", (periods.end.isna() | (periods.start < periods.end)).all())
ck.add("periods of the same mentor do not overlap",
       all((g.start.iloc[1:].values >= g.end.iloc[:-1].fillna("9999").values).all()
           for _, g in periods.sort_values("start").groupby("key") if len(g) > 1))
ck.add("at most one open period (empty end) per mentor", periods[periods.end.isna()].key.is_unique)
ck.add("is_current == the mentor has an open period",
       set(ml[ml.is_current].key) == set(periods[periods.end.isna()].key))
cur = periods[periods.end.isna()].set_index("key").in_auto_pool
ck.add("currently_in_auto_pool == in_auto_pool of the open period",
       all(ml[ml.is_current].apply(lambda r: r.currently_in_auto_pool == cur[r.key], axis=1)))
ck.add("not current -> currently_in_auto_pool is empty", ml[~ml.is_current].currently_in_auto_pool.isna().all())
h = (pd.to_datetime(periods.end, utc=True) - pd.to_datetime(periods.start, utc=True)).dt.total_seconds() / 3600
ck.add("hours == end - start", ((h - periods.hours).abs().fillna(0) < 1e-6).all() and (h.isna() == periods.hours.isna()).all())
ck.add("every period, event and name belongs to a mentor in mentor_list",
       set(periods.key) <= set(ml.key) and set(events.key) <= set(ml.key) and set(names.mentor_uid) <= set(ml.mentor_uid))
ck.add("join_count == number of joined events", (ml.set_index("key").join_count ==
       events[events.event == "joined"].groupby("key").size().reindex(ml.key).fillna(0).values).all())
ck.add("leave_count == number of left events", (ml.set_index("key").leave_count ==
       events[events.event == "left"].groupby("key").size().reindex(ml.key).fillna(0).values).all())
ck.add("(mentor_uid, name) is unique in mentor_names", not names.duplicated(["mentor_uid", "name"]).any())
ck.add("periods: in_auto_pool is never empty", periods.in_auto_pool.notna().all())
'''),
md("""
### 3.2 Independent replay of the raw roster revisions

The code below does **not** call any M1 function. It reads the three raw revision files and, for each
revision, works out the status of every mentor: `auto` (automatic pool), `manual` (manual only) or absent.
Its parsing rules are written out in plain Python:

* wikitext: remove `<!-- -->` comments, keep lines starting with `*`, take the name after `[[User:`;
  normalize the name (underscores to spaces, first letter upper case, drop `/subpage`);
* the manual page counts only while `config_revisions.jsonl` says the wiki setting points to it;
* JSON: use `automaticallyAssigned` if present, else `weight != 0`;
* names in the wikitext period become uids through the JSON roster itself (which has both), then the old uid map.
"""),
code(r'''
SWITCH = "2022-10-26T13:19:10Z"     # first revision of the JSON roster (DATA_PROCESSING section 1)

def content(rev):
    m = rev.get("slots", {}).get("main", {})
    return m.get("content", m.get("*", "")) or ""

def my_name(n):
    n = _html.unescape(n).replace("_", " ").split("/")[0]
    n = re.sub(r"[\s ‎‏]+", " ", n).strip()
    return n[:1].upper() + n[1:]

def my_wikitext_names(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.search(r"\[\[\s*user\s*:\s*([^\]|#]+)", line, flags=re.I) if line.startswith("*") else None
        if m and my_name(m.group(1)) and my_name(m.group(1)) not in out:
            out.append(my_name(m.group(1)))
    return out

def my_json_status(entry):
    if "automaticallyAssigned" in entry: return ("auto" if entry["automaticallyAssigned"] else "manual"), True
    if entry.get("weight") is not None:  return ("auto" if entry["weight"] != 0 else "manual"), True
    return None, False

raw_wiki = [r for r in iter_jsonl(config.EXTERNAL["roster_revisions_wikitext"]) if r["timestamp"] < CUTOFF]
raw_json = [r for r in iter_jsonl(config.EXTERNAL["roster_revisions_json"]) if r["timestamp"] < CUTOFF]
raw_manual = [r for r in read_jsonl(RAW1 / "manual_list_revisions.jsonl") if r["timestamp"] < CUTOFF]
cfg = read_jsonl(RAW1 / "config_revisions.jsonl")
print(f"raw revisions before the cutoff: auto wikitext {len(raw_wiki)}, manual wikitext {len(raw_manual)}, JSON {len(raw_json)}")
print("wiki setting changes for the manual list:", [(c["timestamp"], c["manual_list"]) for c in cfg])

# name -> uid: JSON roster entries first, then the old uid map, then names resolved by collect_roster.py
json_entries = {}                                   # revid -> parsed entries (None if not valid JSON)
for r in raw_json:
    try:
        d = json.loads(content(r)); d = d.get("Mentors", d)
        json_entries[int(r["revid"])] = {k: v for k, v in d.items() if isinstance(v, dict)}
    except (json.JSONDecodeError, AttributeError):
        json_entries[int(r["revid"])] = None
name2uid = {}
for k, v in json.loads((RAW1 / "uid_map_extra.json").read_text()).items():
    if v.get("uid") is not None: name2uid[my_name(k)] = int(v["uid"])
for k, v in json.loads(config.EXTERNAL["roster_uid_map"].read_text()).items():
    if v is not None: name2uid[my_name(k)] = int(v)
for ents in json_entries.values():
    for uid, e in (ents or {}).items():
        if e.get("username") and uid.isdigit(): name2uid[my_name(e["username"])] = int(uid)

def manual_on(ts):
    on = False
    for c in sorted(cfg, key=lambda c: c["timestamp"]):
        if c["timestamp"] <= ts: on = bool(c.get("manual_list"))
    return on

# one snapshot per revision: {uid: (status, weight_known, name)}
seq = [("wikitext_auto", r) for r in raw_wiki if r["timestamp"] < SWITCH] + \
      [("wikitext_manual", r) for r in raw_manual if r["timestamp"] < SWITCH]
seq.sort(key=lambda x: (x[1]["timestamp"], x[0]))
snaps, page_now = [], {"wikitext_auto": [], "wikitext_manual": []}
for src, r in seq:
    page_now[src] = my_wikitext_names(content(r))
    st = {}
    if manual_on(r["timestamp"]):
        for n in page_now["wikitext_manual"]: st[name2uid.get(n, "NAME:" + n)] = ("manual", False, n)
    for n in page_now["wikitext_auto"]: st[name2uid.get(n, "NAME:" + n)] = ("auto", False, n)
    snaps.append((r["timestamp"], int(r["revid"]), src, st))
for r in raw_json:
    ents = json_entries[int(r["revid"])]
    if r["timestamp"] < SWITCH or not ents: continue
    st = {int(u): (*my_json_status(e), my_name(e.get("username") or "")) for u, e in ents.items() if u.isdigit()}
    snaps.append((r["timestamp"], int(r["revid"]), "json", st))
snaps.sort(key=lambda s: s[0])
print("snapshots replayed:", len(snaps), "| JSON revisions that are not valid JSON:",
      sum(v is None for v in json_entries.values()))

# walk through the snapshots and record, for each mentor, every revision where the status changes
timeline = defaultdict(list)      # uid -> [(ts, revid, source, old status, new status)]
name_seen = defaultdict(list)     # (uid, name) -> [ts]
prev = {}
for ts, rid, src, st in snaps:
    for u in set(st) | set(prev):
        old = prev[u][0] if u in prev else "absent"
        new = st[u][0] if u in st else "absent"
        if old != new: timeline[u].append((ts, rid, src, old, new, st[u][1] if u in st else prev[u][1]))
    for u, v in st.items():
        if isinstance(u, int) and v[2]: name_seen[(u, v[2])].append(ts)
    prev = st
final = prev
print("mentors found by the replay:", len(timeline))
'''),
code(r'''
uid_names = json.loads((RAW1 / "uid_names.json").read_text())     # uid -> name, looked up with the API by collect_roster.py
print("mentors whose JSON entries never had a username:", uid_names)

def my_summary(u):
    """Everything mentor_list and mentor_periods say about one mentor, recomputed from the replay timeline."""
    tl = timeline[u]
    statuses = [x[4] for x in tl if x[4] != "absent"]
    last_name = max(((max(ts), n) for (uu, n), ts in name_seen.items() if uu == u), default=(None, None))[1]
    per, open_ = [], None
    for ts, rid, src, old, new, wk in tl:
        if open_: open_["end"] = ts; per.append(open_); open_ = None
        if new != "absent": open_ = {"start": ts, "end": None, "in_auto_pool": new == "auto", "weight_known": wk, "source": src}
    if open_: per.append(open_)
    if last_name is None:     # no name on the roster (JSON entry without username): the name found by uid through the API
        last_name = uid_names.get(str(u))
    return {"username": last_name,
            "first_joined": next(x[0] for x in tl if x[3] == "absent"),
            "last_left": max((x[0] for x in tl if x[4] == "absent"), default=None),
            "is_current": u in final,
            "currently_in_auto_pool": (final[u][0] == "auto") if u in final else None,
            "ever_in_auto_pool": "auto" in statuses, "ever_manual_only": "manual" in statuses,
            "join_count": sum(x[3] == "absent" for x in tl), "leave_count": sum(x[4] == "absent" for x in tl),
            "sources": ",".join(sorted({x[2] for x in tl})), "periods": per}

def same(a, b):
    """Equal, treating None and NaN as the same empty value."""
    na, nb = a is None or (isinstance(a, float) and pd.isna(a)), b is None or (isinstance(b, float) and pd.isna(b))
    return (na and nb) or (not na and not nb and a == b)

LIST_FIELDS = ["username", "first_joined", "last_left", "is_current", "currently_in_auto_pool",
               "ever_in_auto_pool", "ever_manual_only", "join_count", "leave_count", "sources"]
mine = {u: my_summary(u) for u in timeline}
mli = ml.set_index("mentor_uid")
def data_periods(u):
    return [{"start": r.start, "end": r.end, "in_auto_pool": bool(r.in_auto_pool), "weight_known": bool(r.weight_known),
             "source": r.source} for r in periods[periods.mentor_uid == u].sort_values("start").itertuples()]
mism = []
for u, s in mine.items():
    if u not in mli.index: mism.append((u, "missing from mentor_list", None, None)); continue
    for f in LIST_FIELDS:
        a, b = s[f], mli.loc[u, f]
        if not same(a, b): mism.append((u, f, a, b))
    if s["periods"] != data_periods(u): mism.append((u, "periods", len(s["periods"]), len(data_periods(u))))
for u in set(mli.index) - set(mine): mism.append((u, "in mentor_list but not found by the replay", None, None))
mism = pd.DataFrame(mism, columns=["mentor_uid", "field", "replay", "in data"])
'''),
md("### 3.3 Replay versus M1 output, for every mentor and every month"),
code(r'''
ck.add("replay finds the same set of mentors as mentor_list", set(mine) == set(mli.index))
for f in LIST_FIELDS + ["periods"]:
    n = int((mism.field == f).sum())
    ck.add(f"mentor_list.{f} equals the replay for all mentors (mismatches: {n})" if f != "periods"
           else f"mentor_periods (start, end, in_auto_pool, weight_known, source) equal the replay (mismatching mentors: {n})", n == 0)
if len(mism): print("\nMISMATCH rows:"); display(mism.head(30))

# events: one event per status change in the replay; editor, comment and timestamp come from the raw revision
ck.add("number of events per mentor == number of status changes in the replay",
       events.groupby("mentor_uid").size().to_dict() == {u: len(t) for u, t in timeline.items()})
raw_by_id = {int(r["revid"]): r for r in raw_wiki + raw_manual + raw_json}
ev_ok = events.apply(lambda e: int(e.revid) in raw_by_id and raw_by_id[int(e.revid)]["timestamp"] == e.timestamp
                     and (raw_by_id[int(e.revid)].get("user") or "") == e.editor
                     and (raw_by_id[int(e.revid)].get("comment") or "") == e.comment, axis=1)
ck.add(f"every event's revid, timestamp, editor and comment match the raw revision (bad: {int((~ev_ok).sum())})", ev_ok.all())

# names seen on the roster
nm_roster = names[names.source == "roster"]
nm_ok = nm_roster.apply(lambda r: (r.mentor_uid, r["name"]) in name_seen
                        and min(name_seen[(r.mentor_uid, r["name"])]) == r.first_seen
                        and max(name_seen[(r.mentor_uid, r["name"])]) == r.last_seen, axis=1)
ck.add(f"mentor_names (source roster): name, first_seen, last_seen equal the replay (bad: {int((~nm_ok).sum())})", nm_ok.all())
ck.add("every (uid, name) the replay saw is in mentor_names",
       set(name_seen) <= set(zip(names.mentor_uid, names["name"])))

# month-end counts recomputed from the replay
def count_at(t):
    c = {"listed": 0, "auto_pool": 0, "manual_only": 0}
    for u, s in mine.items():
        for p in s["periods"]:
            if p["start"] <= t and (p["end"] is None or p["end"] > t):
                c["listed"] += 1; c["auto_pool" if p["in_auto_pool"] else "manual_only"] += 1
    return c
mon_bad = []
for m, r in mp.iterrows():
    t = (pd.Timestamp(m + "-01") + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%dT23:59:00Z")
    c = count_at(t)
    if c != {"listed": r.listed, "auto_pool": r.auto_pool, "manual_only": r.manual_only}: mon_bad.append((m, c, r.to_dict()))
ck.add(f"monthly_pool.csv equals month-end counts from the replay (bad months: {len(mon_bad)})", not mon_bad)
for x in mon_bad[:5]: print("  MISMATCH", x)
ck.report()
'''),
md("""
## 4. Comparison with the previous version (eval2 frozen roster)

Expected: every difference comes from the **manual list** of the wikitext period (eval2 did not parse it),
or from eval2 using "not later than the cutoff" where M1 uses "strictly before".
"""),
code(r'''
cmp = pd.read_csv(config.RESULTS / "m1" / "compare_with_eval2.csv")
print(cmp.kind.value_counts().to_string())
dfe = pd.DataFrame([{"uid": r["ids"]["uid"], "name": r["ids"]["name"], "kind": r["detail"]["kind"],
                     "new_joined": (r["detail"]["new"] or {}).get("first_joined", "")[:10],
                     "eval2_joined": (r["detail"]["eval2"] or {}).get("first_joined", "")[:10],
                     "new_left": (r["detail"]["new"] or {}).get("last_left", "")[:10],
                     "eval2_left": (r["detail"]["eval2"] or {}).get("last_left", "")[:10],
                     "new_sources": (r["detail"]["new"] or {}).get("sources")}
                    for r in read_jsonl(M1 / "_issues" / "differs_from_eval2.jsonl")])
display(dfe)
expl = dfe.new_sources.fillna("").str.contains("wikitext_manual")
print(f"\ndifferences explained by the manual list (new_sources contains wikitext_manual): {int(expl.sum())} of {len(dfe)}")
print("not explained by the manual list (look at these by hand):")
display(dfe[~expl])
'''),
md("""
## 5. Tracer samples and random samples: raw revisions next to processed values

### 5.T Fixed tracer samples

The mentors in the locked tracer list (pipeline/common/tracer.py), plus Xaosflux (uid 502540), who left
the automatic list on 2021-06-07 ("i don't want to be autoassinged") and has been on the manual list since 2021-09-20.
For each one: the M1 row, the periods, and every raw revision where the replay saw the status change.
"""),
code(r'''
def show_mentor(uid):
    if uid not in mli.index:
        print(f"\n### uid {uid}: NOT in mentor_list (a finding to explain)"); return True
    r = mli.loc[uid]
    print(f"\n### {r.username} (uid {uid})   {contribs(r.username)}")
    print("  names in mentor_names:", list(names[names.mentor_uid == uid].name))
    print("  raw revisions where the status changes (replay):")
    for ts, rid, src, old, new, wk in timeline.get(uid, []):
        print(f"    {ts}  {src:<16} {old:>6} -> {new:<6}  {diff(rid)}   summary: {raw_by_id[rid].get('comment', '')[:60]!r}")
    print("  periods in mentor_periods:")
    display(periods[periods.mentor_uid == uid][["start", "end", "in_auto_pool", "weight_known", "source"]])
    s = mine.get(uid)
    if s is None: print("  replay did not find this mentor"); return False
    return compare({f: s[f] for f in LIST_FIELDS} | {"n_periods": len(s["periods"])},
                   {f: (None if not isinstance(r[f], str) and pd.isna(r[f]) else r[f]) for f in LIST_FIELDS}
                   | {"n_periods": len(data_periods(uid))}, title="  replay (left) vs mentor_list (right):")

tr = tracer_frame()
sample_uids = list(dict.fromkeys([int(u) for u in tr[tr.kind == "mentor"].uid.dropna()] + [502540]))
tracer_ok = [show_mentor(u) for u in sample_uids]
'''),
code(r'''
# Raw text check for Xaosflux: in which revisions of each wikitext page does the name appear?
def presence(revs, name):
    out, prev = [], None
    for r in revs:
        on = name in my_wikitext_names(content(r))
        if on != prev: out.append((r["timestamp"], int(r["revid"]), "listed" if on else "not listed", (r.get("comment") or "")[:70]))
        prev = on
    return out
for label, revs in (("automatic list page", [r for r in raw_wiki if r["timestamp"] < SWITCH]), ("manual list page", raw_manual)):
    print(f"\n{label}:")
    for x in presence(revs, "Xaosflux"): print("  ", x, oldid(x[1]))
'''),
md("""
### 5.R Random mentors

`N_RANDOM` mentors drawn at random (change `SEED` at the top for others), plus `N_RANDOM` random mentors
who have left at least once and `N_RANDOM` who were manual-only at some time, because those are the
harder cases. For each: the raw revisions where the status changes (open the diff links to see the line
being added or removed), the line that mentions them in the raw page text, and replay vs data.
"""),
code(r'''
def raw_line(uid, rid):
    """The raw roster text about this mentor in one revision (the wikitext line or the JSON entry)."""
    r = raw_by_id[rid]
    if rid in json_entries and json_entries[rid] is not None:
        e = json_entries[rid].get(str(uid))
        return json.dumps({k: v for k, v in e.items() if k != "message"}) if e else "(no entry for this uid)"
    names_u = [n for n, u in name2uid.items() if u == uid]
    hit = [ln.strip() for ln in content(r).split("\n") if ln.strip().startswith("*") and any(n.lower() in ln.lower().replace("_", " ") for n in names_u)]
    return hit[0][:120] if hit else "(not listed on this page)"

def show_random(uid):
    ok = show_mentor(uid)
    print("  raw text about this mentor in the revision of each change:")
    for ts, rid, src, old, new, wk in timeline.get(uid, [])[:8]:
        print(f"    {ts} {oldid(rid)}\n        {raw_line(uid, rid)}")
    return ok

rand_ok = []
for label, pool in (("all mentors", ml), ("mentors who left at least once", ml[ml.leave_count > 0]),
                    ("mentors who were manual-only at some time", ml[ml.ever_manual_only])):
    print(f"\n================ random: {label} (n={len(pool)}, SEED={SEED})")
    rand_ok += [show_random(int(u)) for u in pick(pool, N_RANDOM, SEED).mentor_uid]
print(f"\nrandom mentors where replay == data: {sum(rand_ok)} of {len(rand_ok)}")
'''),
md("""
### 5.N Random names: how a name maps to a uid

M2 uses `mentor_names` to decide whether a signature belongs to a mentor. For random names: the raw
evidence (the JSON roster entry with this uid and username, the old uid map, the API lookup, or the
alt-account table in config.py).
"""),
code(r'''
uid_map_old = {my_name(k): v for k, v in json.loads(config.EXTERNAL["roster_uid_map"].read_text()).items()}
def name_evidence(r):
    if r.source == "roster":
        rev = next((rid for ts, rid, src, st in snaps if st.get(r.mentor_uid, (0, 0, ""))[2] == r["name"]), None)
        src = next((s for ts, rid, s, st in snaps if rid == rev), None)
        ev = raw_line(r.mentor_uid, rev) if src == "json" else f"wikitext line, and name2uid[{r['name']!r}] = {name2uid.get(r['name'])}"
        return f"first roster revision showing it: {oldid(rev) if rev else None}\n        {ev}"
    if r.source == "uid_map":  return f"old uid map: {r['name']!r} -> {uid_map_old.get(r['name'])}"
    if r.source == "api_users": return f"uid_names.json (API lookup by uid): {r.mentor_uid} -> {uid_names.get(str(r.mentor_uid))!r}"
    if r.source == "alt_account":
        return f"config.MENTOR_ALT_ACCOUNTS: {[(a, m) for a, m in config.MENTOR_ALT_ACCOUNTS.items() if my_name(a) == r['name']]}"
samp = pd.concat([pick(names, N_RANDOM, SEED), names[names.source != "roster"]]).drop_duplicates(["mentor_uid", "name"])
for _, r in samp.iterrows():
    seen = name_seen.get((r.mentor_uid, r["name"]))
    print(f"\n{r['name']!r} -> uid {r.mentor_uid}  (source {r.source})  {contribs(r['name'])}")
    print("   ", name_evidence(r))
    if r.source == "roster":
        compare({"first_seen": min(seen), "last_seen": max(seen)}, {"first_seen": r.first_seen, "last_seen": r.last_seen})
'''),
md("""
## 6. Edge cases

* All issue counts written by M1.
* **Short-lived states** (under 1 hour): usually someone editing the roster twice in a row. Random examples with the diff to open.
* Mentors who **joined more than once** (left and came back), and mentors whose name **changed** while on the roster.
"""),
code(r'''
summ = json.loads((M1 / "_issues" / "summary.json").read_text())
print(json.dumps(summ, ensure_ascii=False, indent=1))
tr_ = pd.DataFrame(read_jsonl(M1 / "_issues" / "transient_pool_state.jsonl"))
tr_["hours"] = tr_.detail.map(lambda d: d["hours"]); tr_["start"] = tr_.detail.map(lambda d: d["start"])
tr_["uid"] = tr_.ids.map(lambda d: d.get("uid"))
print(f"\nstates lasting less than 1 hour: {len(tr_)} periods, median {tr_.hours.median()*60:.1f} minutes")
for r in pick(tr_, N_RANDOM, SEED).itertuples():
    ch = [x for x in timeline.get(r.uid, []) if x[0] == r.start]
    rid = ch[0][1] if ch else None
    print(f"  uid {r.uid} ({mli.loc[r.uid, 'username'] if r.uid in mli.index else '?'})  start {r.start}  lasted {r.hours*3600:.0f} s"
          f"  {ch[0][3] + ' -> ' + ch[0][4] if ch else ''}  {diff(rid) if rid else ''}")
'''),
code(r'''
multi = ml[ml.join_count > 1]
print(f"mentors who joined more than once: {len(multi)}")
display(pick(multi, N_RANDOM, SEED)[["mentor_uid", "username", "first_joined", "last_left", "join_count", "leave_count", "is_current", "sources"]])
renamed = names[names.source == "roster"].groupby("mentor_uid").name.nunique()
renamed = renamed[renamed > 1]
print(f"\nmentors seen on the roster under more than one name: {len(renamed)}")
for u in renamed.index[:10]:
    print(f"  uid {u}: " + ", ".join(f"{r['name']} ({r.first_seen[:10]} to {r.last_seen[:10]})"
                                    for _, r in names[(names.mentor_uid == u) & (names.source == "roster")].sort_values("first_seen").iterrows())
          + f"   -> username in mentor_list: {mli.loc[u, 'username']}")
print("\nentries with neither automaticallyAssigned nor weight (status unknown):",
      sum(1 for s in snaps if s[2] == "json" for v in s[3].values() if v[0] is None))
'''),
md("""
## 7. Distributions: mentors on the roster at each month end

Public figures to compare with: about 83 mentors in the automatic pool in 2022-11 and about 187 in 2025-02
(discussion around T384505). The red dots are the public figures.
"""),
code(r'''
display(mp.loc[["2021-05", "2021-09", "2022-10", "2022-11", "2023-11", "2024-06", "2025-02", "2026-03"]])
ax = mp.plot(figsize=(10, 3.5), title="Mentors on the roster at month end")
ax.scatter([mp.index.get_loc("2022-11"), mp.index.get_loc("2025-02")], [83, 187], color="red", zorder=5, label="public figures")
print("period length in days (closed periods):")
print((periods.hours.dropna() / 24).describe().round(1).to_string())
'''),
md("## 8. Summary (written to results/m1_report.json, read by M9)"),
code(r'''
report = {"n_mentors": len(ml), "n_current": int(ml.is_current.sum()),
          "n_current_auto": int((ml.currently_in_auto_pool == True).sum()),
          "n_ever_manual_only": int(ml.ever_manual_only.sum()),
          "auto_pool_2022_11": int(mp.loc["2022-11", "auto_pool"]), "auto_pool_2025_02": int(mp.loc["2025-02", "auto_pool"]),
          "differs_from_eval2": int((cmp.kind != "same").sum())}
(config.RESULTS / "m1_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
print(f"\nchecks passed: {sum(ok for _, ok in ck.items)} / {len(ck.items)}; "
      f"random and tracer mentors matching the replay: {sum(rand_ok) + sum(tracer_ok)} / {len(rand_ok) + len(tracer_ok)}")
'''),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "m1_check.ipynb"
    write_notebook(CELLS, out)
    print("wrote", out)
    if "--run" in sys.argv:
        run_as_script(out, cwd=out.parent.parent)
        print("\nnotebook ran end to end")


if __name__ == "__main__":
    main()
