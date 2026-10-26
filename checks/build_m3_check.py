"""
build_m3_check.py: writes checks/m3_check.ipynb, the check notebook for M3 (newcomer activity data)

The notebook follows the fixed 8 sections of PLAN.md section 12:
    1 module description and data dictionary   2 files and manifests   3 invariants (recomputed from raw files)
    4 comparison with the previous report and with independent data   5 tracer samples and random samples
    (raw records next to processed values)   6 edge-case lists   7 distributions   8 summary (results/m3_report.json)

How to run:
    python checks/build_m3_check.py          # writes the notebook
    python checks/build_m3_check.py --run    # writes it and runs it from top to bottom (about 2 to 3 minutes)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks._nbgen import code, md, run_as_script, write_notebook  # noqa: E402

CELLS = [
md("""
# M3 newcomer activity data: check notebook

**What the module does.** For every person who asked a mentor a question (an "asker", from M2), M3 builds:

* a **profile** (`mentee_profile.parquet`): registration time, edit count, block status, user groups, current name,
  the account creation log entry (how the account was made: `create` means the person registered themselves), and
  counts of edits, logs and abuse filter hits;
* all **edits** before the data cutoff (`mentee_edits/`), all **log entries** (`mentee_logs.parquet`) and all
  **abuse filter hits** (`mentee_abuse.parquet`);
* the **revert time** of edits tagged `mw-reverted` (`data/raw/m3/revert_times*.jsonl`; that job is still running).

**How it is built (the steps this notebook covers).**

1. `index_s6.py` re-indexes the s6 files (fetched 2026-04-23 to 24, stored by user name) by user ID and keeps only askers.
2. `collect_topup.py` fetches by user ID, after the cutoff, the askers that s6 does not have (`topup`), and the askers
   whose s6 edit history stops too early (`stale.json`; these become `s6+topup`). It also looks up account creation
   logs by registration time (`creation.jsonl`).
3. `merge_activity.py` merges s6 and the top-up, removes duplicates, drops anything on or after the cutoff
   (2026-04-20T14:00:00Z), and writes the profile, edits, logs and abuse tables.
4. `revert_times.py` finds, for edits tagged `mw-reverted`, the revision that reverted them and when (sha1 method).

**Rules**: DATA_PROCESSING R2.2 (how the account was created), R6.1 to R6.3 (identity by user ID), R7.1, R7.6, R7.7, R7.8.

**Paper numbers that depend on M3.** All pre-question covariates and post-question outcomes in M7 (edit counts,
namespaces, reverts, abuse filter hits, account age), the account creation action used by M4 for the sample rule D10,
and the `late_fetch` sensitivity analysis in M8.

**How to use this notebook.** Run it from top to bottom (about 2 to 3 minutes). Section 3 recomputes every derived
field from the raw files and prints `[OK ]` or `[FAIL]`. Section 5 puts raw records next to processed values for
fixed tracer people and for random people; change `SEED` in the setup cell to draw new random people. Every link
opens the record on Wikipedia so you can check it by eye.
"""),
code(r'''
# ── 1. Setup ──────────────────────────────────────────────────────────────
SEED = 0        # change SEED (any whole number) to draw a fresh random sample in section 5
N_RANDOM = 5    # how many random examples to show per group in section 5

import json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, orjson
ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.common.io import read_jsonl, iter_jsonl
from pipeline.common.tracer import tracer_frame
from checks._show import Checks, compare, pick, oldid, diff, contribs, logid
pd.set_option("display.max_colwidth", 80); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
T0 = time.time()
M3, S6, RAW3 = config.INTERIM / "m3", config.INTERIM / "m3" / "s6", config.RAW / "m3"
CUT = config.DATA_CUTOFF_ISO
# All times in these files are text like "2024-01-31T12:00:00Z". Text in this format sorts in time order,
# so "ts < CUT" is a simple and exact cutoff test (section 3 first checks that every time has this format).
ISO = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$"

prof = pd.read_parquet(M3 / "mentee_profile.parquet")
P = prof.set_index("mentee_uid")
edits = ds.dataset(M3 / "mentee_edits").to_table(
    columns=["uid", "revid", "parentid", "ns", "timestamp", "size", "sizediff", "tags", "source"]).to_pandas()
logs = pd.read_parquet(M3 / "mentee_logs.parquet")
abuse = pd.read_parquet(M3 / "mentee_abuse.parquet")
q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["question_revid", "mentee_uid", "q_ts"])
q = q[q.mentee_uid.notna()].astype({"mentee_uid": "int64"})
ask = set(q.mentee_uid)
prev_report = json.loads((config.RESULTS / "m3_report.json").read_text()) if (config.RESULTS / "m3_report.json").exists() else None

def clean(v):
    """Turns pandas and numpy values into plain Python values so raw and processed values can be compared."""
    if v is None or v is pd.NA: return None
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, (np.ndarray, list, tuple)): return [str(x) for x in v]
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    return v

# Random people for section 5 are drawn here, so the large raw files are read only once (in section 3.0).
SAMPLE = {g: list(pick(prof[prof.source == g], N_RANDOM, SEED).mentee_uid) for g in ("s6", "s6+topup", "topup")}
SAMPLE_UIDS = {u for v in SAMPLE.values() for u in v}
ck = Checks()
print(f"askers {len(prof):,} | edits {len(edits):,} | logs {len(logs):,} | abuse filter hits {len(abuse):,}")
print("random people drawn with SEED =", SEED, ":", SAMPLE)
'''),
md("""
### Data dictionary

One table per output. "Checked in" names the notebook cell that verifies the column (3.x are invariants over all rows,
5.x are random raw-versus-processed examples, 4.x are comparisons with independent data). The cell after each table
confirms that the dictionary lists every column in the file.
"""),
code(r'''
DICT = {}
DICT["mentee_profile"] = {
    "mentee_uid": ("user ID of the asker (primary key)", "3.1 one row per asker"),
    "source": ("where the data came from: s6, s6+topup (s6 was incomplete, fetched again) or topup (not in s6)", "3.1 recomputed from raw file membership; 5.2 to 5.4"),
    "late_fetch": ("True when the person was fetched after the cutoff (top-up), so later deletions and tags cannot be undone", "3.1"),
    "fetched_at": ("when the profile was fetched (2026-04-23 for s6, the top-up time otherwise)", "3.1; 5.2 to 5.4"),
    "name_at_fetch": ("user name at fetch time", "3.1 against raw s6 profile or raw top-up user; 5.2 to 5.4"),
    "registration": ("registration time from the profile (falls back to the 2026-10 registration file when empty)", "3.1; 4.3 against an independent June fetch; 5.2 to 5.4"),
    "editcount": ("MediaWiki edit counter at fetch time (includes deleted edits)", "3.1; 5.2 to 5.4; 7"),
    "blocked": ("blocked at fetch time", "3.1; 5.2 to 5.4"),
    "groups": ("user groups at fetch time", "3.1; 5.2 to 5.4"),
    "hidden_or_missing": ("the API said the user is hidden or missing (top-up only)", "3.1"),
    "current_name": ("user name in the 2026-10 registration file", "3.1; 5.2 to 5.4"),
    "creation_action": ("action of the account creation log: create = registered themselves; create2/byemail = made by another user; autocreate = came from another wiki", "3.4 recomputed for all; 5.2 to 5.4 raw log record"),
    "creation_ts": ("time of the account creation log entry", "3.4; 5.2 to 5.4; 6 (gap to registration)"),
    "creation_performer": ("user who performed the creation (the person themselves for create)", "3.4; 5.2 to 5.4"),
    "creation_logid": ("log ID of the creation entry (open with the logid link)", "3.4; 5.2 to 5.4"),
    "creation_source": ("log = found in the person's own logs; lookup = found by registration time (creation.jsonl); none = not found", "3.4; 6"),
    "n_edits": ("number of rows of this person in mentee_edits", "3.1 recount; 5.2 to 5.4 recount from raw"),
    "n_logs": ("number of rows of this person in mentee_logs", "3.1 recount; 3.3 from raw; 5.2 to 5.4"),
    "n_abuse": ("number of rows of this person in mentee_abuse", "3.1 recount; 5.2 to 5.4 from raw"),
}
DICT["mentee_edits"] = {
    "uid": ("user ID of the editor", "3.2 (uid, revid) pairs come from the raw files; 5.2 to 5.4"),
    "revid": ("revision ID (unique)", "3.2 unique and equal to the union of raw s6 and raw top-up; 5.2 to 5.4"),
    "parentid": ("revision ID of the page version before this edit", "3.2 all top-up rows vs raw; 5.2 to 5.4 field check"),
    "ns": ("namespace number (0 = article)", "3.2 top-up rows; 5.2 to 5.4"),
    "title": ("page title at fetch time", "3.2 top-up rows; 5.2 to 5.4"),
    "timestamp": ("time of the edit; must be before the cutoff", "3.2 format and cutoff; 5.2 to 5.4"),
    "size": ("page size in bytes after the edit", "3.2 top-up rows; 5.2 to 5.4"),
    "sizediff": ("bytes added (negative = removed)", "3.2 top-up rows; 5.2 to 5.4"),
    "tags": ("change tags at fetch time (for example mw-reverted)", "3.2 top-up rows; 5.2 to 5.4"),
    "minor": ("marked as a minor edit", "3.2 top-up rows; 5.2 to 5.4"),
    "new": ("the edit created the page", "3.2 top-up rows; 5.2 to 5.4"),
    "comment": ("edit summary", "3.2 top-up rows; 5.2 to 5.4"),
    "source": ("s6 or topup (when both have the edit, s6 is kept)", "3.2"),
}
DICT["mentee_logs"] = {
    "uid": ("user ID of the asker the entry belongs to", "3.3 key set equals raw"),
    "logid": ("log ID; (uid, logid) is unique", "3.3"),
    "type": ("log type (newusers, block, rights, ...)", "3.3 every row vs raw"),
    "action": ("log action (create, autocreate, ...)", "3.3 every row vs raw"),
    "timestamp": ("time of the log entry; before the cutoff", "3.3 every row vs raw"),
    "ns": ("namespace of the target page", "3.3 every row vs raw"),
    "title": ("target page", "3.3 every row vs raw"),
    "performer": ("user who performed the action", "3.3 every row vs raw"),
    "params": ("log parameters as JSON text (newusers entries hold the new account's userid)", "3.3 every row vs raw"),
    "comment": ("log comment", "3.3 every row vs raw"),
    "source": ("s6 or topup (when both have it, s6 is kept)", "3.3"),
}
DICT["mentee_abuse"] = {
    "uid": ("user ID of the asker", "3.3 key set"),
    "abuse_id": ("abuse log ID; (uid, abuse_id) is unique", "3.3"),
    "filter_id": ("ID of the filter that was hit", "3.3 top-up rows vs raw; 5.2 to 5.4"),
    "filter": ("name of the filter", "3.3 top-up rows; 5.2 to 5.4"),
    "action": ("what the user tried to do (edit, createaccount, ...)", "3.3 top-up rows; 5.2 to 5.4"),
    "result": ("what the filter did (warn, disallow, tag, empty = log only)", "3.3 top-up rows; 5.2 to 5.4"),
    "timestamp": ("time of the hit; before the cutoff", "3.3; 5.2 to 5.4"),
    "ns": ("namespace of the page", "3.3 top-up rows; 5.2 to 5.4"),
    "title": ("page", "3.3 top-up rows; 5.2 to 5.4"),
    "source": ("s6 or topup", "3.3"),
}
DICT["revert_times (raw jsonl)"] = {
    "revid": ("the edit tagged mw-reverted", "3.5 is an mw-reverted edit of the same person"),
    "uid": ("its editor", "3.5"),
    "ts": ("time of the edit", "3.5 equals the edit time"),
    "fetched_at": ("when the lookup was made", "3.5 after the cutoff"),
    "method": ("sha1 = page returned to an earlier version; undo_comment = an undo naming the edit; page_deleted; not_found; api_error", "3.5; 6"),
    "revert_revid": ("revision that reverted the edit", "3.5 later than the edit; 5.5 links"),
    "revert_ts": ("time of the revert", "3.5 after the edit; 5.5"),
    "lag_hours": ("hours from the edit to the revert", "3.5 recomputed; 5.5"),
    "scanned": ("for not_found: how many later revisions were checked", "6"),
    "error": ("for api_error: the error text", "6"),
}
def show_dict(name, cols):
    t = pd.DataFrame([{"column": k, "meaning": v[0], "checked in": v[1]} for k, v in DICT[name].items()])
    print(f"\n{name}"); print(t.to_string(index=False, max_colwidth=110))
    if name.startswith("revert_times"):   # optional fields (scanned, error) appear only once such a result exists
        ck.add(f"data dictionary of {name} lists every field in the file", set(cols) <= set(DICT[name]))
    else:
        ck.add(f"data dictionary of {name} lists exactly the columns in the file", set(DICT[name]) == set(cols))
show_dict("mentee_profile", prof.columns)
show_dict("mentee_edits", ds.dataset(M3 / "mentee_edits").schema.names)
show_dict("mentee_logs", logs.columns)
show_dict("mentee_abuse", abuse.columns)
_rt_cols = set().union(*[set(r) for p in RAW3.glob("revert_times*.jsonl") for r in read_jsonl(p)[:5000]])
show_dict("revert_times (raw jsonl)", _rt_cols)
'''),
md("""
## 2. Files and manifests

Each step writes a manifest (finish time, cutoff, input and output files, counts). The checks below confirm that
the manifests use the same cutoff as `config.py` and that the counts in the M3 manifest equal the tables on disk.
"""),
code(r'''
mans = {}
for name in ("m3_index", "m3_collect", "m3"):
    p = config.INTERIM / name / "manifest.json"
    man = mans[name] = json.loads(p.read_text())
    print(f"\n[{name}] finished {man['finished_at']} | cutoff {man['data_cutoff']} | {man['elapsed_sec']} s")
    display(pd.DataFrame(man["inputs"] + man["outputs"])[[c for c in ("path", "exists", "bytes", "mtime") if c in pd.DataFrame(man["inputs"] + man["outputs"]).columns]])
    print(json.dumps(man["counts"], ensure_ascii=False)[:900])
    ck.add(f"{name} manifest uses the cutoff in config.py", man["data_cutoff"] == CUT)
c3 = mans["m3"]["counts"]
ck.add("m3 manifest counts equal the tables on disk (askers, edits, logs, abuse)",
       (c3["askers"], c3["edits"], c3["logs"], c3["abuse"]) == (len(prof), len(edits), len(logs), len(abuse)))
ck.add("the merge (m3) ran after both m3_index and m3_collect",
       max(mans["m3_index"]["finished_at"], mans["m3_collect"]["finished_at"]) <= mans["m3"]["finished_at"])
# collect_topup used coverage.json from m3_index; if m3_index ran again later, its list of askers missing from s6 must be unchanged
cov = json.loads((S6 / "coverage.json").read_text())
ck.add("askers missing from s6 in coverage.json = the askers with source 'topup'",
       set(cov["missing_uids"]) == set(prof[prof.source == "topup"].mentee_uid))
print("\nraw files of M3 (data/raw/m3):")
for p in sorted(RAW3.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size / 1e6:9.1f} MB  modified {pd.Timestamp(p.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M}")
'''),
md("""
## 3. Invariants (all should be OK)

### 3.0 Read the raw files once

Everything in section 3 is recomputed from the **raw** files, not from the intermediate s6 index, where possible:
the raw s6 profile, edit, log and abuse files (stored by user name), and the raw top-up files. For the two very large
s6 files (edits 3.4 GB, abuse 5.1 GB) we read only the counter at the start of each line, plus the full line for the
random people of section 5. This takes about 30 to 60 seconds.
"""),
code(r'''
t = time.time()
HEAD = re.compile(rb'^\{"username":\s*"((?:[^"\\]|\\.)*)"')
def head_name(line):
    m = HEAD.match(line)
    return json.loads(b'"' + m.group(1) + b'"') if m else None

# raw s6 profiles: user name at fetch time -> profile record (only records that have a user ID)
s6_prof_raw = {}
for line in open(config.EXTERNAL["s6_profiles"], "rb"):
    r = orjson.loads(line)
    if r.get("userid") is not None:
        s6_prof_raw[r["username"]] = r
s6_name = {r["userid"]: n for n, r in s6_prof_raw.items() if r["userid"] in ask}     # asker uid -> s6 name
name2uid = {n: u for u, n in s6_name.items()}
ck.add("no asker has two different names in the raw s6 profiles", len(name2uid) == len(s6_name))

# raw s6 edits: the edit_count field at the start of each line; the full line only for sampled people
EC = re.compile(rb'"edit_count":\s*"?(\d+)')
s6_edit_count, s6_full = {}, {}
for line in open(config.EXTERNAL["s6_contribs"], "rb"):
    u = name2uid.get(head_name(line))
    if u is None: continue
    m = EC.search(line[:3000]); s6_edit_count[u] = int(m.group(1)) if m else None
    if u in SAMPLE_UIDS: s6_full[u] = orjson.loads(line)["edits"]

# raw s6 logs: small enough to parse every asker line
s6_log_rows = []
for line in open(config.EXTERNAL["s6_logevents"], "rb"):
    u = name2uid.get(head_name(line))
    if u is None: continue
    for e in orjson.loads(line).get("events") or []:
        s6_log_rows.append((u, e.get("logid"), e.get("type"), e.get("action"), e.get("timestamp"), e.get("ns"), e.get("title"),
                            e.get("user"), e.get("params") or {}, e.get("comment"), "s6"))

# raw s6 abuse filter hits: the abuse_count field; the full line only for sampled people
AC = re.compile(rb'"abuse_count":\s*"?(\d+)')
s6_abuse_count, s6_abuse_full = {}, {}
for line in open(config.EXTERNAL["s6_abuselog"], "rb"):
    u = name2uid.get(head_name(line))
    if u is None: continue
    m = AC.search(line[:3000]); s6_abuse_count[u] = int(m.group(1)) if m else None
    if u in SAMPLE_UIDS: s6_abuse_full[u] = orjson.loads(line).get("entries") or []

# raw top-up: topup.jsonl, then topup_stale.jsonl; a later record for the same uid replaces an earlier one (as in merge_activity)
topup, n_topup_lines = {}, Counter()
for fname in ("topup.jsonl", "topup_stale.jsonl"):
    for line in open(RAW3 / fname, "rb"):
        r = orjson.loads(line); n_topup_lines[fname] += 1
        if r["uid"] in ask:
            topup[r["uid"]] = dict(r, _file=fname)
creation_raw = {}
for r in iter_jsonl(RAW3 / "creation.jsonl"):
    creation_raw.setdefault(r["uid"], []).append(r)
stale = {int(k): v for k, v in json.loads((RAW3 / "stale.json").read_text()).items()}
stale_edits = {r["uid"]: r["edits"] for r in iter_jsonl(RAW3 / "stale_edits.jsonl")}
reg_now = json.loads(config.RAW_FILES["registration"].read_text())
print(f"raw s6: {len(s6_name):,} askers in profiles, {len(s6_edit_count):,} edit lines, {len(s6_log_rows):,} log events, "
      f"{len(s6_abuse_count):,} abuse lines | top-up lines {dict(n_topup_lines)}, askers {len(topup):,} | "
      f"creation.jsonl {len(creation_raw):,} uids | stale.json {len(stale):,} ({time.time() - t:.0f} s)")
'''),
md("""
### 3.1 Profile

`source` is recomputed from which raw files contain the person; the profile fields are recomputed from the raw s6
profile line (s6 people) or the raw top-up `user` record (top-up people), with the documented fallback to the
2026-10 registration file when the registration time is empty.
"""),
code(r'''
def expected_profile(uid):
    """Profile fields of one asker, built directly from the raw records."""
    now = reg_now.get(str(uid)) or {}
    if uid in topup:
        u = topup[uid]["user"]
        return {"source": "s6+topup" if uid in s6_name else "topup", "late_fetch": True, "fetched_at": topup[uid]["fetched_at"],
                "name_at_fetch": u.get("name"), "registration": u.get("registration") or now.get("registration"),
                "editcount": u.get("editcount"), "blocked": "blockid" in u, "groups": list(u.get("groups") or []),
                "hidden_or_missing": bool(u.get("missing") or u.get("hidden")), "current_name": now.get("name")}
    if uid in s6_name:
        r = s6_prof_raw[s6_name[uid]]
        return {"source": "s6", "late_fetch": False, "fetched_at": "2026-04-23", "name_at_fetch": r["username"],
                "registration": r.get("registration") or now.get("registration"), "editcount": r.get("editcount"),
                "blocked": bool(r.get("blocked")), "groups": list(r.get("groups") or []), "hidden_or_missing": False,
                "current_name": now.get("name")}
    return {"source": "none", "current_name": now.get("name")}

exp = {u: expected_profile(u) for u in P.index}
ck.add("one profile row per asker, and the set of uids equals the M2 askers", prof.mentee_uid.is_unique and set(prof.mentee_uid) == ask)
ck.add("source is s6, s6+topup or topup for everyone (no asker without data)", prof.source.isin(["s6", "s6+topup", "topup"]).all())
bad_fields = {}
for f in ("source", "late_fetch", "fetched_at", "name_at_fetch", "registration", "editcount", "blocked", "groups",
          "hidden_or_missing", "current_name"):
    bad = [u for u in P.index if f in exp[u] and clean(P.at[u, f]) != clean(exp[u][f])]
    bad_fields[f] = bad
    ck.add(f"profile.{f} equals the raw value for every asker (mismatches: {len(bad)})", not bad)
ck.add("late_fetch is True exactly for s6+topup and topup", (prof.late_fetch == (prof.source != "s6")).all())
for col, tbl in (("n_edits", edits), ("n_logs", logs), ("n_abuse", abuse)):
    cnt = tbl.groupby("uid").size()
    ck.add(f"profile.{col} equals the number of rows of the person in its table", (P[col] == cnt.reindex(P.index).fillna(0).astype(int)).all())
for f, bad in bad_fields.items():
    if bad:
        print(f"\n{f}: first mismatches"); print([(u, clean(P.at[u, f]), exp[u].get(f)) for u in bad[:5]])
'''),
md("""
### 3.2 Edits

The merged edit table must be exactly the union of the s6 edits and the raw top-up edits before the cutoff, with
each revision once. For top-up rows every field is compared with the raw API record. For s6 rows we can only check
the counter at the start of each raw line here (the full field check of s6 rows is done for the random people in 5.2).
"""),
code(r'''
t = time.time()
ck.add("every edit time has the ISO format YYYY-MM-DDTHH:MM:SSZ", edits.timestamp.str.match(ISO).all())
ck.add("no edit on or after the cutoff", (edits.timestamp < CUT).all())
ck.add("revid is unique (duplicates removed)", edits.revid.is_unique)
ck.add("every editor is an asker", edits.uid.isin(ask).all())
ck.add("source is s6 or topup", edits.source.isin(["s6", "topup"]).all())

s6idx = ds.dataset(S6 / "edits").to_table(columns=["uid", "revid", "timestamp"]).to_pandas()
tp_rows = [(u, e.get("revid"), e.get("parentid"), e.get("ns"), e.get("title"), e.get("timestamp"), e.get("size"), e.get("sizediff"),
            list(e.get("tags") or []), bool(e.get("minor")), bool(e.get("new")), e.get("comment"))
           for u, r in topup.items() for e in r.get("contribs") or []]
tpe = pd.DataFrame(tp_rows, columns=["uid", "revid", "parentid", "ns", "title", "timestamp", "size", "sizediff", "tags", "minor", "new", "comment"])
tpe_in = tpe[tpe.timestamp < CUT]
s6set, tpset, merged = set(s6idx.revid), set(tpe_in.revid), set(edits.revid)
print(f"s6 index edits {len(s6set):,} | raw top-up edits {len(tpe):,} (on or after cutoff, dropped: {len(tpe) - len(tpe_in):,}) "
      f"| top-up edits already in s6 {len(tpset & s6set):,} | merged {len(merged):,}")
ck.add("merged revids = s6 revids + raw top-up revids before the cutoff (nothing lost, nothing extra)", merged == s6set | tpset)
ck.add("no raw top-up edit on or after the cutoff got into the table", not (set(tpe[tpe.timestamp >= CUT].revid) & merged))
ck.add("source is s6 exactly when the revid is in the s6 index", ((edits.source == "s6") == edits.revid.isin(s6set)).all())
pairs = set(zip(s6idx.uid, s6idx.revid)) | set(zip(tpe_in.uid, tpe_in.revid))
ck.add("every (uid, revid) pair in the table is a pair from the raw files", set(zip(edits.uid, edits.revid)) <= pairs)

# s6 index against the raw s6 lines: per person, the index has at most the raw edit_count (fewer only because of the cutoff)
n_idx = s6idx.groupby("uid").size()
raw_ec = pd.Series(s6_edit_count)
ck.add("every asker in the raw s6 edit file has edit_count >= edits in the s6 index", (n_idx.reindex(raw_ec.index).fillna(0) <= raw_ec).all())
dropped = mans["m3_index"]["counts"]["edits"]["dropped_after_cutoff"]
ck.add(f"sum of raw s6 edit_count ({int(raw_ec.sum()):,}) = s6 index edits ({len(s6idx):,}) + dropped after cutoff ({dropped:,})",
       int(raw_ec.sum()) == len(s6idx) + dropped)

# full field check of every top-up row against the raw API record
mt = ds.dataset(M3 / "mentee_edits").to_table(filter=ds.field("source") == "topup").to_pandas()
j = mt.merge(tpe_in.drop_duplicates("revid"), on="revid", suffixes=("", "_raw"))
ck.add("every top-up row in the table has its raw record", len(j) == len(mt))
for f in ("uid", "parentid", "ns", "title", "timestamp", "size", "sizediff", "tags", "minor", "new", "comment"):
    n_bad = sum(clean(a) != clean(b) for a, b in zip(j[f], j[f + "_raw"]))
    ck.add(f"top-up rows: {f} equals the raw API value ({n_bad} differ)", n_bad == 0)

# stale people: the edits added for them are exactly the ones find_stale found, and they are newer than their last s6 edit
st_tp = {u: r for u, r in topup.items() if r["_file"] == "topup_stale.jsonl"}
ck.add("for every stale person, top-up edits = the edits in stale_edits.jsonl",
       all({e["revid"] for e in r["contribs"]} == {e["revid"] for e in stale_edits.get(u, [])} for u, r in st_tp.items()))
last_s6 = s6idx.groupby("uid").timestamp.max()
newer = tpe_in[tpe_in.uid.isin([u for u, v in stale.items() if v["reason"] == "newer_edits"])]
ck.add("edits added for 'newer_edits' people are all later than their last s6 edit", (newer.timestamp > newer.uid.map(last_s6)).all())
ck.add("every stale.json person is an s6+topup asker", all(P.at[u, "source"] == "s6+topup" for u in stale if u in P.index))

# identity and completeness (R6.1, R7.6)
first_edit = edits.groupby("uid").timestamp.min()
reg = P.registration.dropna()
early = reg.index[reg.values > first_edit.reindex(reg.index).fillna("9999").values]
ck.add(f"registration is not later than the first edit ({len(early)} askers fail)", len(early) == 0)
ck.add("every question revision of M2 is in its asker's edits (R7.6)",
       set(zip(q.mentee_uid, q.question_revid)) <= set(zip(edits.uid, edits.revid)))
ck.add("every asker has at least one edit (the question itself is an edit)", (prof.n_edits > 0).all())
print(f"({time.time() - t:.0f} s)")
'''),
md("""
### 3.3 Logs and abuse filter hits

Logs: the raw s6 log file is small, so **every** log row (s6 and top-up) is compared with its raw record, field by
field. Abuse filter hits: the key set is checked for everyone, every top-up row is compared with the raw record, and
s6 rows are compared with the raw file for the random people in 5.2.
"""),
code(r'''
t = time.time()
LC = ["uid", "logid", "type", "action", "timestamp", "ns", "title", "performer", "params", "comment", "source"]
tp_log_rows = [(u, e.get("logid"), e.get("type"), e.get("action"), e.get("timestamp"), e.get("ns"), e.get("title"), e.get("user"),
                e.get("params") or {}, e.get("comment"), "topup") for u, r in topup.items() for e in r.get("logs") or []]
rawlogs = pd.DataFrame(s6_log_rows + tp_log_rows, columns=LC)
rawlogs = rawlogs[rawlogs.timestamp < CUT].drop_duplicates(["uid", "logid"], keep="first")   # s6 rows come first, so s6 wins
ck.add("every log time has the ISO format", logs.timestamp.str.match(ISO).all())
ck.add("no log entry on or after the cutoff", (logs.timestamp < CUT).all())
ck.add("(uid, logid) is unique in mentee_logs", not logs.duplicated(["uid", "logid"]).any())
ck.add("log keys (uid, logid) = raw s6 keys + raw top-up keys before the cutoff",
       set(zip(logs.uid, logs.logid.astype(int))) == set(zip(rawlogs.uid, rawlogs.logid)))
jl = logs.assign(logid=logs.logid.astype(int)).merge(rawlogs, on=["uid", "logid"], suffixes=("", "_raw"))
for f in ("type", "action", "timestamp", "ns", "title", "performer", "comment", "source"):
    n_bad = sum(clean(a) != clean(b) for a, b in zip(jl[f], jl[f + "_raw"]))
    ck.add(f"logs: {f} equals the raw value in every row ({n_bad} differ)", n_bad == 0)
n_bad = sum(json.loads(a or "{}") != b for a, b in zip(jl.params, jl.params_raw))
ck.add(f"logs: params (parsed JSON) equals the raw params in every row ({n_bad} differ)", n_bad == 0)
ck.add("profile.n_logs equals the count recomputed from raw logs",
       (P.n_logs == rawlogs.groupby("uid").size().reindex(P.index).fillna(0).astype(int)).all())

s6ab = pd.read_parquet(S6 / "abuse.parquet")
AB = ["uid", "abuse_id", "filter_id", "filter", "action", "result", "timestamp", "ns", "title"]
tpa = pd.DataFrame([(u, e.get("id"), str(e.get("filter_id") or ""), e.get("filter"), e.get("action"), e.get("result"), e.get("timestamp"),
                     e.get("ns"), e.get("title")) for u, r in topup.items() for e in r.get("abuse") or []], columns=AB)
tpa = tpa[tpa.timestamp < CUT]
ck.add("every abuse filter time has the ISO format", abuse.timestamp.str.match(ISO).all())
ck.add("no abuse filter hit on or after the cutoff", (abuse.timestamp < CUT).all())
ck.add("(uid, abuse_id) is unique in mentee_abuse", not abuse.duplicated(["uid", "abuse_id"]).any())
ck.add("abuse keys = s6 index keys + raw top-up keys before the cutoff",
       set(zip(abuse.uid, abuse.abuse_id.astype(int))) == set(zip(s6ab.uid, s6ab.abuse_id)) | set(zip(tpa.uid, tpa.abuse_id)))
ja = abuse[abuse.source == "topup"].assign(abuse_id=lambda d: d.abuse_id.astype(int)).merge(tpa.drop_duplicates(["uid", "abuse_id"]), on=["uid", "abuse_id"], suffixes=("", "_raw"))
ck.add("every top-up abuse row has its raw record", len(ja) == int((abuse.source == "topup").sum()))
for f in AB[2:]:
    n_bad = sum(clean(a) != clean(b) for a, b in zip(ja[f], ja[f + "_raw"]))
    ck.add(f"abuse (top-up rows): {f} equals the raw value ({n_bad} differ)", n_bad == 0)
n_ab_idx = s6ab.groupby("uid").size()
raw_ac = pd.Series(s6_abuse_count)
ck.add("every asker in the raw s6 abuse file has abuse_count >= hits in the s6 index", (n_ab_idx.reindex(raw_ac.index).fillna(0) <= raw_ac).all())
print(f"raw s6 abuse_count total {int(raw_ac.sum()):,} | s6 index {len(s6ab):,} | dropped after cutoff (manifest) "
      f"{mans['m3_index']['counts']['abuse']['dropped_after_cutoff']} ({time.time() - t:.0f} s)")
'''),
md("""
### 3.4 Account creation log (R2.2, R7.7)

Recomputed here with simple separate code: among the person's own logs, take the `newusers` entries whose
`params.userid` is the person's uid, and keep the earliest. If there is none, use the first entry with a record in
`creation.jsonl` (found by registration time). Otherwise there is no creation log.
"""),
code(r'''
own = {}
for r in logs[logs.type == "newusers"].sort_values(["timestamp", "logid"]).itertuples():
    if str(json.loads(r.params or "{}").get("userid")) == str(r.uid) and r.uid not in own:
        own[r.uid] = {"creation_action": r.action, "creation_ts": r.timestamp, "creation_performer": r.performer,
                      "creation_logid": int(r.logid), "creation_source": "log"}
def expected_creation(uid):
    if uid in own: return own[uid]
    for r in creation_raw.get(uid, []):
        if r.get("event"):
            e = r["event"]
            return {"creation_action": e.get("action"), "creation_ts": e.get("timestamp"), "creation_performer": e.get("user"),
                    "creation_logid": e.get("logid"), "creation_source": "lookup"}
    return {"creation_action": None, "creation_ts": None, "creation_performer": None, "creation_logid": None, "creation_source": "none"}
expc = {u: expected_creation(u) for u in P.index}
for f in ("creation_action", "creation_ts", "creation_performer", "creation_logid", "creation_source"):
    bad = [u for u in P.index if clean(P.at[u, f]) != clean(expc[u][f])]
    ck.add(f"profile.{f} equals the recomputed creation log for every asker (mismatches: {len(bad)})", not bad)
    if bad: print("   first mismatches:", [(u, clean(P.at[u, f]), expc[u][f]) for u in bad[:5]])
ck.add("creation_action is one of create, create2, byemail, autocreate, forcecreatelocal or empty",
       prof.creation_action.dropna().isin(["create", "create2", "byemail", "autocreate", "forcecreatelocal"]).all())
ck.add("every asker with no creation log was looked up in creation.jsonl and not found",
       all(u in creation_raw and not any(r.get("event") for r in creation_raw[u]) for u in prof[prof.creation_source == "none"].mentee_uid))
cr = prof[prof.creation_action == "create"]
same = (cr.creation_performer == cr.name_at_fetch) | (cr.creation_performer == cr.current_name)
print(f"'create' entries whose performer is the person's own name at fetch time or now: {same.mean():.2%} "
      f"(the rest are mostly renames: the log keeps the name used at registration)")
gap = (pd.to_datetime(prof.creation_ts, utc=True) - pd.to_datetime(prof.registration, utc=True)).dt.total_seconds()
prof["creation_gap_s"] = gap
print("seconds from registration to the creation log entry:", gap.describe().round(1).to_dict())
ck.add("for 99% of askers the creation log is within 2 s before to 10 s after registration (R7.7)", gap.between(-2, 10).mean() > .99)
'''),
md("""
### 3.5 Revert times (R7.8; the job is still running, so the files are partial)

The two raw files are combined, the full run winning over the sample run for the same edit (as in `revert_times.build`).
Each result must point to an edit of the same person that carries the `mw-reverted` tag, and a found revert must come
after the edit. An s6 edit got its tag when s6 was fetched (2026-04-23 to 24), so its revert must have happened
before 2026-04-25; a top-up edit was fetched in October, so its revert may fall after the cutoff (M7 ignores those).
"""),
code(r'''
rt = pd.concat([pd.DataFrame(read_jsonl(RAW3 / n)).assign(_file=n) for n in ("revert_times_sample.jsonl", "revert_times.jsonl")
                if (RAW3 / n).exists()], ignore_index=True).drop_duplicates("revid", keep="last")
rev_edits = edits[edits.tags.map(lambda x: "mw-reverted" in list(x))]
rj = rt.merge(edits[["revid", "uid", "timestamp", "source", "tags"]], on="revid", how="left", suffixes=("", "_edit"))
found = rj[rj.method.isin(["sha1", "undo_comment"])].copy()
found["lag_check"] = (pd.to_datetime(found.revert_ts, utc=True) - pd.to_datetime(found.ts, utc=True)).dt.total_seconds() / 3600
ck.add("each edit has at most one revert result after combining the files", rt.revid.is_unique)
ck.add("every revert result points to an edit in mentee_edits", rj.timestamp.notna().all())
ck.add("that edit carries the mw-reverted tag", rj.tags.dropna().map(lambda x: "mw-reverted" in list(x)).all())
ck.add("uid and edit time in the revert file equal the edit's", ((rj.uid == rj.uid_edit) & (rj.ts == rj.timestamp)).all())
ck.add("method is one of sha1, undo_comment, page_deleted, not_found, api_error",
       rt.method.isin(["sha1", "undo_comment", "page_deleted", "not_found", "api_error"]).all())
ck.add("every found revert has a revert revision and time", found.revert_revid.notna().all() and found.revert_ts.notna().all())
ck.add("the revert revision is later than the edit (larger revid)", (found.revert_revid > found.revid).all())
ck.add("the revert time is after the edit time", (found.revert_ts > found.ts).all())
ck.add("lag_hours equals revert time minus edit time (within 0.001 h)", (found.lag_check - found.lag_hours).abs().max() < 1e-3)
ck.add("s6 edits: the revert happened before s6 was fetched (before 2026-04-25)", (found[found.source == "s6"].revert_ts < "2026-04-25").all())
ck.add("lookups were made after the cutoff", (rt.fetched_at > CUT).all())

# coverage so far: candidates are mw-reverted edits within 30 days before the first question or before the cutoff
fq = pd.to_datetime(q.groupby("mentee_uid").q_ts.min(), utc=True)
ets = pd.to_datetime(rev_edits.timestamp, utc=True); w = pd.Timedelta(days=30)
near_q = (ets < rev_edits.uid.map(fq)) & (ets >= rev_edits.uid.map(fq) - w)
near_c = ets >= config.DATA_CUTOFF - w
cand = rev_edits[near_q | near_c]
cov_all = cand.revid.isin(rt.revid).mean()
print(f"\nmw-reverted edits {len(rev_edits):,} | candidates (30-day window) {len(cand):,} | with a result so far {int(cand.revid.isin(rt.revid).sum()):,} "
      f"({cov_all:.1%}) | results outside the candidate set (random sample of all mw-reverted edits) {int((~rt.revid.isin(cand.revid)).sum()):,}")
print("methods so far:", rt.method.value_counts().to_dict())
print("found reverts after the cutoff, by edit source:", found.assign(after=found.revert_ts >= CUT).groupby("source").after.sum().to_dict())
'''),
code(r'''
ck.report()
'''),
md("""
## 4. Comparison with the previous report and with independent data

### 4.1 Previous report

`results/m3_report.json` from the last run of this notebook, next to the values now. Differences are expected only
after a pipeline rebuild.
"""),
code(r'''
now_vals = {"n_askers": len(prof), "source": prof.source.value_counts().to_dict(), "n_edits": len(edits),
            "creation_action": {str(k): int(v) for k, v in prof.creation_action.value_counts(dropna=False).items()}}
if prev_report:
    display(pd.DataFrame([{"key": k, "previous": json.dumps(prev_report.get(k)), "now": json.dumps(v), "same": prev_report.get(k) == v}
                          for k, v in now_vals.items()]))
else:
    print("no previous report")
'''),
md("""
### 4.2 Independent edit data (val_mentee_contribs)

`val_mentee_contribs.jsonl` was fetched separately by the 2SLS pipeline in June 2026, by user name, with the userid
on every edit. For people in both, we compare the revids before the cutoff. The validation data holds at most 5,000
edits per person and some people are incomplete, so many edits are "only in our data"; that is not an error. What
matters is the other direction: **are the validation edits in our data?** (coverage = shared / validation, target above
99%). Edits only in the validation data are mostly from renames (fetched under an old name) or pages deleted later.
"""),
code(r'''
t = time.time()
mine = edits.groupby("uid").revid.apply(set).to_dict()
both = only_s6 = only_val = 0; n_users = 0
per_user = []
for line in open(config.EXTERNAL["val_mentee_contribs"], "rb"):
    r = orjson.loads(line)
    es = r.get("edits") or []
    if not es: continue
    uid = es[0].get("userid")
    if uid not in mine: continue
    val = {e["revid"] for e in es if e.get("timestamp", "") < CUT and e.get("userid") == uid}
    a = mine[uid]
    n_users += 1; both += len(a & val); only_s6 += len(a - val); only_val += len(val - a)
    per_user.append((uid, len(a), len(val), len(a & val)))
tot = both + only_s6 + only_val
recall = both / (both + only_val)
print(f"people in both {n_users:,}; same revids {both:,}, only in our data {only_s6:,}, only in validation data {only_val:,}")
print(f"share of validation edits found in our data: {recall:.4%} (target > 99%); agreement on the union: {both / tot:.4%}")
pu = pd.DataFrame(per_user, columns=["uid", "n_mine", "n_val", "n_both"])
pu["recall"] = pu.n_both / pu.n_val
print("people with coverage < 99%:", int((pu.recall < .99).sum()), f"({time.time() - t:.0f} s)")
display(pu.sort_values("recall").head(10))
'''),
md("""
### 4.3 Independent registration times (val_mentee_registration, June 2026, by uid)

Registration time never changes, so it must agree with the profile for everyone found in both.
"""),
code(r'''
vreg = {r["userid"]: r.get("registration") for r in iter_jsonl(config.EXTERNAL["val_mentee_registration"]) if r.get("userid")}
common = [u for u in P.index if u in vreg and vreg[u]]
dis = [(u, P.at[u, "registration"], vreg[u]) for u in common if P.at[u, "registration"] != vreg[u]]
print(f"askers in the June registration data: {len(common):,}; registration differs for {len(dis)}")
print("first differences (uid, ours, June):", dis[:10])
reg_agree = 1 - len(dis) / max(len(common), 1)
'''),
md("""
## 5. Tracer samples and random samples: raw records next to processed values

### 5.1 Fixed tracer people

For each fixed tracer asker: the profile, the creation log, the first 5 edits, and links to open the contributions
page and the creation log on Wikipedia.
"""),
code(r'''
tr = tracer_frame()
for uid in [int(u) for u in tr[tr.kind == "mentee"].uid.dropna()]:
    r = P.loc[uid]
    print(f"\n### {r.current_name} (uid {uid})  source={r.source} late_fetch={r.late_fetch}  registered {r.registration}")
    print(f"  creation: {r.creation_action} at {r.creation_ts} by {r.creation_performer} (logid {r.creation_logid}, {r.creation_source})")
    print("  contributions:", contribs(r.current_name))
    if pd.notna(r.creation_logid): print("  creation log:", logid(r.creation_logid))
    e = edits[edits.uid == uid].sort_values("timestamp")
    print(f"  edits {len(e)} (articles, ns 0: {int((e.ns == 0).sum())}); logs {r.n_logs}; abuse {r.n_abuse}")
    display(e.head(5)[["timestamp", "ns", "revid", "sizediff", "tags"]])
'''),
md("""
### Helper for the random examples

`person_check(uid)` reads the person's **raw** records (the raw s6 lines and/or the raw top-up record) and
recomputes, with its own simple code:

* the profile fields, next to the stored profile;
* the edits: raw records, those on or after the cutoff (must be dropped), duplicate revids (must be removed), and then
  count, first and last time, and the revid set, next to the merged table; then every field of every edit;
* the number of log entries and abuse filter hits;
* the account creation log record, shown raw, next to the `creation_*` fields.
"""),
code(r'''
FULL = ds.dataset(M3 / "mentee_edits").to_table(filter=ds.field("uid").isin(sorted(SAMPLE_UIDS))).to_pandas()
ECOLS = ["parentid", "ns", "title", "timestamp", "size", "sizediff", "tags", "minor", "new", "comment"]
def fingerprint(ids):
    ids = sorted(int(i) for i in ids)
    return f"{len(ids)} ids, sum {sum(ids)}"
random_results = []

def person_check(uid):
    p = P.loc[uid]; tp = topup.get(uid)
    print("\n" + "=" * 110)
    print(f"uid {uid}  {p.current_name}  source={p.source}  {contribs(p.current_name)}")
    # profile
    if tp is not None:
        u = tp["user"]; print(f"raw top-up user record ({tp['_file']}, fetched {tp['fetched_at'][:19]}):", json.dumps(u, ensure_ascii=False)[:250])
        hand_p = {"name_at_fetch": u.get("name"), "registration": u.get("registration"), "editcount": u.get("editcount"),
                  "blocked": "blockid" in u, "groups": list(u.get("groups") or []), "late_fetch": True}
    else:
        r = s6_prof_raw[s6_name[uid]]
        print("raw s6 profile line:", json.dumps({k: r.get(k) for k in ("username", "userid", "registration", "editcount", "blocked", "groups")}, ensure_ascii=False))
        hand_p = {"name_at_fetch": r["username"], "registration": r.get("registration"), "editcount": r.get("editcount"),
                  "blocked": bool(r.get("blocked")), "groups": list(r.get("groups") or []), "late_fetch": False}
    if not hand_p["registration"]: hand_p["registration"] = (reg_now.get(str(uid)) or {}).get("registration")
    hand_p["current_name"] = (reg_now.get(str(uid)) or {}).get("name")
    ok = compare({k: clean(v) for k, v in hand_p.items()}, {k: clean(p[k]) for k in hand_p}, "profile: raw vs processed")
    if uid in stale: print(f"stale.json: {stale[uid]}  (stale_edits.jsonl holds {len(stale_edits.get(uid, []))} edits)")

    # edits
    raw = [(e, "s6 raw") for e in s6_full.get(uid, [])] + [(e, tp["_file"]) for e in (tp or {}).get("contribs") or []]
    after = [e for e, _ in raw if e["timestamp"] >= CUT]
    uniq = {}
    for e, src in raw:
        if e["timestamp"] < CUT and e["revid"] not in uniq: uniq[e["revid"]] = (e, src)
    n_dup = sum(1 for e, _ in raw if e["timestamp"] < CUT) - len(uniq)
    m = FULL[FULL.uid == uid].set_index("revid")
    print(f"raw edit records {len(raw)} (s6 {len(s6_full.get(uid, []))}, top-up {len(raw) - len(s6_full.get(uid, []))}); "
          f"on or after the cutoff {len(after)}; duplicate revids {n_dup}")
    ts_raw = [e["timestamp"] for e, _ in uniq.values()]
    hand_e = {"n_edits": len(uniq), "rows in edit table": len(uniq), "first edit": min(ts_raw, default=None), "last edit": max(ts_raw, default=None),
              "revid set": fingerprint(uniq), "edits on or after cutoff": 0}
    data_e = {"n_edits": int(p.n_edits), "rows in edit table": len(m), "first edit": m.timestamp.min() if len(m) else None,
              "last edit": m.timestamp.max() if len(m) else None, "revid set": fingerprint(m.index), "edits on or after cutoff": int((m.timestamp >= CUT).sum())}
    ok &= compare(hand_e, data_e, "edits: raw vs merged")
    nbad = Counter()
    for rid, (e, src) in uniq.items():
        if rid not in m.index: continue
        row = m.loc[rid]
        rawv = {"parentid": e.get("parentid"), "ns": e.get("ns"), "title": e.get("title"), "timestamp": e.get("timestamp"), "size": e.get("size"),
                "sizediff": e.get("sizediff"), "tags": list(e.get("tags") or []), "minor": bool(e.get("minor")), "new": bool(e.get("new")), "comment": e.get("comment")}
        for f in ECOLS:
            if clean(rawv[f]) != clean(row[f]): nbad[f] += 1
        if row["source"] != ("s6" if src == "s6 raw" else "topup"): nbad["source"] += 1
    print("edit fields that differ from raw (field: number of edits):", dict(nbad) or "none")
    ok &= not nbad
    show = sorted(uniq.values(), key=lambda x: x[0]["timestamp"])
    show = show[:2] + show[-2:] if len(show) > 4 else show
    print(pd.DataFrame([{"revid": e["revid"], "raw file": src, "raw time": e["timestamp"], "merged time": m.at[e["revid"], "timestamp"],
                         "raw ns": e.get("ns"), "merged ns": m.at[e["revid"], "ns"], "raw sizediff": e.get("sizediff"),
                         "merged sizediff": m.at[e["revid"], "sizediff"], "merged source": m.at[e["revid"], "source"], "link": diff(e["revid"])}
                        for e, src in show]).to_string(index=False))

    # logs and abuse filter hits
    rl = [(e[1], e[4]) for e in s6_log_rows if e[0] == uid] + [(e.get("logid"), e.get("timestamp")) for e in (tp or {}).get("logs") or []]
    ra = [(e.get("id"), e.get("timestamp")) for e in s6_abuse_full.get(uid, [])] + [(e.get("id"), e.get("timestamp")) for e in (tp or {}).get("abuse") or []]
    hand_c = {"n_logs": len({i for i, ts in rl if ts < CUT}), "n_abuse": len({i for i, ts in ra if ts < CUT})}
    ok &= compare(hand_c, {k: int(p[k]) for k in hand_c}, f"logs and abuse: raw records (logs {len(rl)}, abuse {len(ra)}, with duplicates and post-cutoff) vs counts")
    ab_m = abuse[abuse.uid == uid].set_index("abuse_id")
    raw_ab = {e.get("id"): e for e in s6_abuse_full.get(uid, []) + list((tp or {}).get("abuse") or []) if e.get("timestamp") < CUT}
    n_ab_bad = sum(clean(ab_m.at[i, f]) != clean(str(e.get("filter_id") or "") if f == "filter_id" else e.get(f))
                   for i, e in raw_ab.items() if i in ab_m.index for f in ("filter_id", "filter", "action", "result", "timestamp", "ns", "title"))
    print("abuse filter fields that differ from raw:", n_ab_bad); ok &= n_ab_bad == 0

    # account creation log
    cands = [dict(logid=e[1], type=e[2], action=e[3], timestamp=e[4], title=e[6], user=e[7], params=e[8], file="s6 raw") for e in s6_log_rows if e[0] == uid]
    cands += [dict(logid=e.get("logid"), type=e.get("type"), action=e.get("action"), timestamp=e.get("timestamp"), title=e.get("title"),
                   user=e.get("user"), params=e.get("params") or {}, file=tp["_file"]) for e in (tp or {}).get("logs") or []]
    nu = sorted([c for c in cands if c["type"] == "newusers" and str(c["params"].get("userid")) == str(uid) and c["timestamp"] < CUT],
                key=lambda c: (c["timestamp"], c["logid"]))
    if nu:
        c = nu[0]; print("raw newusers log record:", json.dumps(c, ensure_ascii=False)[:300])
        hand_k = {"creation_action": c["action"], "creation_ts": c["timestamp"], "creation_performer": c["user"], "creation_logid": c["logid"], "creation_source": "log"}
    else:
        ev = next((r["event"] for r in creation_raw.get(uid, []) if r.get("event")), None)
        print("no own newusers entry in the raw logs; creation.jsonl:", json.dumps(creation_raw.get(uid), ensure_ascii=False)[:300])
        hand_k = ({"creation_action": ev.get("action"), "creation_ts": ev.get("timestamp"), "creation_performer": ev.get("user"),
                   "creation_logid": ev.get("logid"), "creation_source": "lookup"} if ev else
                  {"creation_action": None, "creation_ts": None, "creation_performer": None, "creation_logid": None, "creation_source": "none"})
    ok &= compare(hand_k, {k: clean(p[k]) for k in hand_k}, "account creation: raw log record vs profile")
    if hand_k["creation_logid"]: print("  open the log entry:", logid(hand_k["creation_logid"]), f"| registration {p.registration}")
    random_results.append((p.source, uid, ok))
    return ok
'''),
md("""
### 5.2 Random s6 askers

People whose data comes only from the April s6 files.
"""),
code(r'''
for uid in SAMPLE["s6"]: person_check(uid)
'''),
md("""
### 5.3 Random s6+topup askers

People whose s6 edit history was incomplete (`stale.json`). Expected: merged edits = s6 raw edits + top-up edits,
each revision once, and all of them before the cutoff.
"""),
code(r'''
for uid in SAMPLE["s6+topup"]: person_check(uid)
'''),
md("""
### 5.4 Random topup askers

People that s6 does not have at all; everything comes from the October top-up fetched by uid.
"""),
code(r'''
for uid in SAMPLE["topup"]: person_check(uid)
print("\nrandom people: all fields match for", sum(ok for *_, ok in random_results), "of", len(random_results),
      "" if all(ok for *_, ok in random_results) else "-> MISMATCH, see above")
'''),
md("""
### 5.5 Random reverted edits with a revert time

Open the two `diff` links: the first shows the newcomer's edit, the second the revision that undid it. The revert
revision should restore the page (sha1 method) or be an undo whose summary names the edit (undo_comment).
"""),
code(r'''
rs = pick(found, N_RANDOM, SEED)
title = ds.dataset(M3 / "mentee_edits").to_table(filter=ds.field("revid").isin([int(x) for x in rs.revid]), columns=["revid", "title", "tags"]).to_pandas().set_index("revid")
rev_ok = []
for r in rs.itertuples():
    print(f"\nedit {r.revid} by uid {r.uid} on '{title.at[r.revid, 'title']}' at {r.ts} (source {r.source}); tags {list(title.at[r.revid, 'tags'])}")
    print(f"  edit:   {diff(r.revid)}")
    print(f"  revert: {diff(int(r.revert_revid))}  ({oldid(int(r.revert_revid))}) at {r.revert_ts}, method {r.method}")
    print(f"  revert after the edit: {r.revert_ts > r.ts} | revert before the cutoff: {r.revert_ts < CUT}"
          + ("" if r.revert_ts < CUT else " (allowed in this file; M7 does not count it)"))
    hand = {"edit time": edits.loc[edits.revid == r.revid, "timestamp"].iloc[0], "uid": int(edits.loc[edits.revid == r.revid, "uid"].iloc[0]),
            "lag_hours": round((pd.Timestamp(r.revert_ts) - pd.Timestamp(r.ts)).total_seconds() / 3600, 3), "revert after edit": True}
    rev_ok.append(compare(hand, {"edit time": r.ts, "uid": int(r.uid), "lag_hours": r.lag_hours, "revert after edit": r.revert_ts > r.ts}))
'''),
md("""
## 6. Edge-case lists

Issue files are written by each step. Note: a step that skips work already done (for example `collect_topup`,
which resumes) writes an empty `summary.json`, so older issue files can be leftovers from an earlier run. The table
marks files older than their summary; their content is still shown when it matches the current data.
"""),
code(r'''
for name in ("m3_index", "m3_collect", "m3"):
    d = config.INTERIM / name / "_issues"
    summ = json.loads((d / "summary.json").read_text())
    print(f"\n[{name}] summary.json: {({k: v['n'] for k, v in summ.items()})}")
    for f in sorted(d.glob("*.jsonl")):
        n = sum(1 for _ in open(f))
        older = f.stat().st_mtime < (d / "summary.json").stat().st_mtime - 60
        print(f"  {f.name:35s} {n:6,} rows  {'(older than summary.json: from an earlier run)' if older else ''}")
nf = read_jsonl(config.INTERIM / "m3_collect" / "_issues" / "creation_log_not_found.jsonl")
print(f"\ncreation_log_not_found: {len(nf)} rows; askers with creation_source='none' now: {int((prof.creation_source == 'none').sum())}; "
      f"same set: {set(r['ids']['uid'] for r in nf) == set(prof[prof.creation_source == 'none'].mentee_uid)}")
print("registration years of askers without a creation log (logs started in 2005):",
      pd.to_datetime(prof[prof.creation_source == "none"].registration, utc=True).dt.year.value_counts().sort_index().to_dict())
wa = read_jsonl(config.INTERIM / "m3" / "_issues" / "asker_without_activity.jsonl")
print(f"asker_without_activity.jsonl: {len(wa)} rows, of which now without data: {sum(P.at[r['ids']['uid'], 'source'] == 'none' for r in wa if r['ids']['uid'] in P.index)}")
'''),
code(r'''
print("creation log far from registration (outside -2 s to +10 s):")
far = prof[prof.creation_gap_s.notna() & ~prof.creation_gap_s.between(-2, 10)]
display(far[["mentee_uid", "current_name", "registration", "creation_ts", "creation_gap_s", "creation_action", "creation_source"]].sort_values("creation_gap_s").head(15))
print("\nprofile edit counter lower than the edits we hold (editcount counts deleted edits too, so it is usually higher):",
      int((prof.editcount < prof.n_edits).sum()), "askers; by source", prof[prof.editcount < prof.n_edits].source.value_counts().to_dict())
low = prof[prof.editcount < prof.n_edits]
display(low.assign(diff=low.n_edits - low.editcount).sort_values("diff", ascending=False).head(8)[["mentee_uid", "current_name", "source", "registration", "editcount", "n_edits", "diff"]])
print("share of these registered before 2010 (old edit counters are known to be incomplete):", round(float((low.registration < "2010").mean()), 3))
print("\nstale.json reasons:", Counter(v["reason"] for v in stale.values()))
print("people found incomplete because s6 stopped at exactly 500 edits:", [u for u, v in stale.items() if v["reason"] == "capped_500"])
print("\nrevert lookups without a time (method, examples with links):")
for meth in ("page_deleted", "not_found", "api_error"):
    x = rt[rt.method == meth]
    print(f"  {meth}: {len(x)}", [diff(i) for i in pick(x, 3, SEED).revid])
'''),
md("""
## 7. Distributions

Account creation action (D10 uses it); registration year by source; and s6 versus top-up people, to see whether the
top-up people differ in a systematic way. Then the revert lag found so far.
"""),
code(r'''
print("account creation action:\n" + prof.creation_action.value_counts(dropna=False).to_string())
print("\nwhere the creation log came from:", prof.creation_source.value_counts().to_dict())
prof["reg_year"] = pd.to_datetime(prof.registration, utc=True, errors="coerce").dt.year
display(pd.crosstab(prof.reg_year, prof.source))
cmp = prof.groupby("source").agg(n=("mentee_uid", "size"), edits_median=("n_edits", "median"),
        edits_mean=("n_edits", "mean"), abuse_any=("n_abuse", lambda s: (s > 0).mean()), logs_median=("n_logs", "median"))
display(cmp)
print("\nedits per year by source:")
display(pd.crosstab(edits.timestamp.str[:4], edits.source))
print("\nhours from edit to revert (found so far):", found.lag_hours.quantile([.5, .9, .95, .99]).round(2).to_dict(),
      "| share over 30 days:", round(float((found.lag_hours > 24 * 30).mean()), 4))
'''),
md("## 8. Summary (written to results/m3_report.json, used by M9)"),
code(r'''
report = {"n_askers": len(prof), "source": prof.source.value_counts().to_dict(), "n_edits": len(edits),
          "creation_action": {str(k): int(v) for k, v in prof.creation_action.value_counts(dropna=False).items()},
          "val_recall": round(recall, 6), "val_union_agreement": round(both / tot, 6), "val_users": n_users}
(config.RESULTS / "m3_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps(report, indent=1, ensure_ascii=False))
n_fail = sum(not ok for _, ok in ck.items)
print(f"\ninvariants: {len(ck.items) - n_fail} / {len(ck.items)} OK | random people matching raw: "
      f"{sum(ok for *_, ok in random_results)} / {len(random_results)} | random reverts: {sum(rev_ok)} / {len(rev_ok)} | "
      f"registration agrees with June data: {reg_agree:.4%} | revert-time coverage of candidates so far: {cov_all:.1%}")
print(f"notebook run time {time.time() - T0:.0f} s")
'''),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "m3_check.ipynb"
    write_notebook(CELLS, out)
    print("wrote", out)
    if "--run" in sys.argv:
        run_as_script(out, cwd=out.parent.parent)
        print("\nnotebook ran end to end")


if __name__ == "__main__":
    main()
