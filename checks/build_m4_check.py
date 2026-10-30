"""
build_m4_check.py: writes checks/m4_check.ipynb, the check notebook for M4 (assignment eligibility)

The notebook follows the fixed 8 sections of PLAN.md section 12:
    1 module description and data dictionary   2 files and manifest   3 invariants (every derived field
    recomputed with separate simple code, for all askers)   4 comparison with the previous report, an
    independent claim log and the paper sample   5 tracer claim logs and random samples (every decision input next
    to the decision)   6 edge-case lists   7 distributions and diagnostics   8 summary (results/m4_report.json)

How to run:
    python checks/build_m4_check.py          # writes the notebook
    python checks/build_m4_check.py --run    # writes it and runs it from top to bottom (one to three minutes)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks._nbgen import code, md, run_as_script, write_notebook  # noqa: E402

CELLS = [
md("""
# M4 assignment eligibility: check notebook

## Plain-language guide

**Why the study needs this module.** The paper compares newcomers whose assigned mentor replied with newcomers whose
mentor did not. That comparison is only fair if the mentor was handed out by chance. On English Wikipedia the
software normally picks a mentor at random when a person creates an account, but there are exceptions: some accounts
were made before the mentorship program existed, some were made by another person or copied from another wiki, and
sometimes a mentor picks a newcomer on purpose ("claims" them). M4 finds, for every newcomer who asked their mentor a
question (an "asker"), whether their mentor was really assigned at random. Only those people (`eligible = True`)
enter the main analysis.

**What raw data it uses.** M4 collects one new raw file: the mentor claim log, every entry of the "claim mentee"
action in Wikipedia's public log (Special:Log, read through the MediaWiki API `list=logevents`). It was downloaded on
2026-10-06 and has 1,055 entries since the program started in 2021; the 815 entries before the data cutoff
(2026-04-20T14:00Z) are used. Each entry says who claimed which account, from which previous mentor, when, and the
comment the mentor typed. Everything else comes from earlier modules: the askers and their first question (M2), each
asker's registration time and account creation log entry (M3), and the timeline of the mentor list (M1).

**What it does, step by step.**

1. Takes every asker from M2 and finds their first question.
2. Cleans the claim log: keeps entries before the cutoff, turns each claimed user name into a user ID (following
   renames), and labels each claim as `bulk` (an automatic random move after a mentor quit), `self_transfer` (a
   mentor moving people between two of their own accounts) or `manual` (a mentor chose this person). It also marks
   claims whose comment says the account is the mentor's own alternate or test account.
3. For each asker, checks five exclusion rules in a fixed order (listed below). The first rule that holds is the
   reason for exclusion; a person with no reason is eligible.
4. Adds two extra flags for robustness checks and four diagnostic flags about the mentor list at registration time.

**What it outputs (this run).**

* `data/interim/m4/eligibility.parquet`: one row per asker, **37,095 rows**. Key columns: the asker's user ID, the
  first question and its time, registration time, how the account was created, `eligible` (True or False), the
  exclusion reason(s), and the robustness and diagnostic flags. 36,094 askers are eligible.
* `data/interim/m4/claims.parquet`: one row per claim log entry before the cutoff, **815 rows**. Key columns: when,
  who claimed whom, the previous mentor, the comment, the label (`kind`: 450 manual, 358 bulk, 7 self transfer) and
  whether it is an alternate or test account.
* `results/m4_report.json`: the summary counts written in section 8.

**How to read this notebook.** The outputs below are saved from the run on the final data (cutoff
2026-04-20T14:00Z); you do not need to run anything. A line `[OK ]` means a check passed and `[FAIL]` means it did
not (there are no failures in this run). An **invariant** is a rule that must hold for every row; we test it by
recomputing the value with separate, simple code written in this notebook and counting rows that disagree. A
**tracer** is a hand-picked real log entry whose correct answer we know, followed through the processing. A **random
sample** shows a few randomly drawn people or log entries with all raw inputs next to the decision, plus links to
Wikipedia so anyone can check them by eye.

## Technical description

**What the module does.** `pipeline/m4_eligibility/build_eligibility.py` decides, for every asker, whether their
mentor was **assigned at random when they registered**. Only those people (`eligible = True`) enter the main analysis.
The conditions are checked in this order; the first one that holds is the `exclude_reason`, and all that hold are
kept in `exclude_reasons`:

1. `temporary_account`: the user name starts with `~2` (R2.2).
2. How the account was created (R2.2, decision D10): only `create` (the person registered themselves) is kept.
   `autocreate` -> `autocreated_account`; `create2`, `byemail` -> `created_by_other`;
   `forcecreatelocal` -> `forced_local_account`; no creation log -> `no_creation_log`.
3. `registered_before_mentorship`: registered before 2021-05-04 (R2.4, D10).
4. `alt_or_test_account`: a claim log says the account is a mentor's own alternate or test account (R3.4, R3.7), at any time.
5. `manual_claim_before_question`: a mentor claimed the person by hand before the first question (R3.1, R3.3).
   A claim is "manual" unless its comment matches a bulk transfer template (`bulk`, a random transfer after a mentor
   left) or the claimer and previous mentor are two accounts of the same person (`self_transfer`, R3.6).

It also stores `eligible_unrestricted` (eligible without the D10 rules 2 and 3, for robustness checks),
`claimed_after_question` (for the R3.3 robustness check) and diagnostics about the mentor's roster state at registration.

**Paper numbers that depend on M4.** The sample size of the main analysis and the sample construction table
(Appendix Table A), the quasi-random assignment argument (Sec 3), and the robustness checks that use
`eligible_unrestricted` and `claimed_after_question`.

**How to use this notebook.** Run it from top to bottom (about one to three minutes). Section 3 re-derives every column with
separate simple code written in this notebook and counts mismatches over all askers. Section 5 shows, for random
people, every input of the decision next to the decision, with links to the logs on Wikipedia. Change `SEED` in the
setup cell to draw new random people.
"""),
md("""
**What this checks.** This setup cell loads the M4 outputs and the inputs from M1, M2 and M3, and prints the basic
sizes. It is not a test by itself, but every later check uses these tables.
"""),
code(r'''
# ── 1. Setup ──────────────────────────────────────────────────────────────
SEED = 0        # change SEED (any whole number) to draw a fresh random sample in section 5
N_RANDOM = 5    # how many random examples to show per group in section 5

import ast, json, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.common.io import read_jsonl, iter_jsonl
from checks._show import Checks, compare, pick, oldid, diff, contribs, logid
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
T0 = time.time()
M4 = config.INTERIM / "m4"
CUT = config.DATA_CUTOFF_ISO
START = "2021-05-04T00:00:00Z"          # config.MENTORSHIP_START as ISO text
FORCE = "2023-02-02T00:00:00Z"          # config.FORCEMENTOR_LIVE as ISO text
e = pd.read_parquet(M4 / "eligibility.parquet")
c = pd.read_parquet(M4 / "claims.parquet")
E = e.set_index("mentee_uid")
q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["question_revid", "mentee_uid", "mentee_name", "q_ts", "primary_mentor_uid"])
q = q[q.mentee_uid.notna()].astype({"mentee_uid": "int64"})
prof = pd.read_parquet(config.INTERIM / "m3" / "mentee_profile.parquet").set_index("mentee_uid")
mlogs = pd.read_parquet(config.INTERIM / "m3" / "mentee_logs.parquet", filters=[("type", "==", "newusers")])
periods = pd.read_parquet(config.INTERIM / "m1" / "mentor_periods.parquet")
raw_claims = json.loads(config.RAW_FILES["claim_log"].read_text())
prev_report = json.loads((config.RESULTS / "m4_report.json").read_text()) if (config.RESULTS / "m4_report.json").exists() else None
ck = Checks()

def clean(v):
    """Turns pandas and numpy values into plain Python values so two values can be compared."""
    if v is None or v is pd.NA: return None
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, (np.ndarray, list, tuple)): return [str(x) for x in v]
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, np.integer): return int(v)
    return v
print(f"askers {len(e):,} | eligible {int(e.eligible.sum()):,} | claim log rows before the cutoff {len(c):,} | raw claim log entries {len(raw_claims):,}")
'''),
md("""
**What the output shows.** In the run on the final data (cutoff 2026-04-20T14:00Z) there are 37,095 askers, of whom
36,094 (97.3%) are eligible, so M4 removes 1,001 people. The raw claim log has 1,055 entries; 815 are before the
cutoff and are used. The other 240 happened after the cutoff and are ignored on purpose, because the study only
looks at events up to the cutoff.
"""),
md("""
### Data dictionary

**What this checks.** The table explains, in words, every column of the two output files and names the cell that
verifies it ("3.x" are checks over all rows with separate code, "5.x" are random raw-versus-processed examples). The
cell also confirms that the dictionary lists exactly the columns in each file, so no column goes unexplained.
"""),
code(r'''
DICT = {}
DICT["eligibility"] = {
    "mentee_uid": ("user ID of the asker (primary key)", "3.1"),
    "first_question_revid": ("revision ID of the asker's first question (earliest q_ts in M2)", "3.1 recomputed; 5.2 diff link"),
    "first_q_ts": ("time of the first question", "3.1; 5.2"),
    "registered_at": ("registration time from the M3 profile", "3.1; 5.2"),
    "creation_action": ("account creation action from the M3 profile (create = registered themselves)", "3.1; 5.2 with the raw log record"),
    "eligible": ("True when no exclusion reason holds: the mentor was assigned at random at registration", "3.3 re-derived for all; 5.2"),
    "exclude_reason": ("the first reason that holds, in the documented order", "3.3; 5.2"),
    "exclude_reasons": ("all reasons that hold, in order", "3.3; 5.2"),
    "eligible_unrestricted": ("eligible when the D10 rules (creation action, registration date) are ignored", "3.3; 5.2"),
    "claimed_after_question": ("a manual claim happened at or after the first question", "3.3; 5.2"),
    "n_claims": ("number of claim log rows of this person (any kind)", "3.3; 5.2"),
    "n_manual_claims": ("number of those that are manual", "3.3; 5.2"),
    "primary_mentor_uid": ("the mentor the first question went to (from M2)", "3.1"),
    "mentor_on_roster_at_registration": ("that mentor was on the mentor list at the registration time", "3.4 recomputed from M1 periods"),
    "mentor_in_auto_pool_at_registration": ("that mentor was in the automatic assignment pool at registration", "3.4"),
    "assigned_later": ("registration known but the mentor was not on the list then (the mentor came later)", "3.4"),
    "possible_forcementor": ("mentor on the list but not in the auto pool at registration, after 2023-02-02, and no claim by that mentor explains it", "3.4; 7"),
}
DICT["claims"] = {
    "logid": ("log ID of the claim entry", "3.2 equals the raw log; 5.3 link"),
    "ts": ("time of the claim (before the cutoff)", "3.2"),
    "action": ("claimmentee or claimmentee-no-previous-mentor", "3.2"),
    "mentee_name": ("claimed account, from the log title User:Name", "3.2; 5.3"),
    "claimer": ("mentor who made the claim", "3.2; 5.3"),
    "claimer_uid": ("user ID of the claimer", "3.2"),
    "previous_mentor": ("mentor before the claim (empty for claimmentee-no-previous-mentor)", "3.2; 5.3"),
    "comment": ("comment typed by the claimer", "3.2; 5.3"),
    "kind": ("bulk (matches a bulk transfer template), self_transfer (same person's two accounts) or manual", "3.2 re-derived for all; 5.3"),
    "alt_or_test": ("the claimed account is the claimer's own alternate or test account (R3.4, R3.7)", "3.2 re-derived; 5.3; 6"),
    "mentee_uid": ("user ID of the claimed account (name resolved to uid, R3.5)", "3.2 against known names; 6"),
}
def show_dict(name, cols):
    t = pd.DataFrame([{"column": k, "meaning": v[0], "checked in": v[1]} for k, v in DICT[name].items()])
    print(f"\n{name}"); print(t.to_string(index=False, max_colwidth=110))
    ck.add(f"data dictionary of {name} lists exactly the columns in the file", set(DICT[name]) == set(cols))
show_dict("eligibility", e.columns)
show_dict("claims", c.columns)
'''),
md("""
**What the output shows.** Both dictionaries match their files exactly (2 OK lines). The eligibility file has 17
columns and the claims file has 11.
"""),
md("""
## 2. Files and manifest

**What this checks.** When M4 runs it writes a "manifest", a small record of which files it read and wrote, when, and
the main counts. We check that the manifest uses the same cutoff as the rest of the study, that its counts equal the
actual tables, and that M4 ran after M1, M2 and M3 (so it used their latest outputs, not stale ones).
"""),
code(r'''
man = json.loads((M4 / "manifest.json").read_text())
print("finished:", man["finished_at"], "| cutoff:", man["data_cutoff"], "| elapsed:", man["elapsed_sec"], "s")
display(pd.DataFrame(man["inputs"] + man["outputs"])[[x for x in ("path", "exists", "bytes", "mtime") if x in pd.DataFrame(man["inputs"] + man["outputs"]).columns]])
print(json.dumps(man["counts"], ensure_ascii=False, indent=1))
ck.add("manifest uses the cutoff in config.py", man["data_cutoff"] == CUT)
ck.add("manifest counts equal the table (askers, eligible, eligible_unrestricted)",
       (man["counts"]["askers"], man["counts"]["eligible"], man["counts"]["eligible_unrestricted"]) == (len(e), int(e.eligible.sum()), int(e.eligible_unrestricted.sum())))
ck.add("manifest exclusion counts equal the table", man["counts"]["exclude_reason"] == e.exclude_reason.value_counts().to_dict())
up = {m: json.loads((config.INTERIM / m / "manifest.json").read_text())["finished_at"] for m in ("m1", "m2", "m3")}
print("upstream finished:", up)
ck.add("M4 ran after M1, M2 and M3 (it uses their current outputs)", all(t <= man["finished_at"] for t in up.values()))
'''),
md("""
**What the output shows.** All 4 checks pass. The cutoff is 2026-04-20T14:00:00Z and M4 finished after M1, M2 and M3.
The exclusion counts are: 452 accounts created by another person, 225 automatically created (copied from another
wiki), 218 registered before the program started on 2021-05-04, 82 with no creation log, 19 claimed by hand before
their first question, 4 alternate or test accounts, and 1 forced local account. Zero temporary accounts appear, which
is expected because temporary accounts do not get a mentor.
"""),
md("""
## 3. Invariants (all should be OK)

### 3.1 Inputs copied from M2 and M3

**What this checks.** M4 copies some facts from earlier modules: each asker's first question (from M2) and their
registration time and creation action (from M3). We check that there is exactly one row per asker, that the set of
askers equals M2's, and that every copied value equals its source. A wrong copy here would silently put the wrong
people in or out of the sample.
"""),
code(r'''
first = q.sort_values(["q_ts", "question_revid"]).groupby("mentee_uid").head(1).set_index("mentee_uid")
ck.add("one row per asker, and the uids equal the M2 askers", e.mentee_uid.is_unique and set(e.mentee_uid) == set(q.mentee_uid))
ck.add("first_question_revid is the asker's earliest question in M2", (E.first_question_revid == first.question_revid.reindex(E.index)).all())
ck.add("first_q_ts is the time of that question", (E.first_q_ts == first.q_ts.reindex(E.index)).all())
ck.add("primary_mentor_uid is the primary mentor of that question", (E.primary_mentor_uid.astype("float") == first.primary_mentor_uid.reindex(E.index).astype("float")).all())
for a, b in (("registered_at", "registration"), ("creation_action", "creation_action")):
    n_bad = sum(clean(x) != clean(y) for x, y in zip(E[a], prof[b].reindex(E.index)))
    ck.add(f"{a} equals the M3 profile ({n_bad} differ)", n_bad == 0)
'''),
md("""
**What the output shows.** All 6 checks pass with 0 differences for all 37,095 askers.
"""),
md("""
### 3.2 Claim log rows: copied from the raw log, and classified again with separate code

**What this checks.** First, that every claim row before the cutoff in M4's table is exactly the raw log entry
(same IDs, times, names, comments). Second, that the labels (`bulk`, `self_transfer`, `manual`, and alternate or
test account) come out the same when we apply our own simple rules, written below, to every one of the 815 rows.
Third, that claimed user names were turned into the right user IDs. These labels decide who counts as "chosen on
purpose by a mentor", so errors here would bias the sample.

Our own simple rules (written here, not imported from the pipeline):

* **bulk**: the comment is `<name> retired from mentorship.` or `<name> removed <name> from mentorship` (the two
  GrowthExperiments templates, with any names).
* **self_transfer**: the "core" of the claimer's name and of the previous mentor's name are equal, or one contains
  the other and both have at least 4 characters. The core is the name in lower case without text in brackets,
  spaces and punctuation (so `LuniZunie` and `LuniZunie-Mentor` share the core `lunizunie`).
* **manual**: everything else.
* **alt_or_test**: the comment contains a word alt, alts, alternate, alternative, sock, socks, bot, or a word starting
  with test; **and** there is evidence it is the claimer's own account: same core name, one core inside the other
  (both at least 4 characters), or the comment contains my, mine, own, myself, personal or self.
"""),
code(r'''
def norm_name(n):
    """Simple user name form: underscores to spaces, trimmed, first letter upper case."""
    if not n: return None
    n = re.sub(r"\s+", " ", str(n).replace("_", " ")).strip()
    return n[:1].upper() + n[1:] if n else None
def core(n):
    n = re.sub(r"\(.*?\)", "", (norm_name(n) or "")).lower()
    return "".join(ch for ch in n if ch.isalnum())
BULK = [re.compile(r"^.+? retired from mentorship\.\.?$", re.I), re.compile(r"^.+? removed .+? from mentorship\.?$", re.I)]
ALT = re.compile(r"\b(alt|alts|alternate|alternative|sock|socks|bot|test\w*)\b", re.I)
OWN = re.compile(r"\b(my|mine|own|myself|personal|self)\b", re.I)
def my_kind(comment, claimer, prev):
    """Returns (kind, reason in words)."""
    cm = (comment or "").strip()
    for b in BULK:
        if b.match(cm): return "bulk", f"comment matches the bulk template {b.pattern!r}"
    a, p = core(claimer), core(prev)
    if a and p and (a == p or (min(len(a), len(p)) >= 4 and (a in p or p in a))):
        return "self_transfer", f"claimer core {a!r} and previous mentor core {p!r} are the same person"
    return "manual", "no bulk template, and claimer and previous mentor are different people"
def my_alt(comment, mentee, claimer):
    cm = comment or ""
    w = ALT.search(cm)
    if not w: return False, "comment has no alt/sock/bot/test word"
    m, k = core(mentee), core(claimer)
    if m and m == k: return True, f"word {w.group(0)!r}; claimer claimed their own account"
    if len(m) >= 4 and len(k) >= 4 and (m in k or k in m): return True, f"word {w.group(0)!r}; name cores {m!r} and {k!r} overlap"
    o = OWN.search(cm)
    if o: return True, f"word {w.group(0)!r}; comment says {o.group(0)!r}"
    return False, f"word {w.group(0)!r} but no evidence it is the claimer's own account"

raw_rows = []
for r in raw_claims:
    if r.get("action") not in ("claimmentee", "claimmentee-no-previous-mentor") or not r.get("timestamp", "") < CUT: continue
    pr = r.get("params") or {}
    if isinstance(pr, str): pr = ast.literal_eval(pr)
    raw_rows.append({"logid": int(r["logid"]), "ts": r["timestamp"], "action": r["action"],
                     "mentee_name": norm_name((r.get("title") or "").split(":", 1)[-1]), "claimer": norm_name(r.get("user")),
                     "claimer_uid": r.get("userid"), "previous_mentor": norm_name(pr.get("previous-mentor")), "comment": r.get("comment") or ""})
R = pd.DataFrame(raw_rows).set_index("logid")
C = c.set_index("logid")
print("raw claim log entries by action:", Counter(r.get("action") for r in raw_claims), "| on or after the cutoff:",
      sum(1 for r in raw_claims if r.get("timestamp", "") >= CUT))
ck.add("claim rows = raw claim entries before the cutoff (same logids)", set(C.index) == set(R.index) and C.index.is_unique)
ck.add("every claim time is before the cutoff", (c.ts < CUT).all())
for f in ("ts", "action", "mentee_name", "claimer", "claimer_uid", "previous_mentor", "comment"):
    n_bad = sum(clean(C.at[i, f]) != clean(R.at[i, f]) for i in C.index)
    ck.add(f"claims.{f} equals the raw log entry ({n_bad} differ)", n_bad == 0)
mk = {i: my_kind(C.at[i, "comment"], C.at[i, "claimer"], C.at[i, "previous_mentor"]) for i in C.index}
ma = {i: my_alt(C.at[i, "comment"], C.at[i, "mentee_name"], C.at[i, "claimer"]) for i in C.index}
bad_kind = [i for i in C.index if mk[i][0] != C.at[i, "kind"]]
bad_alt = [i for i in C.index if ma[i][0] != bool(C.at[i, "alt_or_test"])]
ck.add(f"kind re-derived with separate code agrees for every claim ({len(bad_kind)} differ)", not bad_kind)
ck.add(f"alt_or_test re-derived with separate code agrees for every claim ({len(bad_alt)} differ)", not bad_alt)
for i in (bad_kind + bad_alt)[:10]:
    print(f"   logid {i}: stored kind={C.at[i, 'kind']} alt={C.at[i, 'alt_or_test']} | ours {mk[i]} {ma[i]} | comment {C.at[i, 'comment']!r}")
# names resolved to uids: where the name is known from M2 or M3 it must give the same uid
known = {}
for n, u in list(zip(prof.current_name, prof.index)) + list(zip(prof.name_at_fetch, prof.index)) + list(zip(q.mentee_name, q.mentee_uid)):
    if n: known.setdefault(norm_name(n), set()).add(int(u))
kn = c[c.mentee_name.map(lambda n: len(known.get(n, ())) == 1)]
ck.add("every claimed name resolved to a uid", c.mentee_uid.notna().all())
ck.add(f"claims on known asker names carry that asker's uid ({len(kn)} rows checked)",
       all(int(r.mentee_uid) in known[r.mentee_name] for r in kn.itertuples()))
print(f"claims on asker names known from M2/M3: {len(kn)}; others resolved through the API or rename log: {len(c) - len(kn)} "
      f"(their uids are asker uids for {int(c[~c.index.isin(kn.index)].mentee_uid.isin(E.index).sum())} rows)")
'''),
md("""
**What the output shows.** All 13 checks pass. The 815 claim rows match the raw log field by field (0 differences),
and our separate rules give the same label for every row (0 differences for `kind`, 0 for alternate or test). All
claimed names got a user ID. For 77 rows the name is already known from M2 or M3 and the ID agrees; the other 738
were looked up through Wikipedia's user data or rename log, and only 10 of those are askers. In other words, most
claimed accounts never asked their mentor a question, so they do not affect the sample at all.
"""),
md("""
### 3.3 Eligibility re-derived for **all** askers with separate code

**What this checks.** This is the core test of the module. We apply the five rules in the documented order using our own classification of the claims from 3.2, and compare
every decision column. A mismatch here would mean either the pipeline or this notebook applies a rule differently.
"""),
code(r'''
EXC = {"autocreate": "autocreated_account", "create2": "created_by_other", "byemail": "created_by_other", "forcecreatelocal": "forced_local_account"}
D10 = {"no_creation_log", "registered_before_mentorship", "autocreated_account", "created_by_other", "forced_local_account"}
cl = c.assign(my_kind=[mk[i][0] for i in c.logid], my_alt=[ma[i][0] for i in c.logid])
by_uid = {int(u): g for u, g in cl.dropna(subset=["mentee_uid"]).groupby("mentee_uid")}
def my_decision(uid):
    p = prof.loc[uid] if uid in prof.index else pd.Series(dtype=object)
    f = first.loc[uid]; g = by_uid.get(uid)
    name = clean(p.get("current_name")) or f.mentee_name or ""
    act, reg = clean(p.get("creation_action")), clean(p.get("registration"))
    man = g[g.my_kind == "manual"] if g is not None else pd.DataFrame(columns=["ts"])
    rs = []
    if name.startswith("~2"): rs.append("temporary_account")
    if act is None: rs.append("no_creation_log")
    elif act != "create": rs.append(EXC.get(act, f"creation_{act}"))
    if reg is None or reg < START: rs.append("registered_before_mentorship")
    if g is not None and g.my_alt.any(): rs.append("alt_or_test_account")
    if (man.ts < f.q_ts).any(): rs.append("manual_claim_before_question")
    return {"eligible": not rs, "exclude_reason": rs[0] if rs else None, "exclude_reasons": rs,
            "eligible_unrestricted": not [r for r in rs if r not in D10 and not r.startswith("creation_")],
            "claimed_after_question": bool((man.ts >= f.q_ts).any()), "n_claims": 0 if g is None else len(g), "n_manual_claims": len(man)}
mine = {u: my_decision(u) for u in E.index}
DEC = ["eligible", "exclude_reason", "exclude_reasons", "eligible_unrestricted", "claimed_after_question", "n_claims", "n_manual_claims"]
mism = {f: [u for u in E.index if clean(mine[u][f]) != clean(E.at[u, f])] for f in DEC}
for f in DEC:
    ck.add(f"{f}: re-derived value equals the stored value for all {len(E):,} askers ({len(mism[f])} mismatches)", not mism[f])
    for u in mism[f][:5]: print(f"   uid {u}: ours {mine[u][f]!r}, stored {clean(E.at[u, f])!r}")
ck.add("eligible is True exactly when exclude_reason is empty", (e.eligible == e.exclude_reason.isna()).all())
ck.add("exclude_reason is the first element of exclude_reasons", all((len(r) == 0 and x is None) or (len(r) > 0 and r[0] == x) for r, x in zip(e.exclude_reasons, e.exclude_reason)))
ck.add("every eligible asker registered themselves (create)", (e[e.eligible].creation_action == "create").all())
ck.add("every eligible asker registered on or after 2021-05-04", (e[e.eligible].registered_at >= START).all())
ck.add("eligible implies eligible_unrestricted", (~e.eligible | e.eligible_unrestricted).all())
'''),
md("""
**What the output shows.** All 12 checks pass: for all 37,095 askers our own decision equals the stored one in every
column (0 mismatches). Every eligible asker registered the account themselves and did so on or after 2021-05-04.
Everyone eligible is also eligible under the looser robustness rule (`eligible_unrestricted`, 37,056 people), as it
should be, since the looser rule only drops conditions.
"""),
md("""
### 3.4 Diagnostics: the primary mentor's roster state at registration (from the M1 periods)

**What this checks.** For each asker we look up, in the M1 timeline of the mentor list, whether their mentor was on
the list, and in the automatic assignment pool, at the moment the asker registered. These flags are only for
diagnosis and do not exclude anyone. We recompute all four flags with separate code and compare.
A period covers time t when `start <= t < end` (an empty end means still on the list).
"""),
code(r'''
per = {int(u): g for u, g in periods.groupby("mentor_uid")}
def my_diag(uid):
    reg = clean(E.at[uid, "registered_at"]); m = E.at[uid, "primary_mentor_uid"]
    if reg is None or pd.isna(m): on = auto = False
    else:
        g = per.get(int(m))
        hit = g[(g.start <= reg) & (g.end.isna() | (g.end > reg))] if g is not None else pd.DataFrame(columns=["in_auto_pool"])
        on, auto = len(hit) > 0, bool(hit.in_auto_pool.any())
    g = by_uid.get(uid)
    explained = g is not None and not pd.isna(m) and (g.claimer_uid == int(m)).any()
    return {"mentor_on_roster_at_registration": on, "mentor_in_auto_pool_at_registration": auto,
            "assigned_later": reg is not None and not on,
            "possible_forcementor": bool(on and not auto and reg is not None and reg >= FORCE and not explained)}
diag = {u: my_diag(u) for u in E.index}
for f in ("mentor_on_roster_at_registration", "mentor_in_auto_pool_at_registration", "assigned_later", "possible_forcementor"):
    bad = [u for u in E.index if diag[u][f] != bool(E.at[u, f])]
    ck.add(f"{f}: re-derived from M1 periods for all askers ({len(bad)} mismatches)", not bad)
    for u in bad[:5]: print(f"   uid {u}: ours {diag[u][f]}, stored {E.at[u, f]}")
ck.add("period times have the ISO format (so text comparison is time comparison)",
       periods.start.str.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$").all() and periods.end.dropna().str.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$").all())
'''),
md("""
**What the output shows.** All 5 checks pass with 0 mismatches. The counts of these flags are discussed in section 7.
"""),
md("""
**Summary of the invariant checks.** The next cell counts every OK and FAIL line from sections 1 to 3.
"""),
code(r'''
ck.report()
'''),
md("""
**What the output shows.** 42 of 42 checks passed; nothing failed.
"""),
md("""
## 4. Comparison with the previous report, an independent claim log, and the paper sample

### 4.1 Previous report

**What this checks.** The main counts now are compared with those saved in `results/m4_report.json` by the previous
run of this notebook. Any change would mean the data or the code changed since then and should be explained.
"""),
code(r'''
now_vals = {"n_askers": len(e), "n_eligible": int(e.eligible.sum()), "n_eligible_unrestricted": int(e.eligible_unrestricted.sum()),
            "exclude_reason": e.exclude_reason.value_counts().to_dict(), "claimed_after_question": int(e.claimed_after_question.sum()),
            "claims": c.kind.value_counts().to_dict()}
if prev_report:
    display(pd.DataFrame([{"key": k, "previous": json.dumps(prev_report.get(k)), "now": json.dumps(v), "same": prev_report.get(k) == v}
                          for k, v in now_vals.items()]))
'''),
md("""
**What the output shows.** All six counts are the same as before ("same" is True in every row): 37,095 askers,
36,094 eligible, 37,056 eligible under the looser rule, 40 claimed after their question, and 450 manual, 358 bulk
and 7 self-transfer claims. So the results are stable from run to run.
"""),
md("""
### 4.2 Independent partial claim log (2SLS pipeline, June 2026)

**What this checks.** `z_manual_claims_log.jsonl` was downloaded separately in June 2026 for an earlier analysis and
holds part of the claim log. It is an independent copy, so it can catch download errors. Every claim in it that is
before the cutoff should be in our claim table with the same claimed name, claimer and previous mentor.
"""),
code(r'''
if not config.EXTERNAL["setmentor_log_partial"].exists():
    print("skipped: the optional validation data (data/input/validation/) is not available")
    z = zin = zbad = []
else:
    z = [r for r in iter_jsonl(config.EXTERNAL["setmentor_log_partial"]) if r.get("action") == "claimmentee" and r.get("timestamp", "") < CUT]
    zin = [r for r in z if r["logid"] in C.index]
    zbad = [r for r in zin if (norm_name(r.get("mentee")), norm_name(r.get("actor")), norm_name(r.get("previous_mentor")))
            != (C.at[r["logid"], "mentee_name"], C.at[r["logid"], "claimer"], C.at[r["logid"], "previous_mentor"])]
    print(f"independent claim entries before the cutoff: {len(z)}; found in ours: {len(zin)} ({len(zin) / max(len(z), 1):.2%}); fields differ: {len(zbad)}")
    pairs = Counter()
    for r in zbad:
        for f, zf in (("mentee_name", "mentee"), ("claimer", "actor"), ("previous_mentor", "previous_mentor")):
            if norm_name(r.get(zf)) != C.at[r["logid"], f]: pairs[(f, norm_name(r.get(zf)), C.at[r["logid"], f])] += 1
    print("differences as (field, June name, our name): number of rows")
    for k, v in pairs.items(): print("  ", k, v)
    print("A name that differs in every row in the same way is a rename between June and October: the API returns the")
    print("user's current name in the user field of old log entries. The claimer_uid in our table does not change with a rename.")
    print("missing from ours:", [r["logid"] for r in z if r["logid"] not in C.index][:10])
'''),
md("""
**What the output shows.** All 709 claims in the June copy are in our table (100%), and none is missing. Ten rows
differ, but only in the claimer's name, and always in the same two ways (7 rows and 3 rows). These are two mentors who
renamed their accounts between June and October: Wikipedia shows the current name on old log entries. The claimer's
user ID does not change with a rename, so these differences do not affect any decision.
"""),
md("""
### 4.3 Paper sample

**What this checks.** How the new eligibility rules change the sample compared with the earlier version of the paper.
The paper's final sample (psm_dataset.npz, N = 35,381) is matched by question revid to our first questions. The paper
excluded people by "mentor weight at question time >= 1"; the new rules are random assignment at registration
(R2.1, D10) and manual claims (R3).
"""),
code(r'''
zz = np.load(config.EXTERNAL["paper_npz"], allow_pickle=True)
s7 = {int(r["conversation_id"]): r["revid"] for r in iter_jsonl(config.EXTERNAL["paper_s7"])}
paper_rev = {int(s7[int(cid)]) for cid in zz["cid_order"] if s7.get(int(cid))}
e["in_paper"] = e.first_question_revid.isin(paper_rev)
print("paper sample", len(zz["cid_order"]), "| matched to a first question here", int(e.in_paper.sum()))
display(pd.crosstab(e.in_paper.rename("in paper sample"), e.eligible.rename("eligible now"), margins=True))
print("\nreasons for people in the paper sample who are excluded now:")
print(e[e.in_paper & ~e.eligible].exclude_reason.value_counts().to_string())
'''),
md("""
**What the output shows.** Of the 35,381 people in the earlier paper sample, 33,860 match a first question here.
Of those, 32,984 are still eligible and 876 (2.6%) are now excluded, mostly for how the account was created (410
created by another person, 183 copied from another wiki, 73 with no creation log) or for registering before
2021-05-04 (191); only 15 are excluded for a manual claim and 4 as alternate or test accounts. In the other direction,
3,110 people are eligible now but were not in the paper sample (the earlier sample used a different rule and earlier
data). The 1,521 paper rows that do not match a first question here are not explained by this check; a likely cause
is that the earlier paper row was not the person's earliest question in the new M2 data, but a reviewer may want to
confirm this.
"""),
md("""
## 5. Tracer samples and random samples

### 5.1 Fixed tracer claim logs

**What this checks.** Five real claim log entries were picked by hand because each is a typical case: a bulk transfer
after a mentor left, a claim of people met at an edit-a-thon, a claim after an email request, a claim after an
adoption request, and a mentor claiming their own alternate account. For each, we print the entry, M4's label, our
label and the asker's resulting decision. Open the links and compare with the Special:Log text.
"""),
code(r'''
tracer_logids = {179458284: "bulk transfer (Asilvering)", 145507520: "edit-a-thon claim (Voidxor)", 176026789: "email request (SignedInteger)",
                 166371855: "adoption request (CFA)", 127195674: "mentor's alt account (NM Demo 2)"}
for lid, why in tracer_logids.items():
    if lid not in C.index:
        print(f"\n{lid} {why}: not among the claim rows before the cutoff"); continue
    r = C.loc[lid]
    print(f"\n{lid} {why}\n  {r.ts}  {r.claimer} claimed {r.mentee_name} (uid {r.mentee_uid}) from {r.previous_mentor}  kind={r.kind} alt_or_test={r.alt_or_test}")
    print(f"  comment: {r.comment!r}\n  ours: {mk[lid]} | {ma[lid]}\n  {logid(lid)}")
    if pd.notna(r.mentee_uid) and int(r.mentee_uid) in E.index:
        print("  asker:", E.loc[int(r.mentee_uid), ["first_q_ts", "eligible", "exclude_reason", "claimed_after_question"]].to_dict())
    else: print("  (this account is not an asker)")
'''),
md("""
**What the output shows.** All five are labeled as expected: the transfer is `bulk`, the other four are `manual`, and
only the alternate account is marked as alternate or test. Two claimed accounts never asked a question, so they do not
affect the sample. The email-request and adoption-request askers are excluded (`manual_claim_before_question`)
because the claim came before their first question; in the email case the claim came less than two minutes before
the question, which is correct by the rule. The alternate account is excluded as `alt_or_test_account`.
"""),
md("""
### 5.2 Random askers: every decision input next to the decision

**What this checks.** Five randomly drawn eligible askers (and, in the next cell, up to five for each exclusion
reason), with all raw inputs printed next to the decision, so a reader can confirm the decision by eye and through the
Wikipedia links. For each person: the creation log record from M3 (with a link), the registration time, the first question (with a
link), every claim with its time relative to the first question, our classification, and the alt/test evidence.
Then our own decision (3.3) next to the stored one.
"""),
code(r'''
random_ok = []
def show_asker(uid):
    r, p, f = E.loc[uid], prof.loc[uid], first.loc[uid]
    print("\n" + "=" * 110)
    print(f"uid {uid}  {p.current_name}  {contribs(p.current_name)}")
    print(f"  first question {r.first_q_ts}  {diff(r.first_question_revid)}")
    print(f"  registered {r.registered_at}  ({'on or after' if (r.registered_at or '') >= START else 'BEFORE'} 2021-05-04)   name starts with ~2: {str(p.current_name).startswith('~2')}")
    lg = mlogs[(mlogs.uid == uid) & (mlogs.logid == p.creation_logid)] if pd.notna(p.creation_logid) else mlogs.iloc[:0]
    if len(lg):
        x = lg.iloc[0]
        print(f"  creation log (raw row in M3 logs): {x.timestamp} {x.type}/{x.action} by {x.performer} params {x.params}  {logid(x.logid)}")
    else:
        print(f"  creation log: none in the person's logs (creation_source={p.creation_source}, action={p.creation_action})")
    g = by_uid.get(uid)
    if g is None: print("  claims: none")
    else:
        rows = [{"logid": x.logid, "ts": x.ts, "vs first question": "before" if x.ts < f.q_ts else "after", "claimer": x.claimer,
                 "previous": x.previous_mentor, "stored kind": x.kind, "our kind": mk[x.logid][0], "alt/test": ma[x.logid][1],
                 "comment": (x.comment or "")[:50], "link": logid(x.logid)} for x in g.sort_values("ts").itertuples()]
        print(pd.DataFrame(rows).to_string(index=False))
    ok = compare({k: mine[uid][k] for k in DEC}, {k: clean(r[k]) for k in DEC}, "  decision: ours vs stored")
    random_ok.append(ok)

print("### eligible askers")
for uid in pick(e[e.eligible], N_RANDOM, SEED).mentee_uid: show_asker(uid)
'''),
md("""
**What the output shows.** Each of the five eligible askers registered the account themselves (`newusers/create`)
after 2021-05-04, has no claims, and our decision matches the stored one in every field. The next cell does the same
for each exclusion reason.
"""),
code(r'''
for reason in e.exclude_reason.dropna().value_counts().index:
    print(f"\n\n### excluded: {reason} ({int((e.exclude_reason == reason).sum())} askers)")
    for uid in pick(e[e.exclude_reason == reason], N_RANDOM, SEED).mentee_uid: show_asker(uid)
print(f"\nrandom askers: decision matches for {sum(random_ok)} of {len(random_ok)}", "" if all(random_ok) else "-> MISMATCH, see above")
'''),
md("""
**What the output shows.** All 35 random askers (5 eligible plus up to 5 per reason; there are only 4 alternate or
test cases and 1 forced local account) match in every field. For example, the "created by another person" cases show a
creation log of type `byemail` made by a different user, which is exactly what the rule looks for. Some people
registered long before their first question (for example in 2022 and asked in 2023); this is fine because the mentor
is drawn at registration, not at the question.
"""),
md("""
### 5.3 Random claim log rows: raw entry next to the classification

**What this checks.** For each kind (and for rows judged to be an alt or test account), the raw log entry as downloaded, the stored
classification, and our classification with the reason in words.
"""),
code(r'''
raw_by_id = {int(r["logid"]): r for r in raw_claims}
claim_ok = []
groups = {k: c[c.kind == k] for k in ("bulk", "self_transfer", "manual")}
groups["alt_or_test = True"] = c[c.alt_or_test]
for name, g in groups.items():
    print(f"\n### {name} ({len(g)} rows)")
    for r in pick(g, N_RANDOM, SEED).itertuples():
        raw = raw_by_id[r.logid]
        print(f"\nlogid {r.logid}  {logid(r.logid)}")
        print("  raw entry:", json.dumps({k: raw.get(k) for k in ("timestamp", "action", "user", "title", "params", "comment")}, ensure_ascii=False)[:400])
        print(f"  kind reason: {mk[r.logid][1]}\n  alt/test reason: {ma[r.logid][1]}")
        claim_ok.append(compare({"kind": mk[r.logid][0], "alt_or_test": ma[r.logid][0]}, {"kind": r.kind, "alt_or_test": bool(r.alt_or_test)}))
print(f"\nrandom claim rows: classification matches for {sum(claim_ok)} of {len(claim_ok)}", "" if all(claim_ok) else "-> MISMATCH")
'''),
md("""
**What the output shows.** All 20 random claim rows (5 each from 358 bulk, 7 self-transfer, 450 manual and 46
alternate or test rows) get the same label from our rules as from M4, with the reason printed in words.
"""),
md("""
## 6. Edge-case lists

**What this checks.** Automatic rules can misread free-text comments, so the unusual cases are listed in full for a
person to read. The done criteria ask a person to read every comment classified as a manual claim. They are grouped
by comment below. Then all alt or test accounts, the claims that mention alt/sock/bot/test but were not judged to be
the mentor's own account, all self transfers, and how claims affect askers.
"""),
code(r'''
summ = json.loads((M4 / "_issues" / "summary.json").read_text())
print("issues:", {k: v["n"] for k, v in summ.items()})
m = c[c.kind == "manual"]
print("manual claims", len(m), "; grouped by comment:")
display(m.assign(comment=m.comment.str.slice(0, 80)).groupby("comment").agg(n=("logid", "size"), claimers=("claimer", lambda s: ", ".join(sorted(set(s))[:3]))).sort_values("n", ascending=False).head(60))
print("\nalt or test accounts of the claimer:")
display(c[c.alt_or_test][["logid", "ts", "claimer", "mentee_name", "kind", "comment"]].assign(comment=lambda d: d.comment.str[:70]))
print("\nclaims that mention alt/sock/bot/test but were NOT judged the claimer's own account (read these):")
mention = c[c.comment.str.contains(ALT) & ~c.alt_or_test]
display(mention[["logid", "claimer", "mentee_name", "comment"]].assign(comment=lambda d: d.comment.str[:80]))
print("\nmoves between two accounts of the same mentor (self_transfer):")
display(c[c.kind == "self_transfer"][["ts", "previous_mentor", "claimer", "mentee_name", "comment"]])
'''),
md("""
**What the output shows.** There are 450 manual claims; the most common comments are empty (40), short reasons such
as "Interested in Wikipedia!", requests from the newcomer, edit-a-thon or training contacts, and tests. 46 claims are
judged to be the mentor's own alternate or test account. The list of claims that mention alt, sock, bot or test but
were *not* judged the mentor's own account is the main place to look for errors: many are other people's accounts
claimed "for testing" (for example staff accounts or other experienced editors), which stay `manual` and still count as
a purposeful claim, so they are excluded if the claim came before the question either way. The 7 self transfers are
mentors moving people between two of their own accounts, which correctly do not count as a new choice. (The
"UserWarning" printed in this cell is a harmless pandas notice about the search pattern.)
"""),
md("""
**What this checks.** How manual claims touch askers: who was claimed before or after their first question, and
which claims fall within one hour of the question, where the order is close and worth checking by hand.
"""),
code(r'''
x = e[e.n_manual_claims > 0]
print("askers with a manual claim", len(x), "| before the question", int(e.exclude_reasons.map(lambda r: "manual_claim_before_question" in list(r)).sum()),
      "| after the question", int(e.claimed_after_question.sum()))
cc = c[c.mentee_uid.isin(x.mentee_uid)].merge(e[["mentee_uid", "first_q_ts", "exclude_reason"]], on="mentee_uid")
display(cc[["mentee_name", "claimer", "ts", "first_q_ts", "kind", "exclude_reason", "comment"]].assign(comment=cc.comment.str[:60]).sort_values("ts"))
print("\nclaims whose account is not an asker:", int((~c.mentee_uid.isin(E.index)).sum()), "of", len(c))
print("askers whose first question and a manual claim are within one hour of each other (timing is close; check by hand):")
near = cc[(cc.kind == "manual") & ((pd.to_datetime(cc.ts) - pd.to_datetime(cc.first_q_ts)).abs() < pd.Timedelta(hours=1))]
display(near[["mentee_name", "ts", "first_q_ts", "claimer", "comment"]])
'''),
md("""
**What the output shows.** 76 askers have at least one manual claim: 37 were claimed before their first question and
40 after (one person has both). Only 19 have `manual_claim_before_question` as their main reason, because the other
18 were already excluded by an earlier rule (for example registered before 2021-05-04). The 40 claimed after the
question stay in the sample and are flagged for a robustness check. Of all 815 claims, 728 concern accounts that never
asked a question. 19 claims fall within one hour of the first question; the rule uses the exact timestamps, so a claim
even one minute before the question excludes the person, which matches the idea that the mentor was then chosen on
purpose.
"""),
md("""
## 7. Distributions and diagnostics

**What this checks.** The breakdown of decisions by registration year, the counts of the two diagnostic flags
(`assigned_later`: the mentor was not yet on the mentor list when the person registered; `possible_forcementor`: the
mentor was on the list but not in the automatic pool, after 2023-02-02, with no claim to explain it), and claims per
year by kind.
"""),
code(r'''
e["reg_year"] = pd.to_datetime(e.registered_at, utc=True, errors="coerce").dt.year
display(pd.crosstab(e.reg_year, e.exclude_reason.fillna("eligible")))
print("\nassigned_later (registration known, primary mentor not on the list at registration):", int(e.assigned_later.sum()))
display(pick(e[e.assigned_later], 10, SEED)[["mentee_uid", "registered_at", "first_q_ts", "primary_mentor_uid", "eligible"]])
print("\npossible_forcementor (mentor on the list but not in the auto pool at registration, after 2023-02-02, no claim explains it):")
display(e[e.possible_forcementor][["mentee_uid", "registered_at", "first_q_ts", "primary_mentor_uid", "eligible"]])
print("\nclaims per year by kind:")
display(pd.crosstab(c.ts.str[:4], c.kind))
'''),
md("""
**What the output shows.** No account registered before 2021 is eligible, and accounts from 2004 to 2012 show up as
"no creation log" because that rule is checked first (very old accounts often lack this log entry). Eligible askers
grow from 789 (2021, from May) to 14,836 (2025). 472 askers are `assigned_later` (their current mentor was not yet a
mentor when they registered). 381 of them are already excluded for other reasons (218 registered before 2021-05-04,
80 with no creation log, 64 copied from another wiki, and a few others); these are mostly old accounts that got a
mentor later, for example when they first opened their homepage. The other 91 are eligible and have almost no claim
log entries (1 of 91), so their mentor most likely changed through a reassignment that the claim log does not record,
such as the system redistributing mentees when a mentor left; by the project's rules that is still random, but this
notebook does not prove it, so it is worth a reviewer's attention (91 people, 0.25% of the sample). Only 2 askers are
`possible_forcementor`, both eligible; this is too few to change results, but they could be checked by hand. Bulk
claims first appear in the log in 2023.
"""),
md("""
## 8. Summary (written to results/m4_report.json, used by M9)

**What this checks.** Writes the summary counts used later when the paper's numbers are produced, and prints the
totals of all checks in this notebook.
"""),
code(r'''
report = {"n_askers": len(e), "n_eligible": int(e.eligible.sum()), "n_eligible_unrestricted": int(e.eligible_unrestricted.sum()),
          "exclude_reason": e.exclude_reason.value_counts().to_dict(), "claimed_after_question": int(e.claimed_after_question.sum()),
          "claims": c.kind.value_counts().to_dict(), "in_paper": int(e.in_paper.sum()),
          "in_paper_now_excluded": int((e.in_paper & ~e.eligible).sum()),
          "assigned_later": int(e.assigned_later.sum()), "possible_forcementor": int(e.possible_forcementor.sum())}
(config.RESULTS / "m4_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps(report, indent=1, ensure_ascii=False))
n_fail = sum(not ok for _, ok in ck.items)
print(f"\ninvariants: {len(ck.items) - n_fail} / {len(ck.items)} OK | random askers matching: {sum(random_ok)} / {len(random_ok)} | "
      f"random claim rows matching: {sum(claim_ok)} / {len(claim_ok)}")
print(f"notebook run time {time.time() - T0:.0f} s")
'''),
md("""
**What the output shows.** In the run on the final data (cutoff 2026-04-20T14:00Z): 42 of 42 invariants OK, 35 of 35
random askers match, and 20 of 20 random claim rows match. M4 keeps 36,094 of 37,095 askers (97.3%) as having a
randomly assigned mentor. Of the earlier paper sample, 876 people who matched are now excluded.
"""),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "m4_check.ipynb"
    write_notebook(CELLS, out)
    print("wrote", out)
    if "--run" in sys.argv:
        run_as_script(out, cwd=out.parent.parent)
        print("\nnotebook ran end to end")


if __name__ == "__main__":
    main()
