"""
build_m2_check.py: writes checks/m2_check.ipynb, the check notebook for M2 (questions and replies)

The notebook follows the fixed 8-section structure (PLAN.md section 12):
    1 module description  2 files and manifest  3 invariants  4 comparison with the previous version (eval2)
    5 tracer samples and random samples (raw records next to processed values)  6 edge cases  7 distributions
    8 summary (results/m2_report.json)

The key idea: the notebook recomputes the derived fields (speaker roles, reply fields, away notice,
talk page title at question time, entry point, removal reason) with its own simple code, for every
question, and compares them with the M2 output. Values that come from the DiscussionTools parser are
shown next to the raw parser record.

How to run:
    python3 checks/build_m2_check.py          # write the notebook
    python3 checks/build_m2_check.py --run    # write it and run it from top to bottom
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks._nbgen import code, md, run_as_script, write_notebook  # noqa: E402

CELLS = [
md("""
# M2 questions and replies: check notebook

## 1. What this module does

**Code**: `pipeline/m2_conversations/` (`collect_topup.py` downloads raw data, `turns.py` splits a thread into turns,
`build_questions.py` builds the tables offline).
**Rules**: DATA_PROCESSING R4.1 to R4.6 (questions), R5.1 to R5.6 (replies), R6.1 and R6.2 (names and uids); issues I-34, I-45, I-55.

When a newcomer clicks "ask your mentor", Wikipedia saves a new section on the mentor's talk page and tags
that revision `mentorship module question` or `mentorship panel question`. M2 does the following:

1. **Questions** (R4.1): one row per tagged revision before the cutoff (2026-04-20T14:00:00Z). The list comes from
   eval2's collection plus a later top-up sweep.
2. **Talk page owner** (I-45): if the talk page was moved later (for example a mentor renamed), the move log is followed
   backwards to get the title at question time (`page_title_at_q`); its owner is `page_mentor_uid`.
3. **Away notice** (R4.2, R4.4): when the newcomer's own mentor is away, the question goes to a backup mentor and starts with
   `Note: X's mentor Y is away.` Y is the **primary mentor**. The note is removed from the text.
4. **Thread and turns** (R5.1, R5.5): the thread holding the question is found in Wikipedia's own DiscussionTools parse of each
   talk page and archive at the cutoff. Sections deleted before the cutoff are recovered from revision history (R5.6).
5. **Speaker role** of each turn (R5.2): `mentee` (the asker), `page_mentor` (owner of the talk page), `primary_mentor`
   (the away mentor), `other_mentor` (another mentor on the roster at that time), `third_party` (anyone else), `unknown`
   (no signature found). The decision uses user ids, M1's mentor names, and an alt-account table.
6. **Reply fields** (R5.3): a reply is any turn not by the newcomer, at or after the question minute and before the cutoff.
   `has_any_reply` is the main treatment; `first_reply_ts` starts the clock; `has_mentor_reply` (page or primary mentor) is the robustness version.

**Paper numbers that depend on it**: the treatment (reply or not), the timing origin (first reply time), reply
features (RQ3), and the question text (M6).

## How to use this notebook

* Run all cells from the top (about 1 to 2 minutes). Every check prints `[OK ]` or `[FAIL]`; mismatch tables print `MISMATCH`.
* Section 3.2 recomputes every derived field for **all** questions with simple code written here (not the pipeline functions).
* Section 5 shows random questions in full: raw wikitext next to cleaned text, the raw DiscussionTools record next to each
  processed turn, the reason for each speaker role, and a recomputed-versus-stored table. Change `SEED` below to see other questions.
"""),
code(r'''
# ── setup ────────────────────────────────────────────────────────────────────
SEED = 0          # change this number to draw a fresh random sample in sections 3.3 and 5
N_RANDOM = 5      # how many random questions to show in each group
N_BULK = 1000     # how many random questions get the bulk "raw DiscussionTools vs turns" check in 3.3

import json, re, sys, html as _html
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.common.io import read_jsonl, iter_jsonl
from pipeline.common.tracer import tracer_frame
from checks._show import Checks, compare, pick, oldid, diff, contribs, logid
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 220)
M2 = config.INTERIM / "m2"; RAW = config.RAW / "m2"; CUTOFF = config.DATA_CUTOFF_ISO
q = pd.read_parquet(M2 / "questions.parquet")
qi = q.set_index("question_revid", drop=False)
turns = {r["question_revid"]: r["turns"] for r in iter_jsonl(M2 / "turns.jsonl")}
pages = {p["title"]: p for p in iter_jsonl(RAW / "pages.jsonl")}
moves = read_jsonl(RAW / "move_log.jsonl")
fb_map = json.loads((RAW / "fallback_map.json").read_text())
removals = json.loads((RAW / "removal_edits.json").read_text())
resolved_raw = json.loads((RAW / "replier_uids.json").read_text())   # signature name -> uid, looked up online in the fallback phase
topup = {int(r["revid"]): r for r in iter_jsonl(RAW / "tag_topup.jsonl")}
topup_wt = json.loads((RAW / "question_wikitext.json").read_text())
EV_KEEP = ("revid", "timestamp", "parentid", "page", "mentor_uid", "mentee_uid", "mentee", "heading", "question_text",
           "tags", "has_any_reply", "first_reply_ts", "first_replier_name")
ev = {int(r["revid"]): {k: r.get(k) for k in EV_KEEP} for r in iter_jsonl(config.EXTERNAL["eval2_conversations"])}
print(f"questions {len(q)} | with turns {len(turns)} | pages {len(pages)} | eval2 records {len(ev)} | top-up {len(topup)}")
'''),
md("## 2. Files and manifest"),
code(r'''
for name in ("m2_collect_pages", "m2_collect_fallback", "m2"):
    p = config.INTERIM / name / "manifest.json"
    if not p.exists():
        print(name, ": no manifest"); continue
    man = json.loads(p.read_text())
    print(f"\n[{name}] finished {man['finished_at']} | cutoff {man['data_cutoff']} | {man['elapsed_sec']} s")
    print(json.dumps(man["counts"], ensure_ascii=False, indent=1)[:1500])
'''),
md("""
### Data dictionary

Every column of `questions.parquet` and every field of a turn in `turns.jsonl`: what it means and where it is checked.
"3.1" = structural invariant, "3.2" = recomputed for all rows, "3.3" = bulk raw-parser comparison, "5.x" = random side-by-side examples.
"""),
code(r'''
DD_Q = [
 ("question_revid", "revision id of the tagged question (primary key)", "3.1 unique; 3.2 present in raw eval2 or top-up"),
 ("parentid", "the revision before the question", "3.2 equals raw record"),
 ("q_ts", "time the question was saved", "3.1 before cutoff; 3.2 equals raw record"),
 ("mentee_uid / mentee_name", "user id and normalized name of the asker (revision author)", "3.2 equals raw record; 5.R"),
 ("tag / source", "which Mentorship tag; eval2 or top-up", "3.1 tag in QUESTION_TAGS; 3.2 tag from raw tags; top-up count"),
 ("heading", "section heading, lower case (eval2) or from the edit summary (top-up)", "3.2 equals raw record; 5.R"),
 ("page_eval", "talk page title as recorded by eval2 (title today)", "3.2 equals raw record"),
 ("page_title_at_q", "talk page title at question time, by following the move log backwards", "3.2 recomputed; 5.M page moves"),
 ("page_mentor_uid", "uid of the owner of page_title_at_q (eval2 mentor if the owner is not a known mentor)", "3.2 recomputed; 5.R"),
 ("eval_mentor_uid", "mentor uid recorded by eval2", "3.2 equals raw record; section 6 page_owner_differs"),
 ("away_routed / primary_mentor_name / primary_mentor_uid", "away notice present; mentor named in it (else page mentor)", "3.2 recomputed by own regex; 5.A"),
 ("editing_page / article_context", "asked from the editor (panel tag + 'on <page>' in heading); that page", "3.2 recomputed; 5.E; section 6 differences"),
 ("question_wikitext", "raw question wikitext with the away notice removed", "3.2 equals raw minus note; 5.R side by side"),
 ("question_text", "plain text of the question (templates, signatures, user links removed)", "3.1 source rule; 5.R side by side; section 6 leftover markup"),
 ("question_text_source / question_text_dt", "where question_text came from; first turn text if the first turn is the mentee", "3.1 rules"),
 ("thread_page / thread_revid / thread_origin", "page and revision where the thread was found; cutoff or fallback type", "3.1 cutoff revid == pages.jsonl; 3.3; 5.R raw thread"),
 ("match_score", "points for heading, asker, owner, and recovered-for-this-question", "5.R recomputed from the raw thread key and heading"),
 ("n_turns", "number of turns kept (before the cutoff)", "3.1 == len(turns); 3.2"),
 ("has_any_reply / first_reply_ts / first_replier_name / first_replier_uid / first_replier_role", "reply by anyone but the newcomer, and the first one", "3.2 recomputed for all rows; 5.R"),
 ("has_mentor_reply / mentor_first_reply_ts", "reply by the page mentor or primary mentor, and its time", "3.2; 5.R"),
 ("n_reply_turns / n_mentor_turns / n_third_party_turns / n_mentee_turns", "counts of turns by type", "3.2; 5.R"),
 ("removal_kind", "why the question was deleted in the next edit (revert, self_removed, merged, other)", "3.2 recomputed from raw edit; 5.D"),
 ("eval_has_any_reply", "eval2's reply status (for comparison)", "3.2 equals raw record; section 4"),
 ("parse_flags", "list of problem flags", "3.1 each flag's rule; section 6 lists"),
]
DD_T = [
 ("author", "signature name found by DiscussionTools (normalized)", "3.3 equals raw author; 5.R"),
 ("signer_uid", "uid of the signer (mentee_uid for the newcomer, mentor uid, or looked-up uid)", "3.2 recomputed; 5.R"),
 ("role", "speaker role (see section 1)", "3.2 recomputed for every turn; 5.R shows the reason"),
 ("ts", "turn time to the minute", "3.1 before cutoff; 3.3 equals raw timestamp"),
 ("level", "reply depth", "3.3 equals raw level"),
 ("comment_id", "DiscussionTools comment id", "3.3 equals raw id"),
 ("text", "plain text of the turn", "5.R next to the raw HTML text; 5.F merged questions"),
 ("links", "internal links, number of external links and list items in the turn HTML", "3.3 n_external and n_list_items counted from raw HTML"),
]
print("questions.parquet columns covered:", sorted(set(q.columns) - {c.strip() for row in DD_Q for c in row[0].split("/")}) or "all")
display(pd.DataFrame(DD_Q, columns=["column", "meaning", "checked in"]))
display(pd.DataFrame(DD_T, columns=["turn field", "meaning", "checked in"]))
'''),
md("""
## 3. Invariants

### 3.1 Structure and flag rules
"""),
code(r'''
ck = Checks()
qts = pd.to_datetime(q.q_ts, utc=True)
qmin = q.q_ts.str[:17] + "00Z"                      # question time rounded down to the minute
all_turn_ts = [t["ts"] for ts in turns.values() for t in ts if t["ts"]]
found = q.thread_page.notna()
ck.add("primary key question_revid is unique", q.question_revid.is_unique)
ck.add("every question has a turns row and vice versa", set(turns) == set(q.question_revid))
ck.add("no question at or after the cutoff", (qts < config.DATA_CUTOFF).all())
ck.add("no turn at or after the cutoff", all(t < CUTOFF for t in all_turn_ts))
ck.add("first reply is not earlier than the question minute",
       (q.first_reply_ts.dropna() >= qmin[q.first_reply_ts.notna()]).all())
ck.add("mentor reply implies any reply", (~q.has_mentor_reply.fillna(False).astype(bool) | q.has_any_reply.fillna(False).astype(bool)).all())
ck.add("every tag is one of the two Mentorship tags", q.tag.isin(config.QUESTION_TAGS).all())
ck.add("all top-up questions before the cutoff are in the table",
       set(q[q.source == "topup"].question_revid) == {r for r, x in topup.items() if x["timestamp"] < CUTOFF})
ck.add("n_turns == number of turns in turns.jsonl", (q.n_turns == q.question_revid.map(lambda r: len(turns[r]))).all())
ck.add("thread not found <=> thread_page empty <=> flag thread_not_found",
       ((~found) == q.parse_flags.map(lambda f: "thread_not_found" in list(f))).all())
ck.add("thread not found -> reply fields empty and no turns",
       q[~found].has_any_reply.isna().all() and (q[~found].n_turns == 0).all())
ck.add("thread_origin cutoff -> thread_revid is the page revision at the cutoff (pages.jsonl)",
       q[q.thread_origin == "cutoff"].apply(lambda r: pages.get(r.thread_page, {}).get("as_of_revid") == r.thread_revid, axis=1).all())
sec_files = defaultdict(set)      # revision id of a recovered section -> questions it was recovered for
for pth in (RAW / "fallback_sections").glob("*.json"):
    sec_files[int(json.loads(pth.read_text())["revid"])].add(int(pth.stem))
fs = q[q.thread_origin == "fallback_section"]
own = fs.apply(lambda r: r.question_revid in sec_files.get(int(r.thread_revid), set()), axis=1)
ck.add("thread_origin fallback_section -> a recovered section file with this thread_revid exists",
       fs.thread_revid.map(lambda x: int(x) in sec_files).all())
print(f"   recovered for this same question: {int(own.sum())}; recovered for another question in the same revision: "
      f"{int((~own).sum())} {fs[~own].question_revid.tolist()[:10]}")
ck.add("flag turn_source_regex <=> thread_origin fallback_regex",
       (q.parse_flags.map(lambda f: "turn_source_regex" in list(f)) == (q.thread_origin == "fallback_regex")).all())
ft = q.question_revid.map(lambda r: turns[r][0]["role"] if turns[r] else None)
ck.add("flag first_turn_not_mentee <=> first turn exists and is not the mentee",
       (q.parse_flags.map(lambda f: "first_turn_not_mentee" in list(f)) == (ft.notna() & (ft != "mentee"))).all())
rbq = pd.Series([any(t["role"] != "mentee" and t["ts"] and t["ts"] < m for t in turns[r]) for r, m in zip(q.question_revid, qmin)], index=q.index)
ck.add("flag reply_before_question <=> some non-mentee turn is earlier than the question minute",
       (q.parse_flags.map(lambda f: "reply_before_question" in list(f)) == rbq).all())
ck.add("question_text_dt == first turn text when the first turn is the mentee, else empty",
       all((turns[r][0]["text"] if turns[r] and turns[r][0]["role"] == "mentee" else None) == d
           for r, d in zip(q.question_revid, q.question_text_dt)))
ck.add("question_text_source rule: wikitext if wikitext gives text, else discussiontools, else none",
       ((q.question_text_source == "none") == (q.question_text == "")).all()
       and (q[q.question_text_source == "discussiontools"].question_text == q[q.question_text_source == "discussiontools"].question_text_dt).all())
ck.add("removal_kind is set exactly when flag section_removed_immediately is set",
       (q.removal_kind.notna() == q.parse_flags.map(lambda f: "section_removed_immediately" in list(f))).all())
ck.add("no away notice -> primary_mentor_uid == page_mentor_uid and primary_mentor_name empty",
       (q[~q.away_routed].primary_mentor_uid == q[~q.away_routed].page_mentor_uid).all() and q[~q.away_routed].primary_mentor_name.isna().all())
print("\nshare of questions with a thread:", f"{found.mean():.2%}", "| origin:", q.thread_origin.value_counts(dropna=False).to_dict())
'''),
md("""
### 3.2 Independent recomputation for every question

The code below re-implements the rules in plain Python, without calling any M2 function:

* **names**: HTML entities decoded, underscores to spaces, first letter upper case, `/subpage` removed, alt accounts mapped to the main account;
* **mentor uid of a name**: from M1's `mentor_names` table (a name that maps to two uids is not used);
* **newcomer names**: the revision author, the asker in the heading, the newcomer named in the away notice, and every
  signature name the fallback phase looked up to the same uid; for flag `question_turn_relabelled`, also the signature of the first turn;
* **role** order: no author -> unknown; newcomer name or uid -> mentee; uid == page mentor -> page_mentor; uid == primary mentor
  -> primary_mentor; a mentor name on the M1 roster at the turn time -> other_mentor; else third_party;
* **reply**: a turn whose role is not mentee, with a time at or after the question minute and before the cutoff.
"""),
code(r'''
def my_name(n):
    if n is None: return None
    n = _html.unescape(str(n)).replace("_", " ")
    n = re.sub(r"[​-‏‪-‮⁠-⁤﻿]", "", n)
    n = re.sub(r"[\s  ]+", " ", n).strip()
    return (n[:1].upper() + n[1:]) or None
ALT = {my_name(a): my_name(m) for a, m in config.MENTOR_ALT_ACCOUNTS.items()}
def person(author):
    """signature name -> the person: drop /subpage, normalize, map alt account to main account"""
    if not author: return None
    p = my_name(str(author).split("/")[0]); return ALT.get(p, p)
mnames = pd.read_parquet(config.INTERIM / "m1" / "mentor_names.parquet")
n2u = defaultdict(set)
for r in mnames.itertuples(): n2u[r.name].add(int(r.mentor_uid))
for a, m in ALT.items(): n2u[a] |= n2u.get(m, set())
MENTOR = {n: next(iter(u)) for n, u in n2u.items() if len(u) == 1}
spans = defaultdict(list)
for r in pd.read_parquet(config.INTERIM / "m1" / "mentor_periods.parquet").itertuples(): spans[r.mentor_uid].append((r.start, r.end))
def on_roster(uid, ts): return any(s <= ts and (e is None or ts < e) for s, e in spans.get(uid, []))
RESOLVED = {my_name(k): v for k, v in resolved_raw.items() if v}

AWAY_RE = re.compile(r"Note:\s*\[\[\s*User(?:[ _]talk)?\s*:\s*([^|\]]+)[^\]]*\]\]\s*'s\s+mentor\s+(?:\[\[\s*User(?:[ _]talk)?\s*:\s*([^|\]]+)[^\]]*\]\]"
                     r"|(?:<span[^>]*>)?\s*\[\S+\s+([^\]]+)\]\s*(?:</span>)?)\s+is\s+away", re.I)
HEAD_ASKER = re.compile(r"^question from (.+?)(?: on .+)? \(\d{1,2}:\d{2}, ")
HEAD_PAGE = re.compile(r"^question from .+? on (.+) \(\d{1,2}:\d{2}, ")
def nhead(h): return re.sub(r"\s+", " ", (h or "").replace("_", " ")).strip().lower()
def raw_wikitext(rv):
    """the question wikitext as downloaded (eval2 record, or question_wikitext.json for top-up questions)"""
    return (ev[rv]["question_text"] if rv in ev else (topup_wt.get(str(rv)) or {}).get("text")) or ""
def my_away(rv):
    m = AWAY_RE.search(raw_wikitext(rv)[:1000])
    return None if not m else {"mentee": my_name(m.group(1)), "primary": my_name(m.group(2) or m.group(3)), "match": m.group(0)}

def mentee_names(r):
    """{lower-case name: why it counts as the newcomer}"""
    out = {}
    if r.mentee_name: out[r.mentee_name.lower()] = "revision author (mentee_name)"
    h = HEAD_ASKER.match(nhead(r.heading))
    if h: out.setdefault(h.group(1), "asker in the heading")
    a = my_away(r.question_revid)
    if a and a["mentee"]: out.setdefault(a["mentee"].lower(), "newcomer named in the away notice")
    for n, u in RESOLVED.items():
        if pd.notna(r.mentee_uid) and u == r.mentee_uid: out.setdefault(n.lower(), f"looked up online: uid {u} == mentee_uid")
    ts = turns[r.question_revid]
    if "question_turn_relabelled" in list(r.parse_flags) and ts and ts[0]["author"]:
        out.setdefault(person(ts[0]["author"]).lower(), "signature of the question turn itself (I-55)")
    return out

def my_role(t, i, r, mn):
    """(role, signer uid, reason) for turn number i of question row r"""
    p = person(t["author"])
    if p is None: return "unknown", None, "DiscussionTools found no signature"
    uid = MENTOR.get(p) or RESOLVED.get(p)
    mu = None if pd.isna(r.mentee_uid) else int(r.mentee_uid)
    if i == 0 and "question_turn_relabelled" in list(r.parse_flags):
        return "mentee", mu, "first turn is the question itself (flag question_turn_relabelled)"
    if p.lower() in mn: return "mentee", mu, f"{p!r} is a newcomer name: {mn[p.lower()]}"
    if uid is not None and uid == mu: return "mentee", mu, f"{p!r} -> uid {uid} == mentee_uid"
    if uid is not None and uid == r.page_mentor_uid: return "page_mentor", uid, f"{p!r} -> uid {uid} == page_mentor_uid (owner of {r.page_title_at_q})"
    if uid is not None and uid == r.primary_mentor_uid: return "primary_mentor", uid, f"{p!r} -> uid {uid} == primary_mentor_uid (away notice)"
    if p in MENTOR and on_roster(uid, t["ts"] or ""): return "other_mentor", uid, f"{p!r} is mentor uid {uid}, on the roster at {t['ts']}, not this question's mentor"
    return "third_party", uid, f"{p!r} (uid {uid}) is not the newcomer and not a mentor on the roster at that time"

def my_reply_fields(ts, q_ts):
    qm = q_ts[:17] + "00Z"
    rep = [t for t in ts if t["role"] != "mentee" and t["ts"] and qm <= t["ts"] < CUTOFF]
    men = [t for t in rep if t["role"] in ("page_mentor", "primary_mentor")]
    f = rep[0] if rep else {}
    return {"has_any_reply": bool(rep), "first_reply_ts": f.get("ts"), "first_replier_name": f.get("author"),
            "first_replier_uid": f.get("signer_uid"), "first_replier_role": f.get("role"),
            "has_mentor_reply": bool(men), "mentor_first_reply_ts": men[0]["ts"] if men else None,
            "n_reply_turns": len(rep), "n_mentor_turns": len(men),
            "n_third_party_turns": sum(t["role"] == "third_party" for t in rep), "n_mentee_turns": sum(t["role"] == "mentee" for t in ts)}

def same(a, b):
    na = a is None or (isinstance(a, float) and pd.isna(a)); nb = b is None or (isinstance(b, float) and pd.isna(b))
    return (na and nb) or (not na and not nb and a == b)
def title_at(title, at):
    cur = title
    for m in sorted(moves, key=lambda m: m["timestamp"], reverse=True):
        if m["timestamp"] > at and m.get("to") and my_name(m["to"]) == my_name(cur): cur = m["from"]
    return cur
def my_removal(e, r):
    if e is None: return None
    tags, com = set(e.get("tags") or []), (e.get("comment") or "").lower()
    if (pd.notna(r.mentee_uid) and e.get("userid") == r.mentee_uid) or my_name(e.get("user")) == r.mentee_name: return "self_removed"
    if tags & {"mw-rollback", "mw-undo", "mw-manual-revert"} or com.startswith(("reverted", "undid", "rv")): return "revert"
    if re.search(r"combin|merg|moved|move to|duplicate", com): return "merged"
    return "other"
'''),
code(r'''
bad = defaultdict(list)          # field -> [question_revid, ...]
role_bad = []
for r in q.itertuples():
    rv = r.question_revid
    # base fields against the raw eval2 / top-up record
    if rv in ev:
        e = ev[rv]
        raw = {"q_ts": e["timestamp"], "parentid": e["parentid"], "page_eval": e["page"], "eval_mentor_uid": e["mentor_uid"],
               "mentee_uid": e["mentee_uid"], "mentee_name": my_name(e["mentee"]), "heading": e["heading"], "source": "eval2",
               "tag": next((t for t in config.QUESTION_TAGS if t in (e["tags"] or [])), None),
               "eval_has_any_reply": str(e["has_any_reply"]) == "True"}
    elif rv in topup:
        e = topup[rv]; m = re.match(r"^/\*\s*(.+?)\s*\*/", e.get("comment") or "")
        raw = {"q_ts": e["timestamp"], "parentid": e["parentid"], "page_eval": e["page"], "eval_mentor_uid": e["mentor_uid"],
               "mentee_uid": e["userid"], "mentee_name": my_name(e["user"]), "heading": nhead(m.group(1)) if m else None,
               "source": "topup", "tag": e["tag"], "eval_has_any_reply": None}
    else:
        bad["not in any raw source"].append(rv); continue
    for k in ("parentid", "eval_mentor_uid", "mentee_uid"):        # numbers may be stored as text or float in the raw files
        raw[k] = None if raw[k] in (None, "") else int(raw[k])
    for k, v in raw.items():
        d = getattr(r, k)
        if k in ("parentid", "eval_mentor_uid", "mentee_uid"): d = None if pd.isna(d) else int(d)
        if not same(v, d): bad[k].append(rv)
    # talk page at question time and its owner
    t_at = title_at(r.page_eval, r.q_ts)
    if t_at != r.page_title_at_q: bad["page_title_at_q"].append(rv)
    owner = MENTOR.get(my_name(t_at.split(":", 1)[-1].split("/")[0]))
    if (owner if owner is not None else r.eval_mentor_uid) != r.page_mentor_uid: bad["page_mentor_uid"].append(rv)
    # away notice
    a = my_away(rv); w = raw_wikitext(rv)
    if bool(a) != r.away_routed: bad["away_routed"].append(rv)
    if (a["primary"] if a else None) != r.primary_mentor_name: bad["primary_mentor_name"].append(rv)
    if (MENTOR.get(a["primary"]) if a else r.page_mentor_uid) != r.primary_mentor_uid: bad["primary_mentor_uid"].append(rv)
    if not a and w != r.question_wikitext: bad["question_wikitext"].append(rv)
    if a:      # expected: the raw text with the line that holds the note removed
        lines = w.split("\n"); k = next(i for i, ln in enumerate(lines) if a["match"].split("\n")[0] in ln)
        if "\n".join(lines[:k] + lines[k + 1:]).strip() != r.question_wikitext.strip(): bad["question_wikitext"].append(rv)
    # entry point (the page name is checked in section 6, because M2 reads it from the heading HTML)
    if (r.tag == "mentorship panel question" and bool(HEAD_PAGE.match(nhead(r.heading)))) != r.editing_page: bad["editing_page"].append(rv)
    # removal reason
    if r.removal_kind is not None and my_removal(removals.get(str(rv)), r) != r.removal_kind: bad["removal_kind"].append(rv)
    # unrecoverable revision deletion
    hidden = (rv in ev and not w) or (rv in topup and "texthidden" in topup[rv])
    if hidden != ("unrecoverable_revdel" in list(r.parse_flags)): bad["flag unrecoverable_revdel"].append(rv)
    # speaker role of every turn, then the reply fields
    ts = turns[rv]; mn = mentee_names(r)
    for i, t in enumerate(ts):
        role, uid, why = my_role(t, i, r, mn)
        if role != t["role"] or not same(uid, t["signer_uid"]): role_bad.append((rv, i, t["author"], role, t["role"], uid, t["signer_uid"], why))
    if r.thread_page is not None:
        for k, v in my_reply_fields(ts, r.q_ts).items():
            if not same(v, getattr(r, k)): bad[k].append(rv)

FIELDS = ["q_ts", "parentid", "page_eval", "eval_mentor_uid", "mentee_uid", "mentee_name", "heading", "source", "tag",
          "eval_has_any_reply", "page_title_at_q", "page_mentor_uid", "away_routed", "primary_mentor_name", "primary_mentor_uid",
          "question_wikitext", "editing_page", "removal_kind", "flag unrecoverable_revdel",
          "has_any_reply", "first_reply_ts", "first_replier_name", "first_replier_uid", "first_replier_role",
          "has_mentor_reply", "mentor_first_reply_ts", "n_reply_turns", "n_mentor_turns", "n_third_party_turns", "n_mentee_turns"]
ck.add(f"every question is in a raw source (missing: {len(bad['not in any raw source'])})", not bad["not in any raw source"])
for f in FIELDS:
    ck.add(f"{f}: recomputed == stored for all rows (mismatches: {len(bad[f])})", not bad[f])
n_turns_all = sum(len(v) for v in turns.values())
ck.add(f"role and signer_uid recomputed == stored for all {n_turns_all} turns (mismatches: {len(role_bad)})", not role_bad)
for f in FIELDS:
    if bad[f]:
        print(f"MISMATCH {f}: {bad[f][:10]}")
        if f == "question_wikitext":
            for rv in bad[f][:3]:
                print(f"   {rv} {oldid(rv)}\n     raw   : {raw_wikitext(rv)[:160]!r}\n     stored: {qi.loc[rv, 'question_wikitext'][:160]!r}")
if role_bad:
    print("MISMATCH roles (first 20):")
    display(pd.DataFrame(role_bad[:20], columns=["revid", "turn", "author", "recomputed", "stored", "uid_recomputed", "uid_stored", "why"]))
'''),
md("""
### 3.3 Raw DiscussionTools record versus processed turns (bulk random sample)

For `N_BULK` random questions whose thread came from a DiscussionTools page parse, find the raw thread in
`data/raw/m2/dt/<revid>.json` (or `fallback_dt/`) and compare it turn by turn with `turns.jsonl`:
author, time (minute), level, comment id, and the number of external links and list items counted from the raw HTML.
Raw comments at or after the cutoff are expected to be dropped.
"""),
code(r'''
_dt_cache = {}
def dt_items(path):
    if path not in _dt_cache:
        if len(_dt_cache) > 50: _dt_cache.clear()
        _dt_cache[path] = json.loads(path.read_text())
    return _dt_cache[path]
def flat_comments(node, out):
    for c in node.get("replies", []):
        if c.get("type") == "comment": out.append(c)
        flat_comments(c, out)
    return out
def raw_thread(r):
    """(heading item, its comments in reading order) from the raw DiscussionTools file, or (None, None).
    The thread is the smallest top-level heading whose comments contain every comment_id of the processed turns."""
    if r.thread_origin not in ("cutoff", "fallback"): return None, None
    path = RAW / ("dt" if r.thread_origin == "cutoff" else "fallback_dt") / f"{int(r.thread_revid)}.json"
    ids = {t["comment_id"] for t in turns[r.question_revid]}
    best = None
    for it in dt_items(path):
        if it.get("type") != "heading": continue
        cs = flat_comments(it, [])
        if ids and ids <= {c["id"] for c in cs} and (best is None or len(cs) < len(best[1])): best = (it, cs)
    return best or (None, None)
def html_text(h): return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", h or ""))).strip()

bulk = pick(q[q.thread_origin.isin(["cutoff", "fallback"]) & (q.n_turns > 0)], N_BULK, SEED)
cnt = Counter(); bulk_bad = []
for r in bulk.itertuples():
    it, cs = raw_thread(r)
    if it is None: cnt["raw thread not found"] += 1; bulk_bad.append((r.question_revid, "raw thread not found")); continue
    kept = [c for c in cs if not (c.get("timestamp") and c["timestamp"] >= CUTOFF)]
    ts = turns[r.question_revid]
    if [c["id"] for c in kept] != [t["comment_id"] for t in ts]:
        cnt["turn list differs"] += 1; bulk_bad.append((r.question_revid, "turn list differs")); continue
    for c, t in zip(kept, ts):
        lk = t["links"] or {}
        for k, ok in (("author", my_name(c.get("author")) == t["author"]), ("ts", (c.get("timestamp") or "")[:16] == (t["ts"] or "")[:16]),
                      ("level", int(c.get("level") or 0) == t["level"]),
                      ("n_external", c.get("html", "").count("mw:ExtLink") == lk.get("n_external")),
                      ("n_list_items", len(re.findall(r"<li[\s>]", c.get("html", ""))) == lk.get("n_list_items"))):
            cnt[k + (" ok" if ok else " MISMATCH")] += 1
            if not ok: bulk_bad.append((r.question_revid, k))
print(f"bulk sample: {len(bulk)} questions, {sum(len(turns[x]) for x in bulk.question_revid)} turns (SEED={SEED})")
print(pd.Series(cnt).sort_index().to_string())
ck.add(f"bulk: raw DiscussionTools thread found and turn ids identical for all sampled questions (bad: {cnt['raw thread not found'] + cnt['turn list differs']})",
       cnt["raw thread not found"] + cnt["turn list differs"] == 0)
for k in ("author", "ts", "level", "n_external", "n_list_items"):
    ck.add(f"bulk: turn {k} equals the raw record (mismatches: {cnt[k + ' MISMATCH']})", cnt[k + " MISMATCH"] == 0)
if bulk_bad: print("MISMATCH examples:", bulk_bad[:10])
ck.report()
'''),
md("""
## 4. Comparison with eval2 (I-34)

eval2 split turns with its own signature regex, and its data were downloaded on 2026-06-12, so it includes turns after
the cutoff. The new version uses DiscussionTools and stops at 2026-04-20T14:00:00Z. Questions are aligned by revid to
see where the reply status changes.
"""),
code(r'''
both = q[q.eval_has_any_reply.notna() & q.has_any_reply.notna()]
ct = pd.crosstab(both.eval_has_any_reply.rename("eval2"), both.has_any_reply.rename("new"), margins=True)
display(ct)
ch = pd.DataFrame(read_jsonl(M2 / "_issues" / "reply_status_changed.jsonl")) if (M2 / "_issues" / "reply_status_changed.jsonl").exists() else pd.DataFrame()
if len(ch):
    ch["revid"] = ch.ids.map(lambda d: d["revid"])
    ch = ch.join(pd.json_normalize(ch.detail))
    print("direction of the change:", ch.groupby(["eval2", "new"]).size().to_dict())
    print("role of the new first replier (eval2 no reply -> new has a reply):", ch[ch.new == True].role.value_counts().to_dict())   # noqa: E712
'''),
code(r'''
# eval2 has a reply but the new version does not: the usual reason is that the reply is after the cutoff.
lost = both[(both.eval_has_any_reply == True) & (both.has_any_reply == False)]   # noqa: E712
lost = lost.assign(eval_first_reply_ts=lost.question_revid.map(lambda r: ev[r]["first_reply_ts"]),
                   eval_first_replier=lost.question_revid.map(lambda r: ev[r]["first_replier_name"]))
lost["after_cutoff"] = pd.to_datetime(lost.eval_first_reply_ts, utc=True, errors="coerce") >= config.DATA_CUTOFF
print(f"eval2 has a reply, new version has none: {len(lost)}; of these, eval2's reply is after the cutoff: {int(lost.after_cutoff.sum())}")
print("the rest (reply before the cutoff in eval2, none now); open the links to check:")
display(lost[~lost.after_cutoff][["question_revid", "q_ts", "page_eval", "eval_first_replier", "eval_first_reply_ts", "parse_flags"]].head(20))
# the reverse direction: random examples with the new first replier
gain = ch[ch.new == True] if len(ch) else pd.DataFrame()   # noqa: E712
for x in pick(gain, N_RANDOM, SEED).itertuples() if len(gain) else []:
    r = qi.loc[x.revid]
    print(f"  {x.revid} {oldid(x.revid)}  new first replier {r.first_replier_name} ({r.first_replier_role}) at {r.first_reply_ts} on {r.thread_page} [{r.thread_origin}]")
'''),
md("""
## 5. Tracer samples and random samples

`show_question(revid)` prints, for one question:

1. the raw records: the raw question wikitext (as downloaded) next to the stored `question_wikitext` and the cleaned `question_text`;
2. the raw DiscussionTools thread (thread key, heading, each raw comment with author, time, level and text) next to the processed
   turn (author, signer uid, role) and **why** that role was given;
3. a table "recomputed vs stored" for the reply fields and other derived fields, and the match score recomputed from the raw thread.
"""),
code(r'''
def _flags(r): return list(r.parse_flags)
def show_question(rv, full=True):
    if rv not in qi.index:
        print(f"\n### {rv}: NOT in the question table (a finding to explain) {oldid(rv)}"); return False
    r = next(qi.loc[[rv]].itertuples())
    print(f"\n{'=' * 100}\n### question {rv}   {oldid(rv)}   diff: {diff(rv)}")
    print(f"  asked {r.q_ts} by {r.mentee_name} (uid {r.mentee_uid})  on {r.page_title_at_q}  tag: {r.tag}  flags: {_flags(r)}")
    if full:
        w = raw_wikitext(rv)
        print(f"  RAW wikitext ({'eval2' if rv in ev else 'top-up'}): {w[:400]!r}")
        if r.away_routed: print(f"  stored question_wikitext (away note removed): {r.question_wikitext[:300]!r}")
        print(f"  CLEAN question_text [{r.question_text_source}]: {r.question_text[:400]!r}")
    mn = mentee_names(r)
    print("  newcomer names used for the role check:", mn)
    it, cs = raw_thread(r)
    if it is not None:
        print(f"  RAW thread key {it.get('name')!r}  heading {html_text(it.get('html'))!r}  file {r.thread_origin}/{int(r.thread_revid)}.json")
    elif r.thread_origin in ("fallback_section", "fallback_regex"):
        p = RAW / ("fallback_sections" if r.thread_origin == "fallback_section" else "fallback_wikitext") / f"{rv}.json"
        info = json.loads(p.read_text())
        ids = re.findall(r'data-mw-comment-start=""[^>]*id="(c-[^"]+)"', info.get("html", "")) if "html" in info else []
        print(f"  RAW recovered section ({r.thread_origin}) from revision {info.get('revid')} {oldid(info.get('revid'))}: {info.get('line') or info.get('heading')!r}")
        print(f"     comment ids in the raw HTML: {[i[:60] for i in ids][:8]}")
    else:
        print("  no thread found")
    raw_by_id = {c["id"]: c for c in (cs or [])}
    rows = []
    for i, t in enumerate(turns[rv]):
        role, uid, why = my_role(t, i, r, mn)
        c = raw_by_id.get(t["comment_id"], {})
        rows.append({"#": i, "raw author": c.get("author", "(no raw DT)"), "raw time": c.get("timestamp"), "raw lvl": c.get("level"),
                     "author": t["author"], "ts": t["ts"], "lvl": t["level"], "signer_uid": t["signer_uid"], "role": t["role"],
                     "role ok": role == t["role"], "why": why})
    for c in (cs or []):
        if c.get("timestamp") and c["timestamp"] >= CUTOFF:
            rows.append({"#": "-", "raw author": c.get("author"), "raw time": c["timestamp"], "why": "dropped: after the cutoff"})
    if rows: display(pd.DataFrame(rows).fillna(""))
    if full:
        for i, t in enumerate(turns[rv][:4]):
            c = raw_by_id.get(t["comment_id"])
            if c: print(f"   turn {i} RAW html text: {html_text(c.get('html'))[:220]!r}")
            print(f"   turn {i} stored text  : {(t['text'] or '')[:220]!r}")
    hand = my_reply_fields(turns[rv], r.q_ts) if r.thread_page is not None else {k: None for k in ("has_any_reply", "first_reply_ts", "first_replier_role")}
    if it is not None:
        h_ok = bool(r.heading) and nhead(html_text(it.get("html"))) == nhead(r.heading)
        key_author = re.sub(r"^h-(.+)-(\d{14}|\d{4}-\d\d-\d\dT[\d:.]+Z?)$", r"\1", it.get("name", "")).replace("_", " ").lower()
        owner = MENTOR.get(my_name(r.thread_page.split(":", 1)[-1].split("/")[0]))
        for_q = r.thread_origin == "fallback" and fb_map.get(str(rv), {}).get("revid") == r.thread_revid
        mn0 = {n for n, why in mn.items() if "I-55" not in why}     # names known before the I-55 relabelling (matching happens first)
        hand["match_score"] = 2 * h_ok + 2 * (key_author in mn0) + (owner is not None and owner == r.page_mentor_uid) + for_q
    a = my_away(rv)
    hand |= {"away_routed": bool(a), "primary_mentor_name": a["primary"] if a else None,
             "page_title_at_q": title_at(r.page_eval, r.q_ts)}
    data = {k: getattr(r, k) for k in hand}
    return compare(hand, data, title="  recomputed (left) vs stored (right):")
'''),
md("""
### 5.T Fixed tracer samples

Questions in the locked tracer list (away notice, revision deleted, wrong section, alt-account reply, signature subpage,
third-party reply) and every question of the tracer newcomers. Shown in short form (no texts).
"""),
code(r'''
tr = tracer_frame()
sample_revids = [int(r) for r in tr.revid.dropna()]
sample_revids += q[q.mentee_uid.isin([int(u) for u in tr[tr.kind == "mentee"].uid.dropna()])].question_revid.tolist()
tracer_ok = [show_question(rv, full=False) for rv in dict.fromkeys(sample_revids)]
print(f"\ntracer questions where every recomputed field matches: {sum(tracer_ok)} of {len(tracer_ok)}")
'''),
md("""
### 5.R Random questions in full

Groups: any question; questions with a reply; questions without a reply (thread found); and one group per first replier role.
"""),
code(r'''
groups = {"any question": q, "with a reply": q[q.has_any_reply == True],          # noqa: E712
          "no reply (thread found)": q[(q.has_any_reply == False)]}               # noqa: E712
for role in sorted(q.first_replier_role.dropna().unique()):
    groups[f"first replier role = {role}"] = q[q.first_replier_role == role]
rand_ok = []
for label, g in groups.items():
    print(f"\n\n################ random: {label} (n={len(g)}, SEED={SEED})")
    rand_ok += [show_question(int(rv)) for rv in pick(g, N_RANDOM, SEED).question_revid]
print(f"\nrandom questions where every recomputed field matches: {sum(rand_ok)} of {len(rand_ok)}")
'''),
md("""
### 5.M Talk page moved after the question (I-45)

`page_eval` is the title today. If the page was moved later, `page_title_at_q` follows the move log backwards.
For random moved pages: the move log entries for that page next to the stored title and owner.
"""),
code(r'''
moved = q[q.page_eval != q.page_title_at_q]
print(f"questions whose talk page was moved after the question: {len(moved)}")
for r in pick(moved, N_RANDOM, SEED).itertuples():
    chain = [m for m in moves if my_name(m.get("to") or "") in {my_name(r.page_eval), my_name(r.page_title_at_q)}
             or my_name(m["from"]) in {my_name(r.page_eval), my_name(r.page_title_at_q)}]
    print(f"\n  question {r.question_revid} at {r.q_ts}  {oldid(r.question_revid)}")
    for m in sorted(chain, key=lambda m: m["timestamp"]):
        print(f"    move {m['timestamp']}  {m['from']} -> {m['to']}  ({'after' if m['timestamp'] > r.q_ts else 'before'} the question)  {logid(m['logid'])}")
    owner = MENTOR.get(my_name(r.page_title_at_q.split(":", 1)[-1].split("/")[0]))
    compare({"page_title_at_q": title_at(r.page_eval, r.q_ts), "page_mentor_uid": owner if owner is not None else r.eval_mentor_uid},
            {"page_title_at_q": r.page_title_at_q, "page_mentor_uid": r.page_mentor_uid})
'''),
md("### 5.A Away notice: raw note next to the stored primary mentor"),
code(r'''
away = q[q.away_routed]
print(f"questions with an away notice: {len(away)}")
for r in pick(away, N_RANDOM, SEED).itertuples():
    a = my_away(r.question_revid)
    print(f"\n  {r.question_revid} {oldid(r.question_revid)}\n    RAW note: {a['match']!r}")
    compare({"primary_mentor_name": a["primary"], "primary_mentor_uid": MENTOR.get(a["primary"]), "page_mentor_uid (owner of the page)": r.page_mentor_uid},
            {"primary_mentor_name": r.primary_mentor_name, "primary_mentor_uid": r.primary_mentor_uid, "page_mentor_uid (owner of the page)": r.page_mentor_uid})
'''),
md("""
### 5.F Question turn relabelled, and question merged into a reply (I-55)

* `question_turn_relabelled`: the first turn is the question itself but its signature does not match the account
  (for example the newcomer signed with another name). It is relabelled as the mentee.
* `question_merged_into_reply`: the question had no signature, so DiscussionTools joined it to the first reply. The question
  text is cut off the reply text; the reply keeps its author and time.

For each random example: raw turns and processed turns side by side. For merged questions, compare the raw HTML text of
the first turn with the stored (cut) text and with the question text.
"""),
code(r'''
for flag in ("question_turn_relabelled", "question_merged_into_reply"):
    g = q[q.parse_flags.map(lambda f: flag in list(f))]
    print(f"\n\n################ {flag}: {len(g)} questions")
    for rv in pick(g, N_RANDOM, SEED).question_revid:
        rand_ok.append(show_question(int(rv)))
        if flag == "question_merged_into_reply":
            r = qi.loc[rv]; it, cs = raw_thread(next(qi.loc[[rv]].itertuples()))
            t0 = turns[rv][0]
            raw0 = html_text(next((c["html"] for c in (cs or []) if c["id"] == t0["comment_id"]), ""))
            qw = set(re.findall(r"\w+", r.question_text.lower()))
            starts = set(re.findall(r"\w+", raw0.lower()[:len(r.question_text) + 50])) >= qw
            print(f"   raw first turn starts with the question words: {starts}")
            print(f"   stored reply text no longer contains the question start: {r.question_text[:40].lower() not in (t0['text'] or '').lower()}")
'''),
md("### 5.E Entry point: asked while editing a page"),
code(r'''
for r in pick(q[q.editing_page], N_RANDOM, SEED).itertuples():
    m = HEAD_PAGE.match(nhead(r.heading))
    print(f"  {r.question_revid}  tag={r.tag!r}\n     heading: {r.heading!r}\n     page in heading: {m.group(1) if m else None!r}  |  stored article_context: {r.article_context!r}")
for r in pick(q[~q.editing_page], 3, SEED).itertuples():
    print(f"  not editing: {r.question_revid} tag={r.tag!r} heading={r.heading!r}")
'''),
md("### 5.D Question deleted in the next edit: raw edit next to removal_kind"),
code(r'''
rem = q[q.removal_kind.notna()]
print(rem.removal_kind.value_counts().to_string())
for kind, g in rem.groupby("removal_kind"):
    for r in pick(g, 2, SEED).itertuples():
        e = removals.get(str(r.question_revid)) or {}
        print(f"\n  [{kind}] question {r.question_revid} by {r.mentee_name} (uid {r.mentee_uid})  next edit {diff(e.get('revid', 0))}")
        print(f"     RAW edit: user={e.get('user')!r} userid={e.get('userid')} tags={e.get('tags')} comment={(e.get('comment') or '')[:100]!r}")
        compare({"removal_kind": my_removal(e, r)}, {"removal_kind": r.removal_kind})
'''),
md("## 6. Edge cases (every class is written in full to `_issues/`; here are counts and examples)"),
code(r'''
summ = json.loads((M2 / "_issues" / "summary.json").read_text())
print(json.dumps({k: v["n"] for k, v in summ.items()}, indent=1))
flags = q.explode("parse_flags").parse_flags.value_counts()
print("\nparse_flags:\n" + flags.to_string())
'''),
code(r'''
for f in ["thread_not_found", "reply_before_question", "first_turn_not_mentee", "ambiguous_thread", "turn_after_cutoff_dropped",
          "page_owner_differs", "unrecoverable_revdel", "primary_mentor_unresolved", "turn_source_mismatch"]:
    g = q[q.parse_flags.map(lambda x: f in list(x))]
    print(f"\n## {f}: {len(g)} (random examples)")
    for r in pick(g, N_RANDOM, SEED).itertuples():
        print(f"  {r.question_revid}  {r.q_ts}  {r.page_eval}  {r.heading!r:.90}  {oldid(r.question_revid)}")
        for t in turns.get(r.question_revid, [])[:4]:
            print(f"       {t['ts']}  {t['role']:<14} {t['author']}")
'''),
code(r'''
# turn splitting: eval2's own signature regex versus DiscussionTools, random examples side by side
p = M2 / "_issues" / "turn_source_mismatch.jsonl"
mm = pd.DataFrame(read_jsonl(p)) if p.exists() else pd.DataFrame()
print("questions where the two disagree:", len(mm))
def others(row, side):
    mentee = {(qi.loc[row.ids["revid"], "mentee_name"] or "").lower()}
    return [a for a in row.detail[side] if a and a.lower() not in mentee]
if len(mm):
    mm["other_differs"] = [others(r, "regex") != others(r, "dt") for r in mm.itertuples()]
    print(mm.other_differs.value_counts().rename({False: "only the newcomer's own turn count differs", True: "speakers other than the newcomer differ"}).to_string())
    for r in pick(mm[mm.other_differs], 8, SEED).itertuples():
        print(f"\n  {r.ids['revid']}  {r.wiki_url}\n    regex: {r.detail['regex']}\n    dt   : {r.detail['dt']}")
'''),
code(r'''
# first reply in thread order is not the earliest reply in time (the stored first_reply_ts follows thread order)
order = []
for r in q[q.has_any_reply == True].itertuples():   # noqa: E712
    rep = [t for t in turns[r.question_revid] if t["role"] != "mentee" and t["ts"] and r.q_ts[:17] + "00Z" <= t["ts"] < CUTOFF]
    if min(t["ts"] for t in rep) != rep[0]["ts"]:
        order.append((r.question_revid, rep[0]["ts"], min(t["ts"] for t in rep), rep[0]["author"], oldid(r.question_revid)))
print(f"questions where the first reply in thread order is later than another reply: {len(order)}")
display(pd.DataFrame(order, columns=["revid", "first_reply_ts (stored, thread order)", "earliest reply ts", "first replier", "link"]).head(25))
'''),
code(r'''
# article_context (from the heading HTML link) versus the page named in the heading text
x = q[q.editing_page].copy()
x["page_in_heading"] = x.heading.map(lambda h: (HEAD_PAGE.match(nhead(h)) or [None, None])[1])
d = x[x.page_in_heading != x.article_context.fillna("").str.lower()]
print(f"editing questions where article_context differs from the heading text (case ignored): {len(d)}")
for r in d.head(15).itertuples():
    it, _ = raw_thread(r)
    links = re.findall(r'title="([^"]+)"', (it or {}).get("html", ""))[:3]
    print(f"  {r.question_revid}: heading page {r.page_in_heading!r} -> stored {r.article_context!r}   raw heading HTML links: {links}")
'''),
code(r'''
# leftover wiki markup in the cleaned question text (worth a look; usually markup inside <nowiki> or broken templates)
w = q[q.question_text_source == "wikitext"]
for pat, label in [(r"\{\{", "{{"), (r"\[\[", "[["), (r"\(UTC\)", "(UTC)"), (r"<[a-z]+[\s>/]", "<html tag")]:
    hit = w[w.question_text.str.contains(pat)]
    print(f"\nquestion_text containing {label!r}: {len(hit)}")
    for r in pick(hit, 2, SEED).itertuples():
        print(f"   {r.question_revid}: {r.question_text[:160]!r}")
# fallback threads recovered for a different question (the same deleted section can hold two questions)
fbq = q[q.thread_origin == "fallback"]
other = fbq[fbq.apply(lambda r: fb_map.get(str(r.question_revid), {}).get("revid") != r.thread_revid, axis=1)]
print(f"\nfallback threads taken from a revision recovered for another question: {len(other)} of {len(fbq)}")
'''),
md("## 7. Distributions: questions per month and reply rate (there should be nothing after the cutoff)"),
code(r'''
m = q.assign(month=pd.to_datetime(q.q_ts, utc=True).dt.strftime("%Y-%m")).groupby("month").agg(
    questions=("question_revid", "size"), reply_rate=("has_any_reply", "mean"), mentor_reply_rate=("has_mentor_reply", "mean"))
display(m.tail(8))
ax = m.questions.plot(figsize=(10, 3), title="Questions per month"); ax.axvline(len(m) - 1, color="red", ls="--")
m[["reply_rate", "mentor_reply_rate"]].plot(figsize=(10, 3), title="Reply rate by question month", ylim=(0, 1))
lag = (pd.to_datetime(q.first_reply_ts, utc=True) - qts).dt.total_seconds() / 3600
print("hours from question to first reply:\n" + lag.describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
print("\nturns per thread:\n" + q[found].n_turns.value_counts().sort_index().head(12).to_string())
'''),
md("## 8. Summary (written to results/m2_report.json, read by M9)"),
code(r'''
report = {"n_questions": len(q), "n_topup": int((q.source == "topup").sum()),
          "thread_found_rate": round(float(q.thread_page.notna().mean()), 4),
          "reply_rate": round(float(q.has_any_reply.mean()), 4),
          "mentor_reply_rate": round(float(q.has_mentor_reply.mean()), 4),
          "n_away_routed": int(q.away_routed.sum()), "n_editing_page": int(q.editing_page.sum()),
          "first_replier_role": q.first_replier_role.value_counts().to_dict(),
          "flags": {k: int(v) for k, v in flags.items()}}
(config.RESULTS / "m2_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
print(f"\nchecks passed: {sum(ok for _, ok in ck.items)} / {len(ck.items)}; "
      f"tracer and random questions fully matching: {sum(tracer_ok) + sum(rand_ok)} / {len(tracer_ok) + len(rand_ok)}")
'''),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "m2_check.ipynb"
    write_notebook(CELLS, out)
    print("wrote", out)
    if "--run" in sys.argv:
        run_as_script(out, cwd=out.parent.parent)
        print("\nnotebook ran end to end")


if __name__ == "__main__":
    main()
