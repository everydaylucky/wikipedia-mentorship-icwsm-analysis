"""
build_eligibility.py: M4 assignment eligibility: was each asker's mentor assigned at random at registration? If
not, give the reason for exclusion

Module: M4 (PLAN.md M4; CODE_PLAN.md section 5, M4)
Rules: DATA_PROCESSING R2.1 to R2.5 (account type and registration time, decision D10), R3.1 to R3.5 (manual
claims), section 8 (identification assumptions)

1. What it does
    For each asker (mentee_uid from M2), check the conditions below in order. The first one that holds is the
    exclusion reason (exclude_reason). All reasons that hold are also stored in exclude_reasons:
        1. temporary_account: a temporary account (user name starts with "~2", R2.2).
        2. How the account was created (R2.2, D10): autocreate -> autocreated_account; create2, byemail ->
           created_by_other; forcecreatelocal -> forced_local_account; no creation log -> no_creation_log.
        3. registered_before_mentorship: registered before 2021-05-04 (R2.4, D10).
        4. alt_or_test_account: when the account was claimed, the comment says it is the mentor's own alt account
           or test account (R3.4), no matter when.
        5. manual_claim_before_question: claimed by hand before the first question (R3.1, R3.3).
    Other outputs:
        eligible_unrestricted: whether the asker is eligible without the D10 limits (items 2 and 3). Used for
            robustness checks (close to the paper version's definition).
        claimed_after_question: claimed by hand only after the first question (for the R3.3 robustness check).
        Diagnostic fields mentor_on_roster_at_registration, mentor_in_auto_pool_at_registration,
        assigned_later, possible_forcementor (section 8; for diagnosis only, not used to exclude anyone).

2. Why it is designed this way
    - At the moment of registration, GrowthExperiments uses the LocalUserCreated hook to draw a mentor at random
      (weighted) from the auto-assignment pool. Accounts auto-created from another wiki do not trigger assignment,
      and for accounts created by someone else the assignment is not guaranteed to be random (DATA_PROCESSING
      section 2). So "registered by themselves, and after the mentorship program launched" is the working
      definition of "was assigned a mentor at random at registration" (R2.5).
    - A manual claim (claimmentee) whose comment matches a system template for bulk transfer ("$1 retired from
      mentorship.", "$2 removed $1 from mentorship", GrowthExperiments i18n) is a random transfer after the
      mentor left. Otherwise it is a mentor claiming the newcomer on purpose. If that claim happened before the
      question, the mentor is not random (R3.1, R3.3).
    - The claim log names newcomers by user name, so we first turn names into uids (R3.5): use the names already
      known from M2 and M3. For names that still do not match, follow the rename log to find the current name.

3. Inputs
    data/interim/m2/questions.parquet (asker, time of first question, primary mentor);
    data/interim/m3/mentee_profile.parquet (registration time, how the account was created, current name);
    data/interim/m1/mentor_periods.parquet (timeline of the mentor list); config.RAW_FILES["claim_log"] (claim log).

4. Outputs (data/interim/m4/)
    eligibility.parquet   one row per asker, primary key mentee_uid
    claims.parquet        all claim log entries before the cutoff: logid, ts, mentee_name, mentee_uid, claimer,
                          kind (bulk/manual), alt_or_test
    _issues/              claim_name_unresolved, manual_claim_review (comments of all manual claims, to read through
                          by hand), and others

6. Possible errors and how they are handled
    - A newcomer name in the claim log does not match any uid: record issue `claim_name_unresolved`. These claims
      are not used in the decision (they are listed for checking by hand).
    - A comment mentions test/alt but means something else: every comment judged to be an alt or test account is
      listed in issue `alt_or_test_review` for checking by hand.

7. How to run
    python -m pipeline.m4_eligibility.build_eligibility           # offline; rename lookups use the API cache
    python -m pipeline.m4_eligibility.build_eligibility --online  # rename lookups missing from the cache are fetched online

8. How to check
    checks/m4_check.ipynb; tests/test_m4.py.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import time

import pandas as pd

import config
from pipeline.common.api import MediaWikiClient, OfflineCacheMiss
from pipeline.common.checks import assert_unique
from pipeline.common.io import write_table
from pipeline.common.issues import IssueLog
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.timeutil import before_cutoff, parse_iso
from pipeline.common.users import norm_user

OUT = config.INTERIM / "m4"
TEMP_PREFIX = "~2"
CLAIM_ACTIONS = ("claimmentee", "claimmentee-no-previous-mentor")

# bulk transfer templates (R3.1): replace $1 and $2 with "any name"
_BULK_RES = [re.compile("^" + re.escape(t).replace(r"\$1", ".+?").replace(r"\$2", ".+?") + r"\.?$", re.I)
             for t in config.REASSIGN_TEMPLATES]
# alt or test accounts (R3.4): only the mentor's (the claimer's) own alt accounts, bots, or test accounts.
# Both conditions must hold: the comment mentions alt/sock/bot/test, and there is evidence that the account is
# the claimer's own. If the comment only says test, or talks about someone else's alt account ("alt acct for X",
# "Claiming alt accounts to unify response"), it is treated as an ordinary manual claim.
_ALT_TEST_RE = re.compile(r"\b(?:alt|alts|alternate|alternative|sock|socks|bot|test\w*|testalt)\b", re.I)
_OWN_RE = re.compile(r"\b(?:my|mine|own|myself|personal|self)\b", re.I)


def classify_claim(comment: str | None, claimer: str | None = None, previous_mentor: str | None = None) -> str:
    """The type of claim (R3.1):
        "bulk": the comment matches a bulk transfer template (the system transfers newcomers at random after a
                mentor leaves or is removed).
        "self_transfer": the claimer and the previous mentor are the same person (the mentor moves the newcomer
                from their main account to their own mentor-only alt account, e.g. LuniZunie -> LuniZunie-Mentor).
                The mentor did not really change, so this is not a manual claim.
        "manual": anything else; the mentor claimed the newcomer on purpose.
    """
    c = (comment or "").strip()
    if any(r.match(c) for r in _BULK_RES):
        return "bulk"
    a, b = _stem(claimer), _stem(previous_mentor)
    if a and b and (a == b or (min(len(a), len(b)) >= 4 and (a in b or b in a))):
        return "self_transfer"
    return "manual"


def _stem(name: str | None) -> str:
    """The core of a name: lowercase, with text in parentheses, "(WMF)", spaces, and punctuation removed. Used to decide whether two accounts belong to the same person."""
    n = re.sub(r"\(.*?\)", "", (norm_user(name) or "")).lower()
    return re.sub(r"[^a-z0-9\u0080-\uffff]", "", n)


def is_alt_or_test(comment: str | None, mentee_name: str | None, claimer: str | None) -> bool:
    """R3.4: the claimed account is the claimer's own alt account, bot, or test account.

    Condition: the comment mentions alt/sock/bot/test, and at least one of these holds: the claimer claimed
    their own account; the core of one name contains the other (for example LuniZunie and LuniZunie-Mentor;
    Nick Moyes and NM Demo 2 do not count this way and are caught by "my alt-account" in the comment); the
    comment contains my/own/personal/self.
    """
    c = comment or ""
    if not _ALT_TEST_RE.search(c):
        return False
    m, k = _stem(mentee_name), _stem(claimer)
    same = bool(m) and m == k
    related = len(k) >= 4 and len(m) >= 4 and (k in m or m in k)
    return same or related or bool(_OWN_RE.search(c))


def load_claims(issues: IssueLog) -> pd.DataFrame:
    """Claim log entries before the cutoff, one row per entry."""
    rows = []
    for e in json.loads(config.RAW_FILES["claim_log"].read_text()):
        if e.get("action") not in CLAIM_ACTIONS or not before_cutoff(e.get("timestamp")):
            continue
        params = e.get("params") or {}
        if isinstance(params, str):
            params = ast.literal_eval(params)
        name = norm_user((e.get("title") or "").split(":", 1)[-1])
        rows.append({"logid": int(e["logid"]), "ts": e["timestamp"], "action": e["action"], "mentee_name": name,
                     "claimer": norm_user(e.get("user")), "claimer_uid": int(e["userid"]) if e.get("userid") else None,
                     "previous_mentor": norm_user(params.get("previous-mentor")), "comment": e.get("comment") or ""})
    df = pd.DataFrame(rows)
    df["kind"] = [classify_claim(r.comment, r.claimer, r.previous_mentor) for r in df.itertuples()]
    df["alt_or_test"] = [is_alt_or_test(r.comment, r.mentee_name, r.claimer) for r in df.itertuples()]
    return df


def name_index(prof: pd.DataFrame, q: pd.DataFrame) -> dict[str, int]:
    """Known "name -> uid" pairs: current names and names at download time from M3, and authors of question revisions from M2. Names that point to more than one uid are left out."""
    pairs = list(zip(prof.current_name, prof.mentee_uid)) + list(zip(prof.name_at_fetch, prof.mentee_uid)) \
        + list(zip(q.mentee_name, q.mentee_uid))
    idx: dict[str, set] = {}
    for n, u in pairs:
        n = norm_user(n)
        if n and pd.notna(u):
            idx.setdefault(n, set()).add(int(u))
    return {n: next(iter(us)) for n, us in idx.items() if len(us) == 1}


def resolve_claim_uids(claims: pd.DataFrame, idx: dict[str, int], client, issues: IssueLog) -> pd.Series:
    """Newcomer names in the claim log -> uid (R3.5).

    1. Known names (M2, M3) are matched directly.
    2. The other names are looked up as current names with list=users, in batches of 50.
    3. Names still not found (the user was renamed and the old name no longer exists) are traced through the
       rename log to the current name.
    """
    from pipeline.m1_roster.collect_roster import resolve_name
    out = {n: idx[n] for n in claims.mentee_name.unique() if n in idx}
    rest = [n for n in claims.mentee_name.unique() if n not in out]
    for i in range(0, len(rest), 50):
        batch = rest[i:i + 50]
        try:
            users = client.get({"action": "query", "list": "users", "ususers": "|".join(batch)})["query"]["users"]
        except OfflineCacheMiss:
            users = []
        for u in users:
            if u.get("userid"):
                out[norm_user(u["name"])] = int(u["userid"])
    for n in [n for n in rest if n not in out]:
        try:
            uid, path = resolve_name(client, n)
        except OfflineCacheMiss:
            uid, path = None, [n, "(offline: not in cache)"]
        out[n] = uid
        if uid is None:
            issues.add("claim_name_unresolved", rule="R3.5", action="flagged", ids={"name": n}, detail={"path": path})
    return claims.mentee_name.map(out)


def mentor_state(periods: pd.DataFrame, uid, at) -> tuple[bool, bool]:
    """For mentor uid at time at: (on the mentor list?, in the auto-assignment pool?)."""
    if pd.isna(uid) or at is None:
        return False, False
    p = periods[periods.mentor_uid == int(uid)]
    on = p[(p.start_dt <= at) & (p.end_dt.isna() | (at < p.end_dt))]
    return (not on.empty), bool((on.in_auto_pool == True).any())   # noqa: E712


def decide(asker: dict) -> tuple[str | None, list[str], bool]:
    """Following the order in section 1 above, return (exclude_reason, all reasons, eligible_unrestricted)."""
    reasons = []
    name = asker.get("current_name") or asker.get("mentee_name") or ""
    if name.startswith(TEMP_PREFIX):
        reasons.append("temporary_account")
    act = asker.get("creation_action")
    if act is None:
        reasons.append("no_creation_log")
    elif act not in config.ELIGIBLE_CREATION_ACTIONS:
        reasons.append(config.CREATION_EXCLUDE_REASON.get(act, f"creation_{act}"))
    reg = parse_iso(asker.get("registration"))
    if reg is None or reg < config.MENTORSHIP_START:
        reasons.append("registered_before_mentorship")
    if asker["alt_or_test"]:
        reasons.append("alt_or_test_account")
    if asker["manual_claim_before_q"]:
        reasons.append("manual_claim_before_question")
    d10 = {"no_creation_log", "registered_before_mentorship", "autocreated_account", "created_by_other",
           "forced_local_account"} | {r for r in reasons if r.startswith("creation_")}
    unrestricted = not [r for r in reasons if r not in d10]
    return (reasons[0] if reasons else None), reasons, unrestricted


def build(log, issues: IssueLog, client) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet")
    q = q[q.mentee_uid.notna()].sort_values("q_ts")
    first = q.groupby("mentee_uid").head(1).set_index("mentee_uid")
    prof = pd.read_parquet(config.INTERIM / "m3" / "mentee_profile.parquet").set_index("mentee_uid")
    periods = pd.read_parquet(config.INTERIM / "m1" / "mentor_periods.parquet")
    periods["start_dt"] = pd.to_datetime(periods.start, utc=True)
    periods["end_dt"] = pd.to_datetime(periods.end, utc=True)

    claims = load_claims(issues)
    claims["mentee_uid"] = resolve_claim_uids(claims, name_index(prof.reset_index(), q), client, issues).astype("Int64")
    for r in claims[claims.kind == "manual"].itertuples():
        issues.add("manual_claim_review", rule="R3.1", action="flagged", ids={"logid": r.logid, "mentee": r.mentee_name},
                   detail={"claimer": r.claimer, "ts": r.ts, "comment": r.comment, "alt_or_test": r.alt_or_test},
                   url=f"https://en.wikipedia.org/w/index.php?title=Special:Log&logid={r.logid}")
    for r in claims[claims.alt_or_test].itertuples():
        issues.add("alt_or_test_review", rule="R3.4", action="flagged", ids={"logid": r.logid, "mentee": r.mentee_name},
                   detail={"claimer": r.claimer, "comment": r.comment, "kind": r.kind})
    log.info("claims before cutoff: %d (%s); alt/test %d; unresolved names %d", len(claims),
             claims.kind.value_counts().to_dict(), int(claims.alt_or_test.sum()), int(claims.mentee_uid.isna().sum()))

    by_uid = {int(u): g for u, g in claims.dropna(subset=["mentee_uid"]).groupby("mentee_uid")}
    rows = []
    for uid, f in first.iterrows():
        uid = int(uid)
        p = prof.loc[uid] if uid in prof.index else pd.Series(dtype=object)
        g = by_uid.get(uid)
        q1 = parse_iso(f.q_ts)
        manual = g[g.kind == "manual"] if g is not None else None
        before = manual is not None and any(parse_iso(t) < q1 for t in manual.ts)
        after = manual is not None and any(parse_iso(t) >= q1 for t in manual.ts)
        alt = g is not None and bool(g.alt_or_test.any())
        a = {"current_name": p.get("current_name"), "mentee_name": f.mentee_name,
             "creation_action": p.get("creation_action"), "registration": p.get("registration"),
             "alt_or_test": alt, "manual_claim_before_q": before}
        reason, reasons, unrestricted = decide(a)
        reg = parse_iso(p.get("registration"))
        on, auto = mentor_state(periods, f.primary_mentor_uid, reg)
        explained = g is not None and (g.claimer_uid == f.primary_mentor_uid).any()
        rows.append({"mentee_uid": uid, "first_question_revid": int(f.question_revid), "first_q_ts": f.q_ts,
                     "registered_at": p.get("registration"), "creation_action": p.get("creation_action"),
                     "eligible": reason is None, "exclude_reason": reason, "exclude_reasons": reasons,
                     "eligible_unrestricted": unrestricted,
                     "claimed_after_question": after, "n_claims": 0 if g is None else len(g),
                     "n_manual_claims": 0 if manual is None else len(manual),
                     "primary_mentor_uid": f.primary_mentor_uid,
                     "mentor_on_roster_at_registration": on, "mentor_in_auto_pool_at_registration": auto,
                     "assigned_later": (reg is not None) and not on,
                     "possible_forcementor": bool(on and not auto and reg is not None and reg >= config.FORCEMENTOR_LIVE
                                                  and not explained)})
    elig = pd.DataFrame(rows)
    elig["primary_mentor_uid"] = elig.primary_mentor_uid.astype("Int64")
    return elig, claims


def main(argv: list[str] | None = None) -> None:
    args = base_parser("M4: assignment eligibility per asker.").parse_args(argv)
    log = setup_logging("m4")
    t0 = time.time()
    check_upstream("m4", ["m1", "m2", "m3"])
    issues = IssueLog("m4")
    client = MediaWikiClient(online=args.online)
    elig, claims = build(log, issues, client)
    assert_unique(elig, "mentee_uid")
    OUT.mkdir(parents=True, exist_ok=True)
    write_table(OUT / "eligibility.parquet", elig)
    write_table(OUT / "claims.parquet", claims)
    issues.write()
    counts = {"askers": len(elig), "eligible": int(elig.eligible.sum()),
              "eligible_unrestricted": int(elig.eligible_unrestricted.sum()),
              "exclude_reason": elig.exclude_reason.value_counts().to_dict(),
              "claimed_after_question": int(elig.claimed_after_question.sum()),
              "assigned_later": int(elig.assigned_later.sum()), "possible_forcementor": int(elig.possible_forcementor.sum())}
    log.info("done: %s", counts)
    write_manifest("m4", inputs=[config.RAW_FILES["claim_log"]], outputs=[OUT / "eligibility.parquet", OUT / "claims.parquet"],
                   params={"cutoff": config.DATA_CUTOFF_ISO}, counts=counts, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
