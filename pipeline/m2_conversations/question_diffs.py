"""
question_diffs.py: M2 part 0: the text of each question exactly as posted, taken from the diff of its tagged revision

Module: M2 (PLAN.md M2, I-60)

1. What it does
    For the tagged question revisions whose collected text is obviously broken (suspects(): the text is wrapped in a
    collapse template such as {{hat}}, nothing is left after removing signatures and templates, or the plain text
    is only the asker's name; text that the community later replaced with {{redacted}} is left alone), asks the API for the diff between the revision and its parent
    (action=compare) and keeps the lines that this edit added, starting at the "== Question from ... ==" heading and
    ending at the first line the edit did not add. The result is the question wikitext at the moment it was asked.
    build_questions.py uses this text when it is available, instead of the section text collected earlier.

2. Why it is designed this way
    The question text collected first (eval2) was cut out of a page version from some time after the question. In
    about 1% of a random sample of 300 it did not match the text as posted: someone had collapsed the section later
    ({{hat}}), the newcomer had edited it, or the newcomer had put their own "== ... ==" heading inside the question
    so that the "Question from" section looked empty. The diff of the tagged revision is the record of exactly what
    was posted. Fetching it for every question would take about 12 hours of API time for a 1% error that barely
    moves the covariates, so only the cases that can be seen from the text alone are fixed; the remaining rate is
    recorded in PLAN.md (I-60).

3. Inputs
    The tagged revisions (the same list build_questions.load_base reads).

4. Outputs
    data/raw/m2/question_text_from_diff.json: {revid: {"text": wikitext, "heading_found": bool, "n_lines": int}}
    Diffs that the API could not return (revision deleted or hidden) are left out; build_questions then keeps the
    text collected earlier.

6. Possible errors and how they are handled
    - No "Question from" heading among the added lines (the question was added without the usual heading): the
      first contiguous block of added lines is used and heading_found is False.
    - A hidden or deleted revision: the API returns an error; the revision is left out and counted in the log.

7. How to run
    python -m pipeline.m2_conversations.question_diffs --online --phase collect    # about 200 requests
    python -m pipeline.m2_conversations.question_diffs --phase build               # offline, from the API cache

8. How to check
    tests/test_m2.py::test_question_from_diff; the "question text from diff" cells of checks/m2_check.ipynb.
"""
from __future__ import annotations

import html
import re
import sys
import time

import pandas as pd

import config
from pipeline.common.api import ApiError, MediaWikiClient, OfflineCacheMiss
from pipeline.common.io import write_json_atomic
from pipeline.common.issues import IssueLog
from pipeline.common.runlog import base_parser, setup_logging

RAW = config.RAW / "m2"
OUT = RAW / "question_text_from_diff.json"
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_ADDED = re.compile(r'<td class="diff-addedline[^"]*">(.*?)</td>', re.S)
_HEADING = re.compile(r"^\s*==\s*Question from\b", re.I)


def _cell_text(cell: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", cell))


def diff_rows(body: str) -> list[tuple[str, str]]:
    """The rows of a MediaWiki diff table as (kind, text): kind is "added" (the row has an added line; its text is
    the new line), "context", "lineno" (a jump to another part of the page) or "deleted"."""
    rows = []
    for tr in _ROW.findall(body or ""):
        a = _ADDED.search(tr)
        if a:
            rows.append(("added", _cell_text(a.group(1))))
        elif "diff-lineno" in tr:
            rows.append(("lineno", ""))
        elif "diff-context" in tr:
            rows.append(("context", ""))
        else:
            rows.append(("deleted", ""))
    return rows


def question_from_diff(body: str) -> dict | None:
    """The question wikitext from one diff: the added lines after the "Question from" heading, up to the first row
    that is not an added line. Returns None if the edit added nothing."""
    rows = diff_rows(body)
    start = next((i for i, (k, t) in enumerate(rows) if k == "added" and _HEADING.match(t)), None)
    found = start is not None
    if start is None:
        start = next((i - 1 for i, (k, _) in enumerate(rows) if k == "added"), None)
        if start is None:
            return None
    lines = []
    for k, t in rows[start + 1:]:
        if k == "deleted":                     # a line removed by the same edit inside the block: skip it
            continue
        if k != "added":
            break
        lines.append(t)
    return {"text": "\n".join(lines).strip(), "heading_found": found, "n_lines": len(lines)}


_COLLAPSE = re.compile(r"\{\{\s*(?:hat|hidden archive top|collapse top|cot|archive top|atop)\b", re.I)


def _letters(s: str | None) -> str:
    return re.sub(r"[\W_]+", "", (s or "").lower())


def suspects(base: pd.DataFrame) -> pd.Series:
    """True for questions whose collected text is obviously broken (see section 1)."""
    from pipeline.common.textclean import to_plain_text
    wt = base.question_wikitext.fillna("")
    hat = wt.map(lambda t: bool(_COLLAPSE.search(t)))
    body = (wt.str.replace(r"\[\[User(?: talk)?:[^\]]*\]\]", "", regex=True)
              .str.replace(r"\{\{[^{}]*\}\}", "", regex=True)
              .str.replace(r"\(UTC\)|\d{1,2}:\d{2}, \d{1,2} \w+ \d{4}|--", "", regex=True))
    empty = body.map(_letters) == ""
    name_only = [_letters(to_plain_text(t)).replace(_letters(n), "") == "" for t, n in zip(wt, base.mentee_name)]
    # Text that the community removed on purpose ({{redacted}}) is not restored, even though the diff still shows it.
    redacted = wt.str.contains(r"\{\{\s*redacted", case=False, regex=True)
    return (hat | empty | pd.Series(name_only, index=base.index)) & ~redacted


def targets(log) -> pd.DataFrame:
    """The suspect revisions (see suspects()) with their parent ids."""
    from pipeline.m2_conversations.build_questions import load_base
    base = load_base(log, IssueLog("m2_question_diffs"), use_diff=False)
    base = base[suspects(base)][["question_revid", "parentid"]].dropna()
    order = pd.Series(1, index=base.question_revid)
    sp = config.INTERIM / "m5" / "sample.parquet"
    if sp.exists():
        first = set(pd.read_parquet(sp, columns=["first_question_revid"]).first_question_revid)
        order[order.index.isin(first)] = 0
    base["order"] = base.question_revid.map(order)
    return base.sort_values(["order", "question_revid"]).reset_index(drop=True)


def params(revid: int, parentid: int) -> dict:
    return {"action": "compare", "fromrev": int(parentid), "torev": int(revid), "prop": "diff"}


def collect(client: MediaWikiClient, todo: pd.DataFrame, log) -> None:
    t0, err = time.time(), 0
    for i, r in enumerate(todo.itertuples(), 1):
        try:
            client.get(params(r.question_revid, r.parentid))
        except ApiError:
            err += 1
        if i % 500 == 0:
            rate = i / (time.time() - t0)
            log.info("  %d/%d diffs (%d API errors), about %.1f h left", i, len(todo), err, (len(todo) - i) / rate / 3600)
    log.info("collected %d diffs, %d API errors", len(todo), err)


def build(client: MediaWikiClient, todo: pd.DataFrame, log) -> None:
    out, miss, err, nohead = {}, 0, 0, 0
    for r in todo.itertuples():
        try:
            d = client.get(params(r.question_revid, r.parentid))
        except OfflineCacheMiss:
            miss += 1
            continue
        except ApiError:
            err += 1
            continue
        res = question_from_diff((d.get("compare") or {}).get("body", ""))
        if res is None:
            continue
        nohead += not res["heading_found"]
        out[str(int(r.question_revid))] = res
    write_json_atomic(OUT, out)
    log.info("question text from diff: %d of %d revisions (not available %d, API errors %d, no Question heading %d)",
             len(out), len(todo), miss, err, nohead)


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("M2: question text as posted, from the diff of each tagged revision.")
    ap.add_argument("--phase", choices=["collect", "build"], required=True)
    args = ap.parse_args(argv)
    log = setup_logging(f"m2_question_diffs_{args.phase}")
    todo = targets(log)
    if args.limit:
        todo = todo.head(args.limit)
    client = MediaWikiClient(online=args.online)
    if args.phase == "collect":
        collect(client, todo, log)
    else:
        build(client, todo, log)


if __name__ == "__main__":
    main(sys.argv[1:])
