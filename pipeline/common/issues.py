"""
issues.py: problem records (IssueLog)

Module: common layer (CODE_PLAN.md sections 4 and 6)

1. What it does
    While data is processed, it records every problem found: what type it is,
    which ids are involved, what the raw evidence is, how it was handled, and
    where on Wikipedia to check it. When a module ends, the records are written
    by type to data/interim/<module>/_issues/<type>.jsonl, and a summary is made.

2. Why it is designed this way
    When the old code met a problem, it either skipped it silently or put in a
    default value (for example, when a reply was earlier than the question, it
    used a pseudo start point instead). There was no way to check this later.
    Here, every type of problem must have its handling decided in advance
    (fixed, flagged, quarantined, or fatal; see CODE_PLAN 6.1). Each record also
    carries a piece of the raw text and a Wikipedia link you can open directly.
    The check notebook shows a sample, and a person compares it with the real
    page to confirm the handling was right.

3. Inputs
    The caller adds records one by one with add(...).

4. Outputs
    _issues/<type>.jsonl: one file per type;
    _issues/summary.json: for each type, the count, the handling, and the rule ID.

6. Possible errors and how they are handled
    action must be one of fixed / flagged / quarantined / fatal; otherwise ValueError.
    add(..., action="fatal") records the problem and then raises FatalIssue at
    once, which stops the run.

8. How to check
    The "problem list" section of each check notebook reads these files.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from pipeline.common.io import write_json_atomic, write_jsonl_atomic

ACTIONS = ("fixed", "flagged", "quarantined", "fatal")
EVIDENCE_MAX_CHARS = 500


class FatalIssue(RuntimeError):
    """A condition that cannot be tolerated was broken (for example, a record after the cutoff date), so the run must stop."""


def wiki_url(*, revid: int | None = None, title: str | None = None,
             logid: int | None = None, uid: int | None = None) -> str | None:
    """Build a Wikipedia link that opens directly in a browser, to check the raw record.

    Priority: revision ID > log ID > page title > user ID.
    """
    base = "https://en.wikipedia.org"
    if revid:
        return f"{base}/w/index.php?oldid={int(revid)}"
    if logid:
        return f"{base}/wiki/Special:Log?logid={int(logid)}"
    if title:
        return f"{base}/wiki/{title.replace(' ', '_')}"
    if uid:
        return f"{base}/wiki/Special:Redirect/user/{int(uid)}"
    return None


@dataclass
class IssueLog:
    """Problem recorder for one module.

    Usage:
        log = IssueLog("m2")
        log.add("reply_before_question", rule="R5.4", action="quarantined",
                ids={"question_revid": 1215863591}, evidence=section_text[:500],
                detail={"q_ts": "...", "reply_ts": "..."})
        ...
        log.write()   # call when the module ends
    """
    module: str
    records: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    meta: dict[str, dict] = field(default_factory=dict)

    def add(self, type_: str, *, rule: str = "", action: str = "flagged",
            ids: dict[str, Any] | None = None, evidence: str | None = None,
            detail: dict[str, Any] | None = None, url: str | None = None) -> None:
        """Record one problem.

        Args:
            type_: problem type, for example "reply_before_question". A type has the same meaning across the whole project.
            rule: the related rule or issue ID, for example "R5.4" or "I-45".
            action: fixed (corrected), flagged (marked), quarantined (set aside), fatal (stop with an error).
            ids: related ids, for example {"question_revid": ..., "mentee_uid": ...}.
            evidence: a piece of the raw text, cut to 500 characters.
            detail: other details, for example the values before and after a fix.
            url: link for checking; if not given, it is built from the revid / logid / uid in ids.

        Raises:
            ValueError: action is not valid.
            FatalIssue: action == "fatal"; raised right after the record is saved.
        """
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        ids = ids or {}
        if url is None:
            url = wiki_url(revid=ids.get("question_revid") or ids.get("revid"),
                           logid=ids.get("logid"), uid=ids.get("mentee_uid") or ids.get("uid"))
        rec = {"type": type_, "module": self.module, "rule": rule, "action": action,
               "ids": ids, "evidence": (evidence or "")[:EVIDENCE_MAX_CHARS],
               "detail": detail or {}, "wiki_url": url}
        self.records[type_].append(rec)
        self.meta.setdefault(type_, {"rule": rule, "actions": Counter()})["actions"][action] += 1
        if action == "fatal":
            self.write()
            raise FatalIssue(f"[{self.module}] {type_}: {ids} {detail or ''}")

    def count(self, type_: str | None = None) -> int:
        """Number of problems of one type (or of all types)."""
        if type_ is not None:
            return len(self.records.get(type_, []))
        return sum(len(v) for v in self.records.values())

    def write(self, directory: Path | None = None) -> Path:
        """Write out all problem records and the summary, and return the folder path."""
        d = directory or config.issues_dir(self.module)
        for type_, recs in self.records.items():
            write_jsonl_atomic(d / f"{type_}.jsonl", recs)
        summary = {t: {"n": len(self.records[t]), "rule": m["rule"], "actions": dict(m["actions"])}
                   for t, m in self.meta.items()}
        write_json_atomic(d / "summary.json", summary)
        return d

    def summary_lines(self) -> list[str]:
        """Summary for printing to the log, one line per type."""
        return [f"  {t:40s} {len(self.records[t]):>7,}  {self.meta[t]['rule']:6s} "
                f"{dict(self.meta[t]['actions'])}" for t in sorted(self.records)]
