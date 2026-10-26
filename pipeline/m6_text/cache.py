"""
cache.py: a cache for results from outside text services (LLM labels, Perspective, embeddings)

Module: M6 (CODE_PLAN.md section 5, M6)

1. What it does
    TextCache: SQLite table results(key PRIMARY KEY, task, model, version, text_sha, value, source, created_at).
    Key = sha256(text after whitespace normalization) | task name | model | version. The value is JSON.

2. Why it is designed this way
    Calling outside services costs money and time, so each piece of text is sent only once. The key includes a
    hash of the text, so if the text changes at all (for example, the cleaning rules change), an old result is
    never reused by mistake. Paper-version results are imported under the rule "the new text is the same as, or
    almost the same as, the paper-version text" (see import_legacy.py). Imported records have source="legacy";
    newly fetched ones have source="api".

6. Possible errors and how they are handled
    The same key is written twice: the later write replaces the earlier one, and this is counted in stats.

8. How to check
    tests/test_m6.py::test_cache_roundtrip.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

DB = config.CACHE / "text" / "text_cache.sqlite"


def norm_text(text: str | None) -> str:
    """Normalize whitespace: turn each run of whitespace into one space and trim both ends."""
    return re.sub(r"\s+", " ", text or "").strip()


def text_sha(text: str | None) -> str:
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


class TextCache:
    """Store and look up results by (text, task, model, version)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, task TEXT, model TEXT, version TEXT, "
                         "text_sha TEXT, value TEXT, source TEXT, created_at TEXT)")
        self.stats = {"hit": 0, "miss": 0, "put": 0}

    @staticmethod
    def key(text: str, task: str, model: str, version: str) -> str:
        return f"{text_sha(text)}|{task}|{model}|{version}"

    def get_many(self, texts: list[str], task: str, model: str, version: str) -> dict[str, object]:
        """Return {text: result}. Texts with no stored result are left out.

        Texts that differ only in whitespace share one key, and each of them must get the result. (Before,
        only one of them got it and the others were treated as missing.)
        """
        keys: dict[str, list[str]] = {}
        for t in dict.fromkeys(texts):
            keys.setdefault(self.key(t, task, model, version), []).append(t)
        out = {}
        items = list(keys)
        for i in range(0, len(items), 500):
            chunk = items[i:i + 500]
            q = f"SELECT key, value FROM results WHERE key IN ({','.join('?' * len(chunk))})"
            for k, v in self.con.execute(q, chunk):
                val = json.loads(v)
                for t in keys[k]:
                    out[t] = val
        self.stats["hit"] += len(out)
        self.stats["miss"] += len(set(texts)) - len(out)
        return out

    def put_many(self, items: dict[str, object], task: str, model: str, version: str, source: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rows = [(self.key(t, task, model, version), task, model, version, text_sha(t), json.dumps(v), source, now)
                for t, v in items.items()]
        self.con.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?)", rows)
        self.con.commit()
        self.stats["put"] += len(rows)

    def missing(self, texts: list[str], task: str, model: str, version: str) -> list[str]:
        have = self.get_many(texts, task, model, version)
        return [t for t in dict.fromkeys(texts) if t not in have]
