"""
io.py: shared read and write functions

Module: common layer (CODE_PLAN.md section 4)

1. What it does
    Provides the file reading and writing used across the whole project:
    line-by-line reading and atomic writing of jsonl, reading and writing tables
    (parquet / csv), file sha256, and reproducible derivation of random seeds.

2. Why it is designed this way
    1. Atomic writes: first write to a temporary file in the same folder, and
       only after that succeeds, rename it with os.replace. If a run fails or
       is stopped halfway, no "half-written" output file is left behind, so
       later steps never read broken data by mistake.
    2. Line-by-line reading: the s6 edits file is 3.4 GB and the abuse-filter
       file is 5.1 GB, too big to load into memory at once. iter_jsonl is a
       generator; together with a filter, it keeps only the records you need.
    3. Derived seeds: every random process gets its seed from config.SEED plus a
       "purpose label". The same purpose always gets the same seed, and
       different purposes get unrelated seeds.

3. Inputs / 4. Outputs
    Paths are given by the caller.

5. Processing steps
    None (this is a function library).

6. Possible errors and how they are handled
    - A jsonl line is not valid JSON: by default, raise ValueError with the file
      name and line number. With iter_jsonl(skip_bad=True), skip it and count it;
      the caller writes the count into an issue.
    - A write fails: the temporary file is deleted, the old output stays as it
      was, and the exception is passed up.

7. How to run
    Not run on its own.

8. How to check
    test_io_* in tests/test_common.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pandas as pd


def iter_jsonl(path: str | Path, *, skip_bad: bool = False,
               keep: Callable[[dict], bool] | None = None) -> Iterator[dict]:
    """Read a jsonl file line by line, keeping only one line in memory at a time.

    Args:
        path: file path.
        skip_bad: if True, skip lines that cannot be parsed (the caller can read the count from iter_jsonl.bad_lines);
                  if False, raise ValueError at the first bad line.
        keep: optional filter function; records for which it returns False are not yielded.
              Used to pick only the people you need from a big file.

    Yields:
        The dict parsed from each line. Empty lines are skipped.

    Raises:
        ValueError: skip_bad=False and a line is not valid JSON; the message has the file name and line number.
    """
    path = Path(path)
    bad = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                if skip_bad:
                    bad += 1
                    continue
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if keep is None or keep(rec):
                yield rec
    iter_jsonl.bad_lines = bad  # type: ignore[attr-defined]


def read_jsonl(path: str | Path, **kwargs: Any) -> list[dict]:
    """Read a whole jsonl file into a list. Only for small files; for big files use iter_jsonl."""
    return list(iter_jsonl(path, **kwargs))


def _atomic_target(path: Path) -> tuple[int, str]:
    """Create a temporary file in the same folder as the target, so that os.replace is an atomic step within one file system."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def write_jsonl_atomic(path: str | Path, records: Iterable[dict]) -> int:
    """Write a jsonl file atomically.

    Args:
        path: target path.
        records: an iterable of records (can be a generator).

    Returns:
        The number of lines written.

    Notes:
        ensure_ascii=False keeps user names in Chinese, Arabic, and other scripts as they are, which makes manual checks easier.
        If an error happens while writing, the temporary file is deleted and the old file is not touched.
    """
    path = Path(path)
    fd, tmp = _atomic_target(path)
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                n += 1
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return n


def write_json_atomic(path: str | Path, obj: Any, indent: int = 1) -> None:
    """Write one JSON file atomically (for small files such as reports and manifests)."""
    path = Path(path)
    fd, tmp = _atomic_target(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_table(path: str | Path, df: pd.DataFrame) -> None:
    """Write a table atomically. The file extension sets the format: .parquet (the recommended default) or .csv.

    Notes:
        parquet keeps data types (times, integers, booleans), and is the standard format for passing tables between modules.
        csv is only for small tables meant for people to read (for example, paper tables).
    """
    path = Path(path)
    fd, tmp = _atomic_target(path)
    os.close(fd)
    try:
        if path.suffix == ".parquet":
            df.to_parquet(tmp, index=False)
        elif path.suffix == ".csv":
            df.to_csv(tmp, index=False)
        else:
            raise ValueError(f"unsupported table format: {path.suffix}")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a table written by write_table."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"unsupported table format: {path.suffix}")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Compute the sha256 of a file (reads it in chunks, so it works for files of several GB)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Compute the sha256 of a string (UTF-8), used as the key for the text cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_seed(root: int, purpose: str) -> int:
    """Derive a fixed 32-bit seed from a root seed and a purpose label.

    Args:
        root: the root seed, usually config.SEED.
        purpose: the purpose label, for example "m5.pseudo_anchor" or "m8.bootstrap.primary".

    Returns:
        An integer from 0 to 2**32-1. The same (root, purpose) always gives the same result.

    Example:
        derive_seed(42, "m5.pseudo_anchor") is the same on every run,
        and is unrelated to derive_seed(42, "m8.bootstrap").
    """
    digest = hashlib.sha256(f"{root}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:4], "big")
