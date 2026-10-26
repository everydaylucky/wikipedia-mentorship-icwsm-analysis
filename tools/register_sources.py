"""
register_sources.py: record every existing raw data file and write data/raw/SOURCES.yaml

Module: tools (CODE_PLAN.md section 3; PLAN.md section 1)

1. What it does
    For every file listed in config.EXTERNAL and config.RAW_FILES, it computes the sha256
    fingerprint, the size, the line count (for jsonl / csv / tsv / txt), and the modified time.
    It writes all of this to data/raw/SOURCES.yaml.

2. Why it is designed this way
    The raw data is not copied into this repository (that would take about 15 GB twice).
    We only record where each file is and its fingerprint. Later, anyone who reruns the
    pipeline can use --verify to confirm they are still reading the same data. If a file
    was overwritten by a new download (this has happened before), we find out right away.

3. Inputs
    The paths in config.EXTERNAL and config.RAW_FILES.
    With --verify: the existing data/raw/SOURCES.yaml.

4. Outputs
    data/raw/SOURCES.yaml, with one entry per file (path, exists, bytes, mtime, sha256, lines).
    For a folder, it records the number of files and the total bytes instead.
    With --verify: no file is written. It prints OK or FAIL for each file.

5. Key parameters
    --verify: check each sha256 against SOURCES.yaml instead of writing a new one.
    LINE_COUNT_SUFFIXES: file types whose lines are counted.

6. Possible errors and how they are handled
    File does not exist: it is recorded as exists: false, and a warning is printed.
    With --verify, a missing file or a changed sha256 counts as a failure, and the exit code is 1.

7. How to run
    python tools/register_sources.py           # write or update SOURCES.yaml (about 5 minutes, mostly sha256 of the large files)
    python tools/register_sources.py --verify  # check every sha256 against the record

8. How to check
    Run with --verify. Every line should say OK and the exit code should be 0.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

import config  # noqa: E402
from pipeline.common.io import sha256_file  # noqa: E402

OUT = config.RAW / "SOURCES.yaml"
LINE_COUNT_SUFFIXES = {".jsonl", ".csv", ".tsv", ".txt"}


def count_lines(path: Path) -> int:
    """Count newline characters block by block (it does not parse the content, so it works on files of several GB)."""
    n = 0
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            n += b.count(b"\n")
    return n


def describe(key: str, path: Path) -> dict:
    """The record for one file."""
    if not path.exists():
        print(f"  WARNING missing: {key}: {path}", flush=True)
        return {"path": str(path), "exists": False}
    st = path.stat()
    info = {"path": str(path), "exists": True, "is_dir": path.is_dir()}
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        info.update({"n_files": len(files), "bytes": sum(p.stat().st_size for p in files)})
        return info
    info.update({
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    })
    if path.suffix in LINE_COUNT_SUFFIXES:
        info["lines"] = count_lines(path)
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--verify", action="store_true", help="verify sha256 against SOURCES.yaml")
    args = ap.parse_args(argv)
    reg = {**{f"external.{k}": v for k, v in config.EXTERNAL.items()},
           **{f"raw.{k}": v for k, v in config.RAW_FILES.items()}}

    if args.verify:
        saved = yaml.safe_load(OUT.read_text())["sources"]
        bad = 0
        for k, info in saved.items():
            p = Path(info["path"])
            if not info.get("exists") or info.get("is_dir"):
                continue
            ok = p.exists() and sha256_file(p) == info["sha256"]
            bad += not ok
            print(f"  {'OK  ' if ok else 'FAIL'} {k}", flush=True)
        raise SystemExit(1 if bad else 0)

    out = {}
    for k, p in reg.items():
        print(f"  {k} ...", flush=True)
        out[k] = describe(k, p)
    doc = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "data_cutoff": config.DATA_CUTOFF_ISO,
           "note": "Record of the existing raw data. Files stay where they are and are not copied into this repository. Use --verify to check them.",
           "sources": out}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
    print(f"wrote {OUT}  ({len(out)} entries)")


if __name__ == "__main__":
    main(sys.argv[1:])
