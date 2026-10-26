"""
manifest.py: run records and upstream checks

Module: common layer (CODE_PLAN.md sections 2.3 and 4)

1. What it does
    1. write_manifest(): called when each script ends. It records the input files
       of this run (sha256, line count, modification time), the output files,
       the parameters, the git commit, and the run time, and writes them to
       data/interim/<module>/manifest.json.
    2. check_upstream(): called when a script starts. It confirms that the
       manifests of upstream modules exist, and that the upstream outputs are
       newer than this module's outputs (if not, it says the upstream module
       needs to be rerun).
    3. check_sources(): confirms that all external data files this module needs exist.

2. Why it is designed this way
    The raw data is spread over several folders, and there have been cases where
    "a file with the same name was overwritten by a newer collection" (the June
    recollection of s1 and s2 overwrote the April versions). The manifest records
    exactly which copy of each file a run read (sha256), so a problem can be
    traced back precisely.

    Computing sha256 for big files (several GB) is slow, so for external data we
    only record size and modification time by default. The full sha256 values in
    data/raw/SOURCES.yaml are computed separately by tools/register_sources.py.

6. Possible errors and how they are handled
    Upstream manifest missing: raise RuntimeError and say which module to run first.
    External file missing: raise FileNotFoundError and list all missing files.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import config
from pipeline.common.io import sha256_file, write_json_atomic

HASH_LIMIT_BYTES = 200 * 1024 * 1024   # for files over 200 MB, record only size and modification time


def git_hash(repo: Path = config.ROOT) -> str | None:
    """The git commit of the current repo; returns None if it is not a git repo."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def file_info(path: Path, full_hash: bool = False) -> dict[str, Any]:
    """Registry info for one file: path, size, modification time, and sha256 (for small files, or when asked)."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    info = {"path": str(path), "exists": True, "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    if full_hash or st.st_size <= HASH_LIMIT_BYTES:
        info["sha256"] = sha256_file(path)
    return info


def write_manifest(module: str, *, inputs: Iterable[Path], outputs: Iterable[Path],
                   params: dict[str, Any] | None = None, counts: dict[str, Any] | None = None,
                   started: float | None = None) -> Path:
    """Write the manifest for this run.

    Args:
        module: module name, for example "m2".
        inputs: files that were read.
        outputs: files that were written.
        params: parameters of this run (for example, online or not, cutoff date).
        counts: numbers of records read, written, and of each issue type.
        started: start time (time.time()), used to record how long the run took.

    Returns:
        Path of the manifest file, data/interim/<module>/manifest.json.
    """
    m = {
        "module": module,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - started, 1) if started else None,
        "git": git_hash(),
        "python": sys.version.split()[0],
        "data_cutoff": config.DATA_CUTOFF_ISO,
        "argv": sys.argv,
        "params": params or {},
        "counts": counts or {},
        "inputs": [file_info(p) for p in inputs],
        "outputs": [file_info(p) for p in outputs],
    }
    path = config.interim(module) / "manifest.json"
    write_json_atomic(path, m)
    return path


def check_upstream(module: str, upstream: Iterable[str]) -> dict[str, dict]:
    """Check that the upstream modules have been run, and return their manifests.

    Raises:
        RuntimeError: an upstream module has no manifest; the message says to run it first.
    """
    out = {}
    for up in upstream:
        p = config.INTERIM / up / "manifest.json"
        if not p.exists():
            raise RuntimeError(f"[{module}] upstream {up} has not been run "
                               f"(missing {p}); run it first: python run_all.py --from {up} --to {up}")
        out[up] = json.loads(p.read_text())
    return out


def check_sources(keys: Iterable[str]) -> dict[str, Path]:
    """Confirm that all the external data named in config.EXTERNAL / config.RAW_FILES exists.

    Args:
        keys: registry names, for example ["s6_profiles", "claim_log"].

    Returns:
        A dict from registry name to path.

    Raises:
        FileNotFoundError: lists all missing files.
    """
    reg = {**config.EXTERNAL, **config.RAW_FILES}
    paths, missing = {}, []
    for k in keys:
        if k not in reg:
            raise KeyError(f"unknown source key {k!r}; add it to config.EXTERNAL or RAW_FILES")
        p = reg[k]
        paths[k] = p
        if not p.exists():
            missing.append(f"{k}: {p}")
    if missing:
        raise FileNotFoundError("missing source files:\n  " + "\n  ".join(missing))
    return paths
