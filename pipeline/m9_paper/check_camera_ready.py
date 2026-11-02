"""
check_camera_ready.py: check every number in the camera-ready paper against the result files

Module: M9 (camera-ready; added 2026-11-02. check_numbers.py checks the submitted version of the paper)

1. What it does
    Reads the paper's LaTeX source and camera_ready_registry.py. For every registered table row it takes the numbers
    in the non-label cells, in order, and compares each with the result-file cell named in the registry; for every
    registered text snippet it finds the snippet in the source and compares the number in it. A number matches when
    |result x scale - paper| is at most half a unit of the last digit the paper prints ("<.001" matches when the
    result is below .001). It also lists the tables of the paper that are neither registered nor listed as unchecked.

2. Why it is designed this way
    The paper must not contain a number that no code produces. Matching rows by their label text (in order) and
    cells by position makes a moved row or an added column show up as an error instead of a silent wrong match.

3. Inputs
    --tex (default config.EXTERNAL["camera_ready_tex"]); results/m6, results/m8 csv files.

4. Outputs
    results/paper_check_camera_ready.csv  one row per checked number (status: ok, mismatch, no_result, lookup_error,
                                          row_not_found, count_mismatch, text_not_found)
    results/paper_check_camera_ready.md   counts per table and every problem

7. How to run
    python -m pipeline.m9_paper.check_camera_ready [--tex path/to/camera_ready.tex]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

import config
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m9_paper.camera_ready_registry import TABLES, TEXT, UNCHECKED

NUM = re.compile(r"(<)?\s*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))")


def clean(s: str) -> str:
    s = s.replace("$-$", "-").replace("{,}", ",")
    s = re.sub(r"\\(midrule|toprule|bottomrule|hline)|\\cmidrule(\(\w*\))?\{[^}]*\}", " ", s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[lcr]\}", "", s)
    s = re.sub(r"\\(textbf|emph|quad)", " ", s)
    s = s.replace("$", "").replace("\\%", " ").replace("{", "").replace("}", "")
    return " ".join(s.split())


def numbers(cell: str) -> list:
    """Numbers in one cell as (value, decimals, is_lt); '--' gives [None]."""
    c = clean(cell)
    if c.strip() in ("--", "---"):
        return [None]
    c = re.sub(r"[A-Za-z]+\d+", " ", c)               # Q1, L2 and the like are not numbers
    c = c.replace("pp", " ")
    out = []
    for lt, tok in NUM.findall(c):
        t = tok.replace(",", "")
        dec = len(t.split(".")[1]) if "." in t else 0
        out.append((float(t), dec, bool(lt)))
    return out


def table_rows(tex: str, label: str) -> list[list[str]] | None:
    pos = tex.find(f"\\label{{{label}}}")
    if pos < 0:
        return None
    start = max(tex.rfind("\\begin{table", 0, pos), tex.rfind("\\begin{longtable", 0, pos))
    ends = [e for e in (tex.find("\\end{table", pos), tex.find("\\end{longtable", pos)) if e >= 0]
    body = tex[start:min(ends)]
    body = re.sub(r"\\caption\{.*?\}\s*$", "", body, flags=re.S)
    return [r.split("&") for r in body.split("\\\\") if "&" in r]


def lookup(spec, src: str, query: str | None, cache: dict):
    """Value for one column spec; returns (value, error)."""
    if isinstance(spec, str):
        col, _, k = spec.partition("*")
        spec = {"col": col, "scale": float(k) if k else 1.0}
    src = spec.get("source", src)
    q = spec.get("q", query)
    path = config.RESULTS / src
    if not path.exists():
        return None, "no_result"
    df = cache.setdefault(src, pd.read_csv(path))
    try:
        sub = df.query(q, engine="python") if q else df
    except Exception as e:                                # noqa: BLE001  a wrong query is a registry error
        return None, f"lookup_error: {e}"
    if len(sub) != 1:
        return None, f"lookup_error: {len(sub)} rows for {q}"
    return float(sub.iloc[0][spec["col"]]) * spec.get("scale", 1.0), None


def compare(paper: tuple, value: float) -> bool:
    v, dec, lt = paper
    if lt:
        return value < v
    return abs(value - v) <= 0.5 * 10 ** -dec + 1e-9


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("Check the numbers of the camera-ready paper against the result files.")
    ap.add_argument("--tex", default=str(config.EXTERNAL.get("camera_ready_tex", "")))
    args = ap.parse_args(argv)
    log = setup_logging("m9_check_camera_ready")
    tex = Path(args.tex).read_text()
    flat = " ".join(tex.split())
    cache: dict = {}
    out = []
    for label, t in TABLES.items():
        rows = table_rows(tex, label)
        if rows is None:
            out.append({"table": label, "row": "", "status": "table_not_found"})
            continue
        p = 0
        for want, query, cols in t["rows"]:
            while p < len(rows):
                lab = clean(" ".join(rows[p][:t["label_cells"]]))
                if (lab.strip() == want) if want.isdigit() else (want in lab):
                    break
                p += 1
            if p == len(rows):
                out.append({"table": label, "row": want, "status": "row_not_found"})
                p = 0
                continue
            nums = [x for c in rows[p][t["label_cells"]:] for x in numbers(c)]
            if len(nums) != len(cols):
                out.append({"table": label, "row": want, "status": "count_mismatch",
                            "paper": str([x[0] if x else x for x in nums]), "result": f"{len(cols)} specs"})
                p += 1
                continue
            for j, (x, spec) in enumerate(zip(nums, cols)):
                if x is None or spec is None:
                    continue
                v, err = lookup(spec, t["source"], query, cache)
                out.append({"table": label, "row": want, "cell": j + 1, "paper": ("<" if x[2] else "") + str(x[0]),
                            "result": v, "status": err.split(":")[0] if err else ("ok" if compare(x, v) else "mismatch"),
                            "detail": err or ""})
            p += 1
    for tid, snip, num, src, q, col, scale in TEXT:
        s = " ".join(snip.split())
        if s not in flat or num not in s:
            out.append({"table": "text", "row": tid, "status": "text_not_found", "paper": num})
            continue
        x = numbers(num)[0]
        v, err = lookup({"col": col, "scale": scale}, src, q, cache)
        out.append({"table": "text", "row": tid, "paper": num, "result": v, "detail": err or "",
                    "status": err.split(":")[0] if err else ("ok" if compare(x, v) else "mismatch")})
    R = pd.DataFrame(out)
    labels = re.findall(r"\\label\{(tab:[^}]+)\}", tex)
    missing = [l for l in labels if l not in TABLES and l not in UNCHECKED]
    R.to_csv(config.RESULTS / "paper_check_camera_ready.csv", index=False)
    summ = R.groupby(["table", "status"]).size().unstack(fill_value=0)
    lines = ["# Camera-ready number check", "", f"Source: {args.tex}", "",
             f"Checked numbers: {len(R)}; ok: {(R.status == 'ok').sum()}; problems: {(R.status != 'ok').sum()}", "",
             summ.to_markdown(), "", "## Problems", "",
             R[R.status != "ok"].to_markdown(index=False) if (R.status != "ok").any() else "None.", "",
             "## Not checked", ""] + [f"- {k}: {v}" for k, v in UNCHECKED.items()] + [
             "", "## Tables in the paper that are not registered", ""] + ([f"- {m}" for m in missing] or ["None."])
    (config.RESULTS / "paper_check_camera_ready.md").write_text("\n".join(lines))
    log.info("checked %d numbers: %d ok, %d problems; unregistered tables: %s", len(R), (R.status == "ok").sum(),
             (R.status != "ok").sum(), missing or "none")


if __name__ == "__main__":
    main(sys.argv[1:])
