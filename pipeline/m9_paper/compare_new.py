"""
compare_new.py: M9 part 2: what each number in the paper becomes on the new data

Module: M9 (PLAN.md M9; CODE_PLAN.md section 5, M9)

1. What it does
    Reads paper_numbers.yaml (for each paper number: where it appears, its value in the paper, and how to look it
    up in the paper-version result tables). For each entry it gets:
        legacy: the value recomputed with the new code on the paper-version data (results/m8_legacy/, M8 step 2);
        new   : the value on the new data (cutoff 2026-04-20T14:00:00Z, the full set of questions by uid, D10 sample)
                (results/m8/, the table with the same name).
    It then marks: the difference between the new value and the paper value, whether the sign changed, and whether
    significance changed (for entries that have a confidence interval).
    On the new data the reply-feature table is called reply_association (only the main outcome is run), so the
    DV_key condition is dropped when looking up values there.

2. Why it is designed this way
    The paper-version tables and the new-data tables are made by the same code with the same table names,
    so one query works on both.

4. Outputs
    results/paper_vs_new.csv, results/paper_vs_new.md

7. How to run
    python -m pipeline.m9_paper.compare_new
"""
from __future__ import annotations

import re
import sys

import pandas as pd
import yaml

import config
from pipeline.common.runlog import base_parser, setup_logging

RENAME = {"association_results.csv": "reply_association.csv",
          "association_by_subgroup.csv": "reply_association_by_subgroup.csv"}


def value_from(path, query, column, scale, cache):
    """Look up column in a result table using query (returns None if not found)."""
    if not path.exists():
        return None
    df = cache.setdefault(str(path), pd.read_csv(path))
    for q in (query, re.sub(r"`DV_key` == '[^']*' and ", "", query or "")):
        try:
            sub = df.query(q) if q else df
        except Exception:                                # noqa: BLE001  if a queried column does not exist, try the next form
            continue
        if len(sub) and column in sub:
            return float(sub[column].iloc[0]) * (scale or 1.0)
    return None


def main(argv=None):
    base_parser("M9: paper numbers vs the new data.").parse_args(argv)
    log = setup_logging("m9_new")
    y = yaml.safe_load((config.ROOT / "pipeline" / "m9_paper" / "paper_numbers.yaml").read_text())
    cache, rows = {}, []
    for e in y["numbers"]:
        src = e.get("source")
        if not src or e.get("kind") != "value":
            continue
        legacy = value_from(config.RESULTS / src, e.get("query"), e.get("column"), e.get("scale"), cache)
        name = src.split("/")[-1]
        new_src = config.RESULTS / "m8" / RENAME.get(name, name)
        new = value_from(new_src, e.get("query"), e.get("column"), e.get("scale"), cache)
        rows.append({"id": e["id"], "where": e["where"], "paper": e["value"], "legacy_recomputed": legacy, "new": new,
                     "new_minus_paper": None if new is None else new - e["value"]})
    df = pd.DataFrame(rows)
    # Significance: look at the three entries with the same prefix (after removing .att/.ci_lo/.ci_hi) together
    df["stem"] = df.id.str.replace(r"\.(att|ci_lo|ci_hi|naive|dr|cohen_d|d_ci_lo|d_ci_hi)$", "", regex=True)
    sig = []
    for stem, g in df.groupby("stem"):
        ids = dict(zip(g.id.str.extract(r"\.([a-z_]+)$")[0], g.index))
        if {"ci_lo", "ci_hi"} <= set(ids):
            lo_p, hi_p = df.loc[ids["ci_lo"], "paper"], df.loc[ids["ci_hi"], "paper"]
            lo_n, hi_n = df.loc[ids["ci_lo"], "new"], df.loc[ids["ci_hi"], "new"]
            if lo_n is not None and hi_n is not None and pd.notna(lo_n):
                sp, sn = (lo_p > 0 or hi_p < 0), (lo_n > 0 or hi_n < 0)
                sig.append({"stem": stem, "paper_sig": sp, "new_sig": sn, "sig_changed": sp != sn})
    sg = pd.DataFrame(sig)
    df = df.merge(sg, on="stem", how="left") if len(sg) else df
    df["sign_flip"] = (df.paper * df.new < 0) & df.new.notna()
    out = config.RESULTS / "paper_vs_new.csv"
    df.to_csv(out, index=False)
    have = df[df.new.notna()]
    lines = ["# Paper numbers vs the new data", "",
             f"- Cutoff {config.DATA_CUTOFF_ISO}; numbers matched to a new-data result table: {len(have)} / {len(df)}",
             f"- Sign changed: {int(have.sign_flip.sum())}; significance changed: {int(df.get('sig_changed', pd.Series(dtype=bool)).fillna(False).sum() // 3)} groups (each group = ATT plus lower and upper bound, three entries)", "",
             "| Location | Paper | Paper data recomputed | New data | New - paper | Significance paper -> new |", "|---|---|---|---|---|---|"]
    for r in have.itertuples():
        s = "" if pd.isna(getattr(r, "paper_sig", None)) else f"{'significant' if r.paper_sig else 'not significant'} -> {'significant' if r.new_sig else 'not significant'}"
        lg = "" if r.legacy_recomputed is None or pd.isna(r.legacy_recomputed) else f"{r.legacy_recomputed:+.4f}"
        lines.append(f"| {r.where} | {r.paper:+.4f} | {lg} | {r.new:+.4f} | {r.new_minus_paper:+.4f} | {s} |")
    (config.RESULTS / "paper_vs_new.md").write_text("\n".join(lines))
    log.info("paper numbers matched to new results: %d / %d; sign flips %d", len(have), len(df), int(have.sign_flip.sum()))


if __name__ == "__main__":
    main(sys.argv[1:])
