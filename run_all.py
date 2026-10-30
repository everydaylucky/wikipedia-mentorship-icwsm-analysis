"""
run_all.py: run all stages in order

Module: main controller (CODE_PLAN.md section 8)

1. What it does
    Calls the offline script of each module in order, from M1 to M9 (in M8: the main analysis, then the
    camera-ready analyses: later periods, descriptive analyses, edit quality, full bootstrap; in M9: the
    figures, then the number check). Before each
    stage starts, it checks the upstream manifest. After each stage ends, it
    prints a summary. If any step fails, it stops at once and says which stage
    failed.

2. Why it is designed this way
    Online data collection (scripts run with --online) is not run here. Those
    scripts only need to run once, and their results go into the cache.
    run_all.py only does offline computation, so "the same raw data -> the same
    results" always holds.

    Stage A (replication of the paper version) runs on its own with --legacy:
    it only runs legacy_parity in M8 and the checks in M9.

3. How to run
    python run_all.py                    # all offline stages
    python run_all.py --from m5 --to m7  # only some of the stages
    python run_all.py --legacy           # only the paper-version replication
    python run_all.py --list             # list all stages and their status

6. Possible errors and how they are handled
    The module for a stage is not written yet: print "not implemented yet" and
    stop (--list shows how far the implementation has come).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time

import config

#: Stage table: (stage name, module path, description). The order is the run order.
STAGES: list[tuple[str, str, str]] = [
    ("m1", "pipeline.m1_roster.build_roster", "Mentor roster"),
    ("m2", "pipeline.m2_conversations.build_questions", "Questions and replies"),
    ("m3", "pipeline.m3_activity.merge_activity", "Newcomer activity"),
    ("m4", "pipeline.m4_eligibility.build_eligibility", "Assignment eligibility"),
    ("m5", "pipeline.m5_sample.build_sample", "Sample, treatment, outcomes"),
    ("m6", "pipeline.m6_text.build_text", "Text features"),
    ("m7", "pipeline.m7_dataset.build_dataset", "PSM dataset"),
    ("m8", "pipeline.m8_analysis.run_analysis", "Analysis"),
    ("m8_followup", "pipeline.m8_analysis.followup", "Later periods"),
    ("m8_descriptive", "pipeline.m8_analysis.descriptive", "Descriptive"),
    ("m8_quality", "pipeline.m8_analysis.quality", "Edit quality"),
    ("m8_full_bootstrap", "pipeline.m8_analysis.full_bootstrap", "Full bootstrap"),
    ("m9_figures", "pipeline.m9_paper.figures", "Paper figures"),
    ("m9", "pipeline.m9_paper.check_numbers", "Paper number check"),
]
LEGACY_STAGES: list[tuple[str, str, str]] = [
    ("m8_legacy", "pipeline.m8_analysis.legacy_parity", "Paper replication"),
    ("m9", "pipeline.m9_paper.check_numbers", "Paper number check"),
]


def stage_status(name: str, module_path: str) -> dict[str, str]:
    """Progress of one stage, shown by --list.

    Returns a dict with these items:
        code     is the code written: "done" if the entry file exists, otherwise "missing"
        run      has it been run: if a manifest exists, show its finish time, otherwise "not run"
        test     does the matching tests/test_<stage>.py exist
        check    does the matching checks/<stage>_check.ipynb exist
        issues   number of problem records (from _issues/summary.json); empty if none
    """
    short = name.split("_")[0]
    src = config.ROOT / (module_path.replace(".", "/") + ".py")
    man = config.INTERIM / name / "manifest.json"
    run = "not run"
    if man.exists():
        run = json.loads(man.read_text()).get("finished_at", "")[:16].replace("T", " ")
    summ = config.INTERIM / name / "_issues" / "summary.json"
    issues = ""
    if summ.exists():
        issues = str(sum(v["n"] for v in json.loads(summ.read_text()).values()))
    return {
        "code": "done" if src.exists() else "missing",
        "run": run,
        "test": "yes" if (config.ROOT / "tests" / f"test_{short}.py").exists() else "-",
        "check": "yes" if (config.ROOT / "checks" / f"{short}_check.ipynb").exists() else "-",
        "issues": issues,
    }


def print_status(stages: list[tuple[str, str, str]]) -> None:
    """Print the progress table, plus the status of the common-layer tests and the raw-data registry."""
    print(f"  {'Stage':19s}{'Content':29s}{'Code':6s}{'Run':18s}{'Test':6s}{'Check':6s}{'Issues':>8s}")
    print("  " + "-" * 92)
    for n, mp, d in stages:
        s = stage_status(n, mp)
        print(f"  {n:19s}{d:29s}{s['code']:6s}{s['run']:18s}{s['test']:6s}{s['check']:6s}{s['issues']:>8s}")
    common_ok = (config.ROOT / "tests" / "test_common.py").exists()
    sources = config.RAW / "SOURCES.yaml"
    print("\n  Common layer:", "done (pytest tests/test_common.py)" if common_ok else "missing")
    print("  Raw-data registry:", f"{sources}" if sources.exists() else "not created yet (python tools/register_sources.py)")
    print("  Notes: Code = does the entry file exist; Run = time of the last successful run; "
          "Test/Check = matching files under tests/ and checks/; Issues = number of records in _issues/")


def run_stage(name: str, module_path: str, desc: str) -> None:
    """Import a stage and run its main(). If the module does not exist, report "not implemented yet"."""
    print(f"\n{'=' * 72}\n  {name}: {desc}  ({module_path})\n{'=' * 72}", flush=True)
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if module_path.split(".")[-1] in str(e):
            raise SystemExit(f"[{name}] not implemented yet: {module_path}")
        raise
    t = time.time()
    mod.main([])
    print(f"[{name}] finished in {time.time() - t:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the offline pipeline stages in order.")
    ap.add_argument("--from", dest="start", default=None, help="first stage (e.g. m5)")
    ap.add_argument("--to", dest="end", default=None, help="last stage (e.g. m7)")
    ap.add_argument("--legacy", action="store_true", help="run only the paper-version reproduction")
    ap.add_argument("--list", action="store_true", help="list stages and their status")
    args = ap.parse_args(argv)

    stages = LEGACY_STAGES if args.legacy else STAGES
    names = [s[0] for s in stages]
    if args.list:
        print_status(stages)
        return
    i0 = names.index(args.start) if args.start else 0
    i1 = names.index(args.end) if args.end else len(stages) - 1
    for n, mp, d in stages[i0:i1 + 1]:
        run_stage(n, mp, d)


if __name__ == "__main__":
    main(sys.argv[1:])
