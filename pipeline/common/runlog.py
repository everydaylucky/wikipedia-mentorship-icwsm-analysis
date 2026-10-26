"""
runlog.py: shared setup for logging and command-line arguments

Module: common layer (CODE_PLAN.md section 2.3)

1. What it does
    1. setup_logging(module): logs go both to the screen and to results/logs/<module>_<time>.log.
    2. base_parser(desc): command-line arguments shared by all module scripts (--online, --refresh, --limit).
    3. Timer: records the start time, so the run time can be printed at the end.

2. Why it is designed this way
    The record of every script run stays under results/logs/. Later you can look
    up what a given run read, what it wrote, and which problems came up. Argument
    names are the same across the whole project.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

import config


def setup_logging(module: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logging: screen + file. Returns the logger named module."""
    config.RESULTS.joinpath("logs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = config.RESULTS / "logs" / f"{module}_{stamp}.log"
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(logfile, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    logger = logging.getLogger(module)
    logger.info("log file: %s", logfile)
    logger.info("DATA_CUTOFF = %s", config.DATA_CUTOFF_ISO)
    return logger


def base_parser(description: str) -> argparse.ArgumentParser:
    """Command-line arguments shared by all module scripts.

    --online   allow network access (default is offline: fail if the item is not in the cache)
    --refresh  ignore the API cache and request again (needs --online)
    --limit N  process only the first N items, for debugging
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--online", action="store_true", help="allow network requests")
    p.add_argument("--refresh", action="store_true", help="ignore API cache (requires --online)")
    p.add_argument("--limit", type=int, default=None, help="process only the first N items (debug)")
    return p


class Timer:
    """Simple timer: t = Timer(); ...; t.elapsed()"""

    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start
