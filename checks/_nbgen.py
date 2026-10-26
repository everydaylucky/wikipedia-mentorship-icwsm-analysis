"""
_nbgen.py: a small tool that writes the check notebooks

Module: checks (CODE_PLAN.md section 7.2)

1. What it does
    Describes notebook cells as a Python list and writes them to an .ipynb file (nbformat 4).
    run_as_script() joins all code cells into one script and runs it, so we can confirm that a
    notebook runs from top to bottom without a Jupyter kernel.

2. Why it is designed this way
    The environment has neither nbformat nor nbconvert. An .ipynb file is just JSON, so we write it
    directly. The content of each notebook lives in checks/build_<module>_check.py, which keeps it
    easy to review and change.

3. How to use
    from checks._nbgen import md, code, write_notebook, run_as_script
    cells = [md("# Title"), code("print(1)")]
    write_notebook(cells, "checks/m1_check.ipynb")
    run_as_script("checks/m1_check.ipynb")      # runs every cell in order; raises on the first error
"""
from __future__ import annotations

import json
from pathlib import Path


def md(src: str) -> dict:
    """One markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    """One code cell (no outputs; Jupyter fills them in when you run it)."""
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def write_notebook(cells: list[dict], path: str | Path) -> Path:
    """Write the .ipynb file."""
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    path = Path(path)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return path


def run_as_script(path: str | Path, cwd: str | Path | None = None) -> None:
    """Run all code cells of a notebook in order, to confirm it runs from top to bottom.

    display() and other IPython display helpers are replaced by print. matplotlib uses a
    non-interactive backend.
    """
    import os
    nb = json.loads(Path(path).read_text())
    src = ("import matplotlib\nmatplotlib.use('Agg')\n"
           "def display(x):\n    print(x.to_string() if hasattr(x, 'to_string') else x)\n")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src += f"\nprint('\\n########## cell {i}')\n" + "".join(c["source"]) + "\n"
    old = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        exec(compile(src, str(path), "exec"), {"__name__": "__nbrun__"})
    finally:
        os.chdir(old)
