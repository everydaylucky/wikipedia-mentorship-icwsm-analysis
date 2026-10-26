"""
build_paper_numbers.py: build a first draft of paper_numbers.yaml from the paper's LaTeX source

Module: M9 (PLAN.md M9; CODE_PLAN.md section 5, M9)

1. What it does
    It reads every table in the paper (config.EXTERNAL["paper_md"]) and records each number
    as one entry: which table, which row and column, the string and value written in the paper,
    the result file and how to look the value up in it, and the allowed error. It also adds the
    diagnostic numbers that the main text quotes directly. It writes pipeline/m9_paper/paper_numbers.yaml.

2. Why it is designed this way
    The paper has several hundred numbers, and copying them by hand leads to mistakes. Reading
    them straight from the LaTeX source makes sure the recorded "paper value" is exactly the
    string the paper prints.
    The allowed error follows the number of decimals shown in the paper: ".017" allows ±0.0005,
    and "+.0174" allows ±0.00005. So a result that matches the paper after rounding counts as a
    match, and we do not get false alarms just from display precision.

    This tool only needs to be rerun when the paper is finalized or a table changes. After that,
    a person can add a note to entries in the yaml by hand (the issue ID of a known problem and
    a short explanation).

3. Inputs
    The paper's LaTeX source at config.EXTERNAL["paper_md"].

4. Outputs
    pipeline/m9_paper/paper_numbers.yaml: the list of numbers ("numbers") and the tables that later
    modules will reproduce ("pending_tables").

5. Key parameters
    L: the results folder of stage A (under results/).
    ATT_LABELS, ABLATION, SUBGROUPS, REPLY_SG, FEATURES: map row labels in the paper to the names
    used in the result files.
    TEXT_NUMBERS: numbers quoted in the main text. PENDING_TABLES: tables not covered by stage A.

6. Possible errors and how they are handled
    A table label is not found: tex.index raises ValueError and the run stops.
    A row label is not in one of the maps above: KeyError, and the run stops. Add the label to the map.
    A cell has no number: the cell is skipped.

7. How to run
    python tools/build_paper_numbers.py

8. How to check
    The printed count of numbers should be about the same as before. Then run
    pipeline/m9_paper/check_numbers.py and look for rows that do not match.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

import config  # noqa: E402

OUT = config.ROOT / "pipeline" / "m9_paper" / "paper_numbers.yaml"
L = "m8_legacy/"   # results folder of stage A (under results/)

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def clean(cell: str) -> str:
    """Remove LaTeX markup and keep the readable text."""
    c = cell.replace("$-$", "-").replace("$<$", "<").replace("\\,", "").replace("{,}", ",")
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)
    c = c.replace("{[}", "[").replace("{]}", "]").replace("$", "").replace("\\", "").replace("~", " ")
    return c.strip()


def table_rows(tex: str, label: str) -> list[list[str]]:
    """Get the data rows of the table with a given label (between \\midrule and \\bottomrule)."""
    pos = tex.index(f"\\label{{{label}}}")
    # First find the environment that holds the label (table, table* or longtable), then take the first table inside it.
    # Some tables put the caption and label before the table, so we cannot just "look back for the nearest tabular".
    env = max(tex.rfind("\\begin{table", 0, pos), tex.rfind("\\begin{longtable", 0, pos))
    if tex.startswith("\\begin{longtable", env):
        begin = env
    else:
        begin = tex.find("\\begin{tabular}", env)
    end = tex.find("\\bottomrule", begin)
    if end == -1 or end > pos + 20000:
        end = tex.find("\\end{tabular}", begin)
    body = tex[begin:end]
    mid = body.find("\\midrule")
    body = body[mid + len("\\midrule"):] if mid != -1 else body
    rows = []
    for raw in body.split("\\\\"):
        raw = raw.replace("\\midrule", "").strip()
        if not raw or raw.startswith("\\multicolumn"):
            continue
        rows.append([clean(c) for c in raw.split("&")])
    return rows


def num(s: str) -> tuple[float | None, int, str]:
    """Turn a number string from the paper into (value, number of decimals, kind). Kind is "value" or "lt" (like <.001)."""
    s = s.replace(",", "").replace("*", "").strip()
    kind = "lt" if s.startswith("<") else "value"
    m = NUM_RE.search(s)
    if not m:
        return None, 0, kind
    t = m.group(0)
    dec = len(t.split(".")[1]) if "." in t else 0
    return float(t), dec, kind


def entry(id_, where, paper_str, source, query, column, note="", scale=1.0) -> dict | None:
    """One record. scale is for percentages (the paper writes 8.8%, and the result file has 8.8)."""
    v, dec, kind = num(paper_str)
    if v is None:
        return None
    tol = 0.5 * 10 ** (-dec) + 1e-12          # in the units the paper displays
    e = {"id": id_, "where": where, "paper": paper_str, "value": v, "kind": kind, "tol": round(tol, 12),
         "source": source, "query": query, "column": column, "scale": scale}
    if note:
        e["note"] = note
    return e


def ci(s: str) -> tuple[str, str]:
    """"[+.006, +.028]" becomes ("+.006", "+.028")."""
    a, b = s.strip("[] ").split(",")
    return a.strip(), b.strip()


ATT_LABELS = {"1+ edit (14d)": "1+_mainspace_edit_14d", "N edits (14d)": "n_mainspace_edits_14d",
              "Unique NS (14d)": "unique_namespace_14d", "2+ active days (14d)": "2+_active_days_14d",
              "Active days (30d)": "active_days_30d", "Reverted (14d)": "reverted_any_14d"}
ABLATION = {"Full model": "Full model (E+Qtext+Qpersp+Qtype+Emb+T)", "Full - Qtype": "Full – Qtype (no LLM annotation)",
            "Full - temporal": "Full – temporal", "Full - embeddings": "Full – embeddings", "Full - Qtext": "Full – Qtext",
            "Edit history only": "Edit history only", "Question feat. only": "Question features only (Qtext+Qpersp+Qtype)",
            "Qtype only": "Qtype only", "Full + mentor": "Full + mentor features"}
SUBGROUPS = {"Full sample": "Full sample", "Technical": "Technical", "Normative (Q4)": "Normative (Q4)",
             "Appraisal (Q3)": "Appraisal (Q3)", "Referent (Q2)": "Referent (Q2)", "Non-substantive": "Non-substantive",
             "Own work (Q5=Y)": "Own work (Q5=Y)", "No own work (Q5=N)": "No own work (Q5=N)"}
REPLY_SG = {"Technical": "Technical", "Referent": "Referent (Q2)", "Normative": "Normative (Q4)",
            "Non-subst.": "Non-substantive", "Q5=Y": "Own work (Q5=Y)", "Q5=N": "No own work (Q5=N)"}
FEATURES = {"Reply lag (hours)": "Reply lag (hours)", "Q--R cosine sim.": "Q-R semantic similarity",
            "Q--R cos. sim.": "Q-R semantic similarity", "Positive language": "Positive language",
            "Pos. language": "Positive language", "Question marks": "Question marks in reply",
            "Quest. marks": "Question marks in reply", "VADER compound": "VADER compound sentiment",
            "VADER sent.": "VADER compound sentiment", "``We'' rate": "\"We\" rate (per word)",
            "``I'' rate": "\"I\" rate (per word)", "``You'' rate": "\"You\" rate (per word)",
            "Reply word count": "Reply word count", "Reply/Q word ratio": "Reply/question word ratio",
            "Flesch-Kincaid grade": "Flesch-Kincaid grade", "TextBlob subjectivity": "TextBlob subjectivity",
            "Imperative sentences": "Imperative sentences", "Has list": "Has list", "Has greeting": "Has greeting",
            "Gratitude": "Gratitude expression", "Deference": "Deference", "Policy refs": "Policy references",
            "Wikilinks": "Wikilinks", "Help page refs": "Help page references",
            "Resource links (total)": "Resource links (total)", "Resource links": "Resource links (total)"}


def q(**kw) -> str:
    """Build a pandas query string. String values are quoted (inner double quotes are escaped)."""
    parts = []
    for k, v in kw.items():
        if isinstance(v, str):
            parts.append(f"`{k}` == {v!r}")
        else:
            parts.append(f"`{k}` == {v}")
    return " and ".join(parts)


def build(tex: str) -> list[dict]:
    E: list[dict] = []
    add = lambda e: e and E.append(e)  # noqa: E731

    # Table 4 (tab:att): main results
    for r in table_rows(tex, "tab:att"):
        o = ATT_LABELS[r[0]]
        w = f"Table 4 (tab:att), row {r[0]}"
        add(entry(f"att.{o}.naive", w + ", Naive", r[1], L + "B_main_results.csv", q(Outcome=o), "Naive"))
        add(entry(f"att.{o}.att", w + ", ATT", r[2], L + "B_main_results.csv", q(Outcome=o), "ATT"))
        lo, hi = ci(r[3])
        add(entry(f"att.{o}.ci_lo", w + ", CI low", lo, L + "B_main_results.csv", q(Outcome=o), "CI_lo"))
        add(entry(f"att.{o}.ci_hi", w + ", CI high", hi, L + "B_main_results.csv", q(Outcome=o), "CI_hi"))

    # Appendix: number of strata (tab:strata_sens)
    for r in table_rows(tex, "tab:strata_sens"):
        K = int(r[0]); w = f"Appendix tab:strata_sens, K={K}"
        qq = q(Outcome="1+_mainspace_edit_14d", Strata=K)
        add(entry(f"strata.K{K}.att", w, r[1], L + "C1_strata_sensitivity.csv", qq, "ATT"))
        lo, hi = ci(r[2])
        add(entry(f"strata.K{K}.ci_lo", w + " CI low", lo, L + "C1_strata_sensitivity.csv", qq, "CI_lo"))
        add(entry(f"strata.K{K}.ci_hi", w + " CI high", hi, L + "C1_strata_sensitivity.csv", qq, "CI_hi"))

    # Appendix: ablation (tab:ablation)
    for r in table_rows(tex, "tab:ablation"):
        name = ABLATION[r[0].replace("  ", " ")]
        w = f"Appendix tab:ablation, {r[0]}"
        add(entry(f"ablation.{r[0]}.p", w + ", Feat.", r[1], L + "A7_feature_ablation.csv", q(Model=name), "p"))
        add(entry(f"ablation.{r[0]}.auc", w + ", AUC", r[2], L + "A7_feature_ablation.csv", q(Model=name), "AUC"))
        add(entry(f"ablation.{r[0]}.att", w + ", ATT", r[3], L + "A7_feature_ablation.csv", q(Model=name), "ATT"))

    # Appendix: trimming (tab:trimming)
    for r in table_rows(tex, "tab:trimming"):
        lo_s, hi_s = ci(r[0])
        rng = f"[{float(lo_s)},{float(hi_s)}]"       # same format as in the result table, for example "[0.1,0.9]"
        w = f"Appendix tab:trimming, {rng}"
        add(entry(f"trim.{rng}.n", w + ", N trimmed", r[1], L + "C3_ps_trimming.csv", q(Range=rng), "Trimmed"))
        add(entry(f"trim.{rng}.att", w + ", ATT", r[2], L + "C3_ps_trimming.csv", q(Range=rng), "ATT"))

    # Appendix: window (tab:window)
    for r in table_rows(tex, "tab:window"):
        wd = int(r[0].split()[0]); w = f"Appendix tab:window, {wd} days"
        add(entry(f"window.{wd}.att", w, r[1], L + "C4_window_sensitivity.csv", q(window=wd), "ATT"))
        lo, hi = ci(r[2])
        add(entry(f"window.{wd}.ci_lo", w + " CI low", lo, L + "C4_window_sensitivity.csv", q(window=wd), "CI_lo"))
        add(entry(f"window.{wd}.ci_hi", w + " CI high", hi, L + "C4_window_sensitivity.csv", q(window=wd), "CI_hi"))

    # Appendix: stratum details (tab:strata_details)
    for r in table_rows(tex, "tab:strata_details"):
        k = int(r[0]); w = f"Appendix tab:strata_details, stratum {k}"
        for col, src in (("N", r[1]), ("Treated", r[2]), ("Control", r[3])):
            add(entry(f"strata_detail.{k}.{col}", w + f", {col}", src, L + "A3_strata_diagnostics.csv",
                      q(Stratum=k), col))
        lo, hi = ci(r[4])
        add(entry(f"strata_detail.{k}.ps_hi", w + ", PS high", hi, L + "A3_strata_diagnostics.csv", q(Stratum=k), "PS_hi"))

    # Table 5 (tab:subgroup): subgroups
    for r in table_rows(tex, "tab:subgroup"):
        sg = SUBGROUPS[r[0]]; w = f"Table 5 (tab:subgroup), {r[0]}"
        qq = q(Subgroup=sg, Outcome="1+_mainspace_14d")
        add(entry(f"subgroup.{sg}.N", w + ", N", r[1], L + "subgroup_att.csv", qq, "N"))
        add(entry(f"subgroup.{sg}.att", w + ", ATT", r[2], L + "subgroup_att.csv", qq, "ATT"))
        lo, hi = ci(r[3])
        add(entry(f"subgroup.{sg}.ci_lo", w + ", CI low", lo, L + "subgroup_att.csv", qq, "CI_lo"))
        add(entry(f"subgroup.{sg}.ci_hi", w + ", CI high", hi, L + "subgroup_att.csv", qq, "CI_hi"))

    # Table 6 (tab:reply_full)
    for r in table_rows(tex, "tab:reply_full"):
        f = FEATURES[r[0]]; w = f"Table 6 (tab:reply_full), {r[0]}"
        qq = q(DV_key="primary", Feature=f)
        add(entry(f"reply.full.{f}.beta", w + ", beta", r[1], L + "association_results.csv", qq, "Beta_std"))
        add(entry(f"reply.full.{f}.p", w + ", p", r[2], L + "association_results.csv", qq, "p"))

    # Table 7 (tab:reply_subgroup)
    for r in table_rows(tex, "tab:reply_subgroup"):
        sg = REPLY_SG[r[0]]; f = FEATURES[r[2]]; w = f"Table 7 (tab:reply_subgroup), {r[0]} / {r[2]}"
        qq = q(Subgroup=sg, DV_key="primary", Feature=f)
        add(entry(f"reply.sg.{sg}.N", w + ", N", r[1], L + "association_by_subgroup.csv", qq, "N"))
        add(entry(f"reply.sg.{sg}.{f}.beta", w + ", beta", r[3], L + "association_by_subgroup.csv", qq, "Beta_std"))
        add(entry(f"reply.sg.{sg}.{f}.p", w + ", p", r[4], L + "association_by_subgroup.csv", qq, "p"))

    # Appendix: full table of reply features (tab:reply_subgroup_full). Column order: Technical, Referent, Normative, Non-subst., Q5=Y, Q5=N
    sg_cols = ["Technical", "Referent (Q2)", "Normative (Q4)", "Non-substantive", "Own work (Q5=Y)", "No own work (Q5=N)"]
    rows = table_rows(tex, "tab:reply_subgroup_full")
    for r in rows:
        if r[0] not in FEATURES:
            continue
        f = FEATURES[r[0]]
        for sg, cell in zip(sg_cols, r[1:]):
            add(entry(f"reply.full_table.{sg}.{f}", f"Appendix tab:reply_subgroup_full, {r[0]} / {sg}", cell,
                      L + "association_by_subgroup.csv", q(Subgroup=sg, DV_key="primary", Feature=f), "Beta_std"))
    return E


#: Numbers quoted directly in the main text (not in a table). Each value is (id, where, string in the paper, source, query, column, scale, note)
TEXT_NUMBERS = [
    ("text.n", "Abstract, Sec 3.2", "35,381", L + "s13_text_numbers.csv", "", "n", 1, ""),
    ("text.n_treated", "Sec 3.2", "26,278", L + "s13_text_numbers.csv", "", "n_treated", 1, ""),
    ("text.n_control", "Sec 3.2", "9,103", L + "s13_text_numbers.csv", "", "n_control", 1, ""),
    ("text.treated_pct", "Sec 2.3", "74.3", L + "s13_text_numbers.csv", "", "treated_pct", 1, ""),
    ("text.auc", "Sec 4.3", "0.618", L + "s13_text_numbers.csv", "", "auc", 1, ""),
    ("text.baseline", "Sec 5.1 (18.7%)", "18.7", L + "s13_text_numbers.csv", "", "control_baseline_primary", 100, ""),
    ("text.rel_primary", "Abstract (8.8%)", "8.8", L + "s13_text_numbers.csv", "", "relative_effect_primary_pct", 1, ""),
    ("text.rel_2days", "Abstract (14.4%)", "14.4", L + "s13_text_numbers.csv", "", "relative_effect_2plus_days_pct", 1, ""),
    ("text.n_raw_smd", "Sec 4.3 (9 covariates exceed)", "9", L + "s13_text_numbers.csv", "", "n_raw_smd_gt_0.1", 1, ""),
    ("text.max_raw_smd", "Sec 4.3 (max = 0.34)", "0.34", L + "s13_text_numbers.csv", "", "max_raw_smd", 1, ""),
    ("text.max_strat_smd", "Sec 4.3 (max stratified SMD = 0.033)", "0.033", L + "s13_text_numbers.csv", "",
     "max_strat_smd_legacy", 1, "paper-version definition (strata weighted by total size); for the ATT definition see max_strat_smd_att (I-35)"),
    ("text.outside_support", "Sec 4.3 (3 observations)", "3", L + "s13_text_numbers.csv", "",
     "n_outside_common_support", 1, ""),
    ("text.evalue_ci", "Sec 5.2, Limitations (E-value 1.080)", "1.080", L + "evalues.csv", "index == 2", "e_ci", 1,
     "I-19: the paper uses 1+ATT; here we compare against the corrected E-value"),
    ("text.ablation_mentor_auc", "Sec 4.1 (AUC 0.62 → 0.80)", "0.80", L + "A7_feature_ablation.csv",
     q(Model="Full + mentor features"), "AUC", 1, ""),
    ("text.ablation_mentor_att", "Sec 4.1 (ATT → +0.007)", "0.007", L + "A7_feature_ablation.csv",
     q(Model="Full + mentor features"), "ATT", 1, ""),
    ("text.48h_range_hi", "Sec 5.2 (all in [+0.015, +0.018], incl. 48h)", "0.018", L + "C6_robustness_summary.csv",
     q(Spec="D=48h"), "ATT", 1, "I-21: the ATT under the 48-hour definition is not in this range"),
    ("text.tech_retention", "Sec 6.3 (Technical retention +0.017)", "0.017", L + "subgroup_att.csv",
     q(Subgroup="Technical", Outcome="2+_active_days_14d"), "ATT", 1, ""),
    ("text.q5y_retention", "Sec 6.3 (Q5=Y retention +0.026)", "0.026", L + "subgroup_att.csv",
     q(Subgroup="Own work (Q5=Y)", Outcome="2+_active_days_14d"), "ATT", 1, ""),
    ("text.q5n_retention", "Sec 6.3 (Q5=N retention +0.014)", "0.014", L + "subgroup_att.csv",
     q(Subgroup="No own work (Q5=N)", Outcome="2+_active_days_14d"), "ATT", 1, ""),
]

#: Tables that later modules reproduce (stage A does not cover them). For now we only record which module will do it.
PENDING_TABLES = {
    "tab:qtypes": "M6 (shares of Q1 to Q5)", "tab:agreement": "M6 agreement.py (κ)",
    "tab:llm_full": "M6 agreement.py", "tab:iter1": "M6 agreement.py", "tab:llm_comparison": "M6 (model comparison)",
    "tab:sample": "M5 (sample construction table)", "tab:close_reading": "M8 close_reading.py (close reading of cases)",
}


def _rel_to_base(p) -> str:
    """The paper path relative to config.WIKI_BASE (no local home folders in the YAML)."""
    try:
        return str(Path(p).resolve().relative_to(config.WIKI_BASE))
    except ValueError:
        return Path(p).name


def main() -> None:
    paper = config.EXTERNAL["paper_md"]
    tex = paper.read_text()
    entries = build(tex)
    for id_, where, s, src, qq, col, scale, note in TEXT_NUMBERS:
        e = entry(id_, where, s, src, qq, col, note, scale=scale)
        if e:
            entries.append(e)
    doc = {"paper": _rel_to_base(paper), "results_root": "results/", "n_entries": len(entries),
           "pending_tables": PENDING_TABLES, "numbers": entries}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200))
    print(f"wrote {OUT}: {len(entries)} numbers")


if __name__ == "__main__":
    main()
