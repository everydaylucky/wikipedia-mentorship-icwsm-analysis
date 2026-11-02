"""
camera_ready_registry.py: where every number in the camera-ready paper comes from (used by check_camera_ready.py)

Module: M9 (camera-ready; added 2026-11-02)

TABLES: for each table label, the result file, how many leading cells of a row are labels, and one entry per
    table row (in the order of the rows after the table's \\midrule): (text the row label must contain, pandas query
    that picks one row of the result file, list of column specs). The numbers in the non-label cells of the paper
    row are matched to the column specs in order. A column spec is
        "Col" or "Col*k"                      a column of the selected row, multiplied by k;
        {"q": ..., "col": ..., "scale": k,    another row (or another file) for this one number;
         "source": ...}
        None                                  the number is not checked (written next to it: why).
TEXT: numbers in the running text: (id, a snippet of the LaTeX source that contains the number, the number as
    printed, result file, query, column, scale).
UNCHECKED: tables and numbers without a result file, and why.
"""
from __future__ import annotations


def nv(name: str, scale: float = 1, source: str | None = None) -> dict:
    """One value of a name/value file (paper_counts.csv, mentor_side_summary.csv)."""
    d = {"q": f"name == '{name}'", "col": "value", "scale": scale}
    if source:
        d["source"] = source
    return d


ATT3 = ["ATT", "CI_lo", "CI_hi"]
PCT = ["ATT*100", "CI_lo*100", "CI_hi*100"]
B = "Outcome == '{}'"
OUTC = {"1+ edit": "1+_mainspace_edit_14d", "N edits": "n_mainspace_edits_14d", "Unique NS": "unique_namespace_14d",
        "2+ article-edit days": "2+_active_days_14d", "Active days": "active_days_30d", "Reverted": "reverted_any_14d"}
AGREE = "set == '{}' and pair == '{}' and dim == '{}'"

TABLES: dict[str, dict] = {
    "tab:qtypes": {"source": "m8/paper_counts.csv", "label_cells": 3, "rows": [
        (lab, None, [nv(f"qtype share: q_{c}", 100)]) for lab, c in
        (("Substantive", "substantive"), ("Referent", "referent"), ("Appraisal", "appraisal"),
         ("Normative", "normative"), ("Own work", "own_work"))]},
    "tab:agreement": {"source": "m6/agreement.csv", "label_cells": 1, "rows": [
        (f"Q{k}", None, [{"q": AGREE.format("full (1-241)", "A1-A2 (human-human)", d), "col": "kappa", "scale": 1},
                         {"q": AGREE.format("test (121-241)", "LLM-A1", d), "col": "kappa", "scale": 1},
                         {"q": AGREE.format("test (121-241)", "LLM-A2", d), "col": "kappa", "scale": 1}])
        for k, d in ((1, "Q0"), (2, "Q2"), (3, "Q3"), (4, "Q4"), (5, "Q5"))]},
    "tab:llm_full": {"source": "m6/agreement.csv", "label_cells": 1, "rows": [
        (f"Q{k}", None, [{"q": AGREE.format("full (1-241)", "LLM-A1", d), "col": "kappa", "scale": 1},
                         {"q": AGREE.format("full (1-241)", "LLM-A2", d), "col": "kappa", "scale": 1}])
        for k, d in ((1, "Q0"), (2, "Q2"), (3, "Q3"), (4, "Q4"), (5, "Q5"))]},
    "tab:att": {"source": "m8/B_main_results.csv", "label_cells": 1, "rows": [
        (lab, B.format(OUTC[lab]), ["Naive", "ATT", "CI_lo", "CI_hi"])
        for lab in ("1+ edit", "N edits", "Unique NS", "2+ article-edit days", "Active days")]},
    "tab:later": {"source": "m8/followup_cohort_periods.csv", "label_cells": 1, "rows": [
        (lab, f"period == 'days {per}' and threshold == '{th}'",
         ["Treat_mean", "Ctrl_mean_reweighted", "ATT", "CI_lo", "CI_hi", "q_BH"])
        for th in ("1+", "5+") for lab, per in (("1--14", "1-14"), ("15--28", "15-28"), ("29--60", "29-60"),
                                                ("61--180", "61-180"), ("181--365", "181-365"))]},
    "tab:subgroup": {"source": "m8/subgroup_att.csv", "label_cells": 1, "rows": [
        (lab, f"Subgroup == '{sg}' and Outcome == '1+_mainspace_14d'", ["N", "ATT", "CI_lo", "CI_hi"])
        for lab, sg in (("Full sample", "Full sample"), ("Technical", "Technical"), ("Normative", "Normative (Q4)"),
                        ("Appraisal", "Appraisal (Q3)"), ("Referent", "Referent (Q2)"),
                        ("Non-substantive", "Non-substantive"), ("Own work", "Own work (Q5=Y)"),
                        ("No own work", "No own work (Q5=N)"))]},
    "tab:reply_full": {"source": "m8/reply_association.csv", "label_cells": 1, "rows": [
        (lab, f"Feature == '{f}'", ["Beta_std", "p"]) for lab, f in
        (("Reply lag", "Reply lag (hours)"), ("cosine", "Q-R semantic similarity"), ("VADER", "VADER compound sentiment"),
         ("We", '"We" rate (per word)'), ("TextBlob", "TextBlob subjectivity"), ("Has list", "Has list"))]},
    "tab:covariates": {"source": "m8/paper_counts.csv", "label_cells": 1, "rows": [
        ("editing history", None, [nv("covariates: E")]), ("Question text", None, [nv("covariates: Qtext")]),
        ("Perspective", None, [nv("covariates: Qpersp")]), ("Morrison", None, [nv("covariates: Qtype")]),
        ("embeddings", None, [nv("covariates: emb20")]),
        ("Temporal", None, [nv("covariates: temporal"), nv("covariates: year-month indicators"),
                            None])]},          # 2 missing-data indicators: 62 - 59 - 1 (article context)
    "tab:strata_sens": {"source": "m8/C1_strata_sensitivity.csv", "label_cells": 1, "rows": [
        (str(k), f"Outcome == '1+_mainspace_edit_14d' and Strata == {k}", ATT3) for k in (5, 10, 20)]},
    "tab:ablation": {"source": "m8/A7_feature_ablation.csv", "label_cells": 1, "rows": [
        (lab, f"Model == '{m}'", ["p", "AUC", "ATT"]) for lab, m in
        (("Full model", "Full model (E+Qtext+Qpersp+Qtype+Emb+T)"), ("Qtype", "Full – Qtype (no LLM annotation)"),
         ("temporal", "Full – temporal"), ("embeddings", "Full – embeddings"), ("Qtext", "Full – Qtext"),
         ("Perspective", "Full – Qpersp"), ("Edit history only", "Edit history only"),
         ("Question feat", "Question features only (Qtext+Qpersp+Qtype)"), ("Qtype only", "Qtype only"),
         ("mentor", "Full + mentor features"))]},
    "tab:trimming": {"source": "m8/C3_ps_trimming.csv", "label_cells": 1, "rows": [
        (lab, f"Range == '{r}'", ["Trimmed", "ATT"]) for lab, r in
        (("0.01", "[0.01,0.99]"), ("0.05", "[0.05,0.95]"), ("0.10", "[0.1,0.9]"), ("0.15", "[0.15,0.85]"))]},
    "tab:later_all": {"source": "m8/followup_later_periods.csv", "label_cells": 1, "rows": [
        (f"{t}+ edits, days {lab}", f"period == 'days {per}' and threshold == '{t}plus'",
         ["N", "Ctrl_mean", "ATT", "CI_lo", "CI_hi"])
        for lab, per in (("15--28", "15-28"), ("29--60", "29-60"), ("61--180", "61-180")) for t in (1, 5)]},
    "tab:window": {"source": "m8/window_followup_complete.csv", "label_cells": 1, "rows": [
        (f"{w} days", f"window == {w}", ["N", "ATT", "CI_lo", "CI_hi"]) for w in (7, 14, 21, 28, 30, 60, 180)]},
    "tab:strata_details": {"source": "m8/A3_strata_diagnostics.csv", "label_cells": 1, "rows": [
        (str(k), f"Stratum == {k}", ["N", "Treated", "Control", "PS_lo", "PS_hi"]) for k in range(10)]},
    "tab:sample": {"source": "m8/paper_counts.csv", "label_cells": 2, "rows": [
        ("Tagged first questions", None, [nv("sample: tagged first questions")])] + [
        (lab, None, [nv(f"sample: removed: {lab}", -1)]) for lab in
        ("Account created by another user", "Automatically created account", "Registered before mentorship began",
         "No account creation record", "Claimed by a mentor before asking",
         "Other account exclusions (test, forced local)", "Conversation not found",
         "Reply recorded before the question", "Question removed in the next edit", "Question text hidden or empty",
         "Not in English")] + [("Final sample", None, [nv("sample: final")])]},
    "tab:robust_specs": {"source": "m8/robustness.csv", "label_cells": 1, "rows": [
        (lab, f"Spec == '{s}'", ["N", "ATT", "CI_lo", "CI_hi"]) for lab, s in
        (("Main", "Main (D10 sample)"), ("Assigned mentor", "Treatment = assigned mentor replied (vs no reply)"),
         ("No account-type", "No D10 restriction (R2.5)"), ("late-fetched", "Exclude late-fetched askers (M3)"),
         ("30 days before cutoff", "Questions ≥ 30 days before the cutoff (I-11)"),
         ("without full 14-day", "Also people without a full 14-day follow-up (before I-10)"),
         ("Complete text", "Complete text annotations (no imputation)"),
         ("reviewed sample", "Askers also in the paper sample"))] + [
        ("Reply within 48 hours", "Spec == 'Main (D10 sample)'",
         ["N"] + [{"source": "m8/C6_robustness_summary.csv", "q": "Spec == 'D=48h'", "col": c, "scale": 1}
                  for c in ATT3])]},
    "tab:mentor_reply": {"source": "m8/mentor_reply_versions.csv", "label_cells": 2, "rows": [
        ("assigned mentor", None, [nv("who replied: (a) page mentor first", source="m8/paper_counts.csv")]),
        ("Someone else first", None, [nv("who replied: (b) other first, mentor later", source="m8/paper_counts.csv")]),
        ("Only other", None, [nv("who replied: (c) only others", source="m8/paper_counts.csv")]),
        ("No reply", None, [nv("who replied: (d) no reply", source="m8/paper_counts.csv")])] + [
        (lab, f"Spec == \"{s}\"", ATT3) for lab, s in
        (("anyone replied", "Anyone replied vs no reply (main)"),
         ("(a)(b) vs.\\ (c)(d)", "Assigned mentor replied vs no reply or only others"),
         ("(a)(b) vs.\\ (d)", "Assigned mentor replied vs no reply (only others dropped)"),
         ("first reply", "Same, measured from the assigned mentor's first reply"))]},
    "tab:full_boot": {"source": "m8/full_bootstrap.csv", "label_cells": 1, "rows": [
        (lab, B.format(OUTC[lab]), ["ATT", "fixed_CI_lo", "fixed_CI_hi", "full_CI_lo", "full_CI_hi"])
        for lab in ("1+ edit", "N edits", "Active days", "Unique NS", "Reverted", "2+ article-edit days")]},
    "tab:ps_models": {"source": "m8/ps_model_comparison.csv", "label_cells": 1, "rows": [
        (lab, f"Model == '{m}'", ["AUC", "Max_SMD_after", "Mean_SMD_after"] + ATT3) for lab, m in
        (("C = 1", "L2 logit, C = 1 (main)"), ("C = 0.01", "L2 logit, C = 0.01"), ("C = 0.1", "L2 logit, C = 0.1"),
         ("C = 10", "L2 logit, C = 10"), ("leaf \\geq 50", "Random forest, min leaf 50"), ("leaf \\geq 5", "Random forest, min leaf 5"),
         ("Gradient", "Gradient boosting"))]},
    "tab:anchor": {"source": "m8/C7_pseudo_time.csv", "label_cells": 1, "rows": [
        ("treated lags (main)", "Strategy == 'Random draw (main)'", ATT3),
        ("Mean of 20 random", "Strategy == 'Multi-draw avg'", ATT3),
        ("Random draw within", None, [{"source": "m8/anchor_within_stratum.csv", "q": "Spec == 'one draw within the PS stratum'",
                                       "col": c, "scale": 1} for c in ATT3]),
        ("Mean of 20 draws within", None, [{"source": "m8/anchor_within_stratum.csv",
                                            "q": "Spec == 'mean of 20 draws within the PS stratum'", "col": c, "scale": 1}
                                           for c in ATT3]),
        ("Median", "Strategy == 'Median lag'", ATT3), ("25th", "Strategy == '25th pctl lag'", ATT3),
        ("75th", "Strategy == '75th pctl lag'", ATT3), ("No lag", "Strategy == 'Zero lag (immediate)'", ATT3),
        ("Both groups from the question", None, [{"source": "m8/followup_question_anchor.csv", "q": "N == N",
                                                  "col": c, "scale": 1} for c in ATT3])]},
    "tab:mentor_side": {"source": "m8/mentor_side_summary.csv", "label_cells": 1, "rows": [
        ("Past reply rate", None, [nv("past_rate_bottom", 100), nv("past_rate_top", 100)]),
        ("Newcomers", None, [nv("n_bottom"), nv("n_top")]),
        ("Received a reply", None, [nv("reply_share_bottom", 100), nv("reply_share_top", 100)]),
        ("SMD|, all", None, [nv("max_abs_smd_askers")]),
        ("Without a reply", None, [nv("n_bottom_no_reply"), nv("n_top_no_reply")]),
        ("SMD|, no reply", None, [nv("max_abs_smd_no_reply")]),
        ("Per 10 points (adjusted)", None, [nv("nc_adj_slope_per_10pt", 100), nv("nc_adj_CI_lo", 100),
                                            nv("nc_adj_CI_hi", 100)]),
        ("Bottom to top", None, [nv("nc_adj_implied_bottom_to_top", 100), nv("nc_adj_implied_CI_lo", 100),
                                 nv("nc_adj_implied_CI_hi", 100)]),
        ("Per 10 points (strata)", None, [nv("nc_strata_slope_per_10pt", 100), nv("nc_strata_CI_lo", 100),
                                          nv("nc_strata_CI_hi", 100)])]},
    "tab:quality": {"source": "m8/quality_cuts.csv", "label_cells": 1, "rows": [
        (lab, f"cut == '{c}'", [f"g2_{x}" for x in ("Ctrl_reweighted", "ATT", "CI_lo", "CI_hi")]
         + [f"g3_{x}" for x in ("Ctrl_reweighted", "ATT", "CI_lo", "CI_hi")])
        for lab, c in (("At least one edit", "at least one edit"), ("50", "50%"), ("60", "60%"), ("70", "70%"),
                       ("80 (main", "80%"), ("90", "90%"), ("All edits", "all edits"))]},
    "tab:quality_self": {"source": "m8/quality_self_versions.csv", "label_cells": 2, "rows": [
        (lab, f"self_reverts == '{a}' and unmatched == '{b}'",
         ["ctrl_mean_rejected_share*100"] + [f"g{g}_{x}" for g in (2, 3) for x in ("ATT", "CI_lo", "CI_hi")])
        for lab, a, b in (("Left out", "leave out", "rejected"), ("Left out", "leave out", "kept"),
                          ("Rejected", "rejected", "rejected"), ("Kept", "kept", "rejected"))]},
    "tab:lag_strata": {"source": "m8/anchor_lag_strata.csv", "label_cells": 1, "rows": [
        (str(k), f"stratum == {k}", ["treated", "p25_hours", "median_hours", "p75_hours"]) for k in range(1, 11)]},
    "tab:landmark": {"source": "m8/anchor_landmark.csv", "label_cells": 1, "rows": [
        (lab, f"L_hours == {h} and controls == 'no reply within L'", ["N", "Treated"] + ATT3)
        for lab, h in (("12 h", 12), ("24 h", 24), ("48 h", 48), ("72 h", 72), ("7 days", 168))]},
    "tab:trajectories": {"source": "m8/editor_kinds_levels.csv", "label_cells": 1, "rows": [
        (lab, f"outcome == '{pre}{o}'", ["treated_share*100", "ctrl_share_reweighted*100"] + PCT
         + (["q_BH"] if pre == "" else []))
        for pre in ("", "days 15-365 only: ", "article edits only: ") for lab, o in
        (("10 edits", "10 edits"), ("Active", "active: 5 edits in a 30-day window"),
         ("Very active", "very active: 100 edits in a 30-day window"), ("500 edits", "500 edits"))]},
    "tab:editor_type": {"source": "m8/editor_kinds_types.csv", "label_cells": 1, "rows": [
        (lab, f"outcome == 'type: {t}'", ["treated_share*100", "ctrl_share_reweighted*100"] + PCT) for lab, t in
        (("No edits", "no edits after day 14"), ("articles", "articles"), ("drafts", "drafts"),
         ("user and talk", "user_and_talk"), ("other", "other"), ("Mixed", "mixed"))]},
    "tab:conversation": {"source": "m8/descriptive_conversation.csv", "label_cells": 1, "rows": [
        ("Wrote back", "measure == 'mentee wrote back after the first reply'", ["value*100"]),
        ("more than one reply", "measure == 'a mentor replied more than once'", ["value*100"]),
        ("Median turns", "measure == 'median turns in the thread'", ["value"]),
        ("Median time", "measure == 'median days the thread was active'", ["value*24"]),
        ("later question", "measure == 'asked at least one later question'", ["value*100"]),
        ("same mentor", "measure == \"later question to the same mentor's page\"", ["value*100"]),
        ("if wrote back", None, [{"source": "m8/descriptive_followup_association.csv", "q": "N == N",
                                  "col": c, "scale": 100} for c in ("share_continued_wrote_back", "share_continued_no")]),
        ("difference within PS strata", None, [{"source": "m8/descriptive_followup_association.csv", "q": "N == N",
                                                "col": "difference_within_strata", "scale": 100}])]},
}

_SG = ["Technical", "Referent (Q2)", "Normative (Q4)", "Non-substantive", "Own work (Q5=Y)", "No own work (Q5=N)"]
_RF = [("Reply lag", "Reply lag (hours)"), ("cosine", "Q-R semantic similarity"), ("VADER", "VADER compound sentiment"),
       ("We", '"We" rate (per word)'), ("TextBlob", "TextBlob subjectivity"), ("Has list", "Has list"),
       ("Question marks", "Question marks in reply"), ("Positive", "Positive language"), ("I''", '"I" rate (per word)'),
       ("You", '"You" rate (per word)'), ("word count", "Reply word count"), ("ratio", "Reply/question word ratio"),
       ("Flesch", "Flesch-Kincaid grade"), ("Imperative", "Imperative sentences"), ("greeting", "Has greeting"),
       ("Gratitude", "Gratitude expression"), ("Deference", "Deference"), ("Policy", "Policy references"),
       ("Help page", "Help page references")]
TABLES["tab:reply_subgroup_full"] = {"source": "m8/reply_association_by_subgroup.csv", "label_cells": 1, "rows": [
    (lab, None, [{"q": f"Subgroup == '{sg}' and Feature == '{f}'", "col": "Beta_std", "scale": 1} for sg in _SG])
    for lab, f in _RF]}

# (id, snippet of the LaTeX source, number as printed, source, query, column, scale)
S13 = "m8/s13_text_numbers.csv"
TEXT: list[tuple] = [
    ("abstract: sample", "first questions that 34,073 newcomers", "34,073", "m8/paper_counts.csv", "name == 'sample: final'", "value", 1),
    ("abstract: ATT", "were 1.8 percentage points (pp; 10.5\\% relative)", "1.8", "m8/B_main_results.csv", B.format("1+_mainspace_edit_14d"), "ATT", 100),
    ("abstract: relative", "(pp; 10.5\\% relative)", "10.5", S13, "n == n", "relative_effect_primary_pct", 1),
    ("abstract: 2+ days", "and 1.8pp (20\\% relative) more likely", "1.8", "m8/B_main_results.csv", B.format("2+_active_days_14d"), "ATT", 100),
    ("abstract: 2+ days relative", "1.8pp (20\\% relative)", "20", S13, "n == n", "relative_effect_2plus_days_pct", 1),
    ("abstract: normative", "rules and norms (+3.4pp)", "3.4", "m8/subgroup_att.csv", "Subgroup == 'Normative (Q4)' and Outcome == '1+_mainspace_14d'", "ATT", 100),
    ("abstract: own work", "earlier editing (+3.4pp)", "3.4", "m8/subgroup_att.csv", "Subgroup == 'Own work (Q5=Y)' and Outcome == '1+_mainspace_14d'", "ATT", 100),
    ("abstract: technical", "how-to questions (+2.4pp)", "2.4", "m8/subgroup_att.csv", "Subgroup == 'Technical' and Outcome == '1+_mainspace_14d'", "ATT", 100),
    ("estimation: AUC", "AUC = 0.626; see Section", "0.626", S13, "n == n", "auc", 1),
    ("estimation: tree AUC low", "predict no better (AUC 0.616 to 0.620)", "0.616", "m8/ps_model_comparison.csv", "Model == 'Random forest, min leaf 5'", "AUC", 1),
    ("estimation: tree AUC high", "predict no better (AUC 0.616 to 0.620)", "0.620", "m8/ps_model_comparison.csv", "Model == 'Random forest, min leaf 50'", "AUC", 1),
    ("estimation: tree SMD low", "(largest stratified SMD 0.051 to 0.065, against 0.027)", "0.051", "m8/ps_model_comparison.csv", "Model == 'Gradient boosting'", "Max_SMD_after", 1),
    ("estimation: tree SMD high", "(largest stratified SMD 0.051 to 0.065, against 0.027)", "0.065", "m8/ps_model_comparison.csv", "Model == 'Random forest, min leaf 5'", "Max_SMD_after", 1),
    ("diagnostics: outside support", "Only 2 observations (0.01\\%)", "2", S13, "n == n", "n_outside_common_support", 1),
    ("diagnostics: raw SMD count", "In the raw data, 9 covariates exceed", "9", S13, "n == n", "n_raw_smd_gt_0.1", 1),
    ("diagnostics: raw SMD max", "exceed this threshold (max = 0.40)", "0.40", S13, "n == n", "max_raw_smd", 1),
    ("diagnostics: strat SMD max", "(max stratified SMD = 0.027)", "0.027", S13, "n == n", "max_strat_smd_att", 1),
    ("overall: naive", "within 14 days is 4.6pp", "4.6", "m8/B_main_results.csv", B.format("1+_mainspace_edit_14d"), "Naive", 100),
    ("overall: relative", "a relative increase of 10.5\\%", "10.5", S13, "n == n", "relative_effect_primary_pct", 1),
    ("overall: baseline", "control baseline of 17.5\\%", "17.5", S13, "n == n", "control_baseline_primary", 100),
    ("overall: active days", "over the following month (+0.13)", "0.13", "m8/B_main_results.csv", B.format("active_days_30d"), "ATT", 1),
    ("overall: namespaces", "in more namespaces (+0.028)", "0.028", "m8/B_main_results.csv", B.format("unique_namespace_14d"), "ATT", 1),
    ("overall: edits", "(+0.78 in 14 days)", "0.78", "m8/B_main_results.csv", B.format("n_mainspace_edits_14d"), "ATT", 1),
    ("overall: mentor boot lo", "instead of newcomers ([+0.03, +1.53])", "0.03", "m8/B_mentor_boot.csv", B.format("n_mainspace_edits_14d"), "Mentor_CI_lo", 1),
    ("overall: mentor boot hi", "instead of newcomers ([+0.03, +1.53])", "1.53", "m8/B_mentor_boot.csv", B.format("n_mainspace_edits_14d"), "Mentor_CI_hi", 1),
    ("overall: full boot lo", "in the full bootstrap ([+0.17, +1.55]", "0.17", "m8/full_bootstrap.csv", B.format("n_mainspace_edits_14d"), "full_CI_lo", 1),
    ("overall: zero edits", "with 79\\% of newcomers making no article edit", "79", "m8/descriptive_edit_count_distribution.csv", "N == N", "share_zero", 100),
    ("quality: group 2", "raises the share in group 2 by 1.9pp", "1.9", "m8/quality_cuts.csv", "cut == '80%'", "g2_ATT", 100),
    ("quality: group 2 ctrl", "(15.2\\% among reweighted controls", "15.2", "m8/quality_cuts.csv", "cut == '80%'", "g2_Ctrl_reweighted", 100),
    ("persistence: cohort", "We therefore follow the 19,512 newcomers", "19,512", "m8/followup_cohort_periods.csv", "period == 'days 1-14' and threshold == '1+'", "N", 1),
    ("persistence: 1-14 1+", "at least once (+2.5pp, 95\\% CI [+1.1, +3.9])", "2.5", "m8/followup_cohort_periods.csv", "period == 'days 1-14' and threshold == '1+'", "ATT", 100),
    ("persistence: 1-14 5+", "at least five times (+1.6pp, [+0.6, +2.6])", "1.6", "m8/followup_cohort_periods.csv", "period == 'days 1-14' and threshold == '5+'", "ATT", 100),
    ("persistence: 181-365", "days 181--365 (+1.5pp, 95\\% CI [+0.5, +2.4], $q = .04$)", "1.5", "m8/followup_cohort_periods.csv", "period == 'days 181-365' and threshold == '1+'", "ATT", 100),
    ("persistence: 181-365 5+", "same period is +0.5pp ($q = .26$)", "0.5", "m8/followup_cohort_periods.csv", "period == 'days 181-365' and threshold == '5+'", "ATT", 100),
    ("editors: 10 edits", "made 10 edits by 2.1pp (95\\% CI [+0.7, +3.3], $q = .006$)", "2.1", "m8/editor_kinds_levels.csv", "outcome == '10 edits'", "ATT", 100),
    ("editors: active", "active level by 3.0pp ([+1.5, +4.3], $q < .001$)", "3.0", "m8/editor_kinds_levels.csv", "outcome == 'active: 5 edits in a 30-day window'", "ATT", 100),
    ("editors: very active", "very active level (+0.5pp, $q = .08$)", "0.5", "m8/editor_kinds_levels.csv", "outcome == 'very active: 100 edits in a 30-day window'", "ATT", 100),
    ("editors: 500", "on 500 edits (+0.4pp, $q = .06$)", "0.4", "m8/editor_kinds_levels.csv", "outcome == '500 edits'", "ATT", 100),
    ("editors: after day 14, 10 edits", "on 10 edits (+0.9pp) and on the active level (+0.5pp)", "0.9", "m8/editor_kinds_levels.csv", "outcome == 'days 15-365 only: 10 edits'", "ATT", 100),
    ("editors: after day 14, active", "on 10 edits (+0.9pp) and on the active level (+0.5pp)", "0.5", "m8/editor_kinds_levels.csv", "outcome == 'days 15-365 only: active: 5 edits in a 30-day window'", "ATT", 100),
    ("editors: type test", "do not differ between the groups ($p = .25$", ".25", "m8/editor_kinds_wald.csv", "df == df", "p", 1),
    ("robustness: full boot lo", "95\\% CI of [+1.2, +3.2] for the 1.8pp", "1.2", "m8/full_bootstrap.csv", B.format("1+_mainspace_edit_14d"), "full_CI_lo", 100),
    ("robustness: full boot hi", "95\\% CI of [+1.2, +3.2] for the 1.8pp", "3.2", "m8/full_bootstrap.csv", B.format("1+_mainspace_edit_14d"), "full_CI_hi", 100),
    ("robustness: E-value", "The E-value is 1.41", "1.41", "m8/evalues.csv", "method.str.startswith('Fixed')", "e_point", 1),
    ("robustness: E-value CI", "and 1.20 for the lower confidence limit", "1.20", "m8/evalues.csv", "method.str.startswith('Fixed')", "e_ci", 1),
    ("anchor: question start", "the increase is 0.9pp (95\\% CI [$-$0.3, +2.1])", "0.9", "m8/followup_question_anchor.csv", "N == N", "ATT", 100),
    ("anchor: 48 h", "it is 0.8pp (95\\% CI [$-$0.1, +1.7])", "0.8", "m8/anchor_landmark.csv", "L_hours == 48 and controls == 'no reply within L'", "ATT", 100),
    ("anchor: 48 h hi", "it is 0.8pp (95\\% CI [$-$0.1, +1.7])", "+1.7", "m8/anchor_landmark.csv", "L_hours == 48 and controls == 'no reply within L'", "CI_hi", 100),
    ("anchor: question 8.1", "appear 8.1pp less likely", "8.1", "m8/C7_pseudo_time.csv", "Strategy == 'Zero lag (immediate)'", "ATT", -100),
    ("anchor: 75th 3.7", "appear 3.7pp more likely", "3.7", "m8/C7_pseudo_time.csv", "Strategy == '75th pctl lag'", "ATT", 100),
    ("A.4: no reply 24 h", "Among the 13,561 newcomers", "13,561", "m8/anchor_reply_timing.csv", "B == B", "n_no_reply_24h", 1),
    ("A.4: edited 24 h", "($n = 2{,}971$)", "2{,}971", "m8/anchor_reply_timing.csv", "B == B", "n_edited_24h", 1),
    ("A.4: edited share", "slightly more often (35.4\\% vs.\\ 31.7\\%)", "35.4", "m8/anchor_reply_timing.csv", "B == B", "reply_24_168h_edited", 100),
    ("A.4: not edited share", "slightly more often (35.4\\% vs.\\ 31.7\\%)", "31.7", "m8/anchor_reply_timing.csv", "B == B", "reply_24_168h_not_edited", 100),
    ("A.4: within strata", "Within propensity strata the difference is +0.008", "0.008", "m8/anchor_reply_timing.csv", "B == B", "diff_within_strata", 1),
    ("A.4: within strata lo", "the difference is +0.008 [$-$0.011, +0.028]", "$-$0.011", "m8/anchor_reply_timing.csv", "B == B", "CI_lo", 1),
    ("landmark caption: never replied", "the 48-hour estimate is +.014 [+.002, +.024]", "+.014", "m8/anchor_landmark.csv", "L_hours == 48 and controls == 'never replied'", "ATT", 1),
    ("landmark caption: never replied lo", "the 48-hour estimate is +.014 [+.002, +.024]", "+.002", "m8/anchor_landmark.csv", "L_hours == 48 and controls == 'never replied'", "CI_lo", 1),
    ("landmark caption: never replied hi", "the 48-hour estimate is +.014 [+.002, +.024]", "+.024", "m8/anchor_landmark.csv", "L_hours == 48 and controls == 'never replied'", "CI_hi", 1),
    ("lags: lowest median", "with medians of 4.9 to 6.4 hours", "4.9", "m8/anchor_lag_strata.csv", "median_hours == median_hours.min()", "median_hours", 1),
    ("lags: highest median", "with medians of 4.9 to 6.4 hours", "6.4", "m8/anchor_lag_strata.csv", "median_hours == median_hours.max()", "median_hours", 1),
    ("lags: lowest 75th", "75th percentiles of 17 to 21 hours", "17", "m8/anchor_lag_strata.csv", "p75_hours == p75_hours.min()", "p75_hours", 1),
    ("lags: highest 75th", "75th percentiles of 17 to 21 hours", "21", "m8/anchor_lag_strata.csv", "p75_hours == p75_hours.max()", "p75_hours", 1),
    ("A.4: one draw", "gives +0.017 [+0.005, +0.028] with one draw", "0.017", "m8/anchor_within_stratum.csv", "Spec == 'one draw within the PS stratum'", "ATT", 1),
    ("A.4: 20 draws hi", "+0.017 [+0.007, +0.027] averaged over 20 draws", "+0.027", "m8/anchor_within_stratum.csv", "Spec == 'mean of 20 draws within the PS stratum'", "CI_hi", 1),
]

UNCHECKED = {
    "tab:close_reading": "close reading of 54 cases coded by hand (no result file)",
    "tab:iter1": "first annotation iteration (kappa and agreement); no code in this repository",
    "tab:llm_comparison": "comparison of five LLMs on the validation set; no code in this repository",
    "tab:outcomes": "definitions only",
    "tab:features": "definitions only",
    "tab:covariates_full": "list of covariate names",
}
