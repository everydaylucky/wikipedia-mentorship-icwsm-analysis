"""
agreement.py: agreement between labellers on question types (paper Table 3; PLAN M6 item 2; Issues I-31)

Module: M6 (CODE_PLAN.md section 5, M6)

1. What it does
    241 human-labelled examples (the 241 rows of irr_sample_200.csv):
        two labellers A1 and A2, the gold standard (gold), and the paper-version LLM (llm_fewshot_dsv4flash_nt.jsonl).
    For Q0, Q2, Q3, Q4, Q5 it computes Cohen's κ on three subsets, the same as the paper version s10_3:
        test set (items 121 to 241, never used while developing the prompt), independent set (drop the first 20
        calibration items, so items 21 to 241), and the full set.
    Pairs compared: A1-A2 (human vs human), LLM-A1, LLM-A2, LLM-gold (human vs LLM).
    With --online, it labels these 241 items again with services.annotate_llm (same prompt, same model,
    temperature 0) to check whether the LLM labels can be reproduced (agreement rate and κ of the new labels vs
    the paper-version LLM).

2. Why it is designed this way
    The paper must keep apart "human vs human κ ≥ 0.85" and "human vs LLM κ between 0.68 and 0.90" (I-31), so the
    two kinds are listed separately here.

4. Outputs
    results/m6/agreement.csv, results/m6/llm_reproducibility.csv (--online)

7. How to run
    python -m pipeline.m6_text.agreement [--online]
"""
from __future__ import annotations

import json
import sys

import pandas as pd

import config
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m6_text import services as S

OUT = config.RESULTS / "m6"


def to_yn(v) -> str:
    s = str(v).strip().upper()
    return s if s in ("Y", "N") else "N"


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's κ (same as cohens_kappa in the paper version s10_3)."""
    n = len(a)
    yy = sum(x == "Y" and y == "Y" for x, y in zip(a, b))
    nn = sum(x == "N" and y == "N" for x, y in zip(a, b))
    yn = sum(x == "Y" and y == "N" for x, y in zip(a, b))
    ny = sum(x == "N" and y == "Y" for x, y in zip(a, b))
    po = (yy + nn) / n
    pe = ((yy + yn) * (yy + ny) + (yn + nn) * (ny + nn)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main(argv=None):
    args = base_parser("M6: inter-annotator agreement and LLM reproducibility.").parse_args(argv)
    log = setup_logging("m6_agreement")
    OUT.mkdir(parents=True, exist_ok=True)
    a1 = pd.read_excel(config.EXTERNAL["irr_a1"])
    a2 = pd.read_csv(config.EXTERNAL["irr_a2"])
    gold = pd.read_csv(config.EXTERNAL["irr_gold"])
    # LLM labels: the full-corpus labels that the analysis uses (corpus_annotations_v2, keyed by conversation_id),
    # looked up for the 241 IRR questions through irr_sample_200.csv. This is what the paper's Table 3 compares;
    # the separate IRR-only run (llm_irr) gives slightly different numbers and is not used (I-54).
    a1, a2 = a1.sort_values("idx").reset_index(drop=True), a2.sort_values("idx").reset_index(drop=True)
    smp = pd.read_csv(config.EXTERNAL["irr_sample"]).sort_values("idx").reset_index(drop=True)
    if not (smp.idx.values == a1.idx.values).all() or not (a1.idx.values == a2.idx.values).all():
        raise ValueError("IRR files do not have the same idx order")
    corpus = {}
    for x in config.EXTERNAL["llm_annotations"].read_text().splitlines():
        if x.strip():
            d = json.loads(x)
            corpus[int(str(d.get("cid", d.get("id"))).replace("q_", ""))] = d
    missing = [int(c) for c in smp.conversation_id if int(c) not in corpus]
    if missing:
        raise ValueError(f"{len(missing)} IRR questions have no corpus LLM label, e.g. {missing[:5]}")
    L = {k: [to_yn(corpus[int(c)][k]) for c in smp.conversation_id] for k in S.DIMS}
    A1 = {k: [to_yn(v) for v in a1[k]] for k in S.DIMS}
    A2 = {k: [to_yn(v) for v in a2[k]] for k in S.DIMS}
    G = {k: [to_yn(v) for v in gold[k]] for k in S.DIMS}
    sets = {"test (121-241)": range(120, 241), "independent (21-241)": range(20, 241), "full (1-241)": range(0, 241)}
    pairs = {"A1-A2 (human-human)": (A1, A2), "LLM-A1": (L, A1), "LLM-A2": (L, A2), "LLM-gold": (L, G)}
    rows = []
    for sname, idx in sets.items():
        for pname, (x, y) in pairs.items():
            for k in S.DIMS:
                rows.append({"set": sname, "pair": pname, "dim": k, "n": len(idx),
                             "kappa": kappa([x[k][i] for i in idx], [y[k][i] for i in idx])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "agreement.csv", index=False)
    log.info("kappa (test set):\n%s", df[df.set.str.startswith("test")].pivot(index="pair", columns="dim", values="kappa").round(3))
    if args.online:
        texts = list(gold.question.astype(str))
        got = S.annotate_llm(texts, log)
        rep = []
        for k in S.DIMS:
            new = [got.get(t, {}).get(k, "?") for t in texts]
            ok = [i for i, v in enumerate(new) if v in ("Y", "N")]
            rep.append({"dim": k, "n": len(ok), "agree_with_paper_llm": sum(new[i] == L[k][i] for i in ok) / len(ok),
                        "kappa_vs_paper_llm": kappa([new[i] for i in ok], [L[k][i] for i in ok]),
                        "kappa_vs_gold": kappa([new[i] for i in ok], [G[k][i] for i in ok])})
        pd.DataFrame(rep).to_csv(OUT / "llm_reproducibility.csv", index=False)
        log.info("LLM reproducibility:\n%s", pd.DataFrame(rep).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
