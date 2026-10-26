# Question-type annotation: codebooks, human labels and LLM labels

This folder holds everything needed to reproduce the question-type annotation (Morrison's information-seeking
types) and the agreement numbers in the paper (Table 3 and Appendix Table A6).

## Label names

The files use the codebook names Q0, Q2, Q3, Q4, Q5. The paper calls them Q1 to Q5:

| File column | Paper | Meaning |
|---|---|---|
| Q0 | Q1 Substantive | The message contains a question or request about Wikipedia editing |
| Q2 | Q2 Referent | The newcomer lacks task direction |
| Q3 | Q3 Appraisal | The newcomer asks for feedback on existing work |
| Q4 | Q4 Normative | The newcomer asks about rules, policies or norms |
| Q5 | Q5 Own work | The message shows the newcomer has already edited |

Values are Y or N. If Q0 = N, all other labels are N.

## Files

| File | What it is |
|---|---|
| `codebook_human.md` | The codebook the two human annotators used (English translation) |
| `codebook_human_original_zh.md` | The same codebook in its original Chinese, as used during Iteration 2 (2026-05-01 version) |
| `codebook_llm_prompt.md` | The codebook part of the LLM system prompt |
| `llm_fewshot_examples.xlsx` | Its first 120 rows are the labelled examples appended to the LLM prompt (validation set). This file is from an earlier labelling round than Iteration 2; it is kept exactly as the paper used it |
| `irr_sample_241.csv` | The 241 questions that were annotated by humans: `idx` (1 to 241), `sample_type`, `conversation_id` (paper-version id), question text and first reply. User names were removed |
| `irr_iteration2_annotator_a1.xlsx` | Annotator A1, Iteration 2 (independent re-annotation, 2026-04-30 to 2026-05-04). The consensus labels were also recorded by A1 |
| `irr_iteration2_annotator_a2.csv` | Annotator A2, Iteration 2 |
| `irr_consensus_labels.csv` | Consensus (gold-standard) labels after the Iteration 1 meeting |
| `irr_llm_labels_dsv4flash.csv` | DeepSeek-v4-Flash (non-thinking) labels for the 241 questions, taken from the full-corpus run below |
| `llm_labels_corpus_v2.jsonl` | DeepSeek-v4-Flash labels for the full corpus (one line per question, keyed by `cid`, the paper-version conversation id). M6 reuses these labels when the question text has not changed |

Rows of the IRR files line up by `idx`. The validation set is idx 1 to 120 (used to tune the prompt); the held-out test
set is idx 121 to 241.

## How to reproduce

From the repository root:

    python -m pipeline.m6_text.agreement            # writes results/m6/agreement.csv
    python -m pipeline.m6_text.agreement --online   # also re-labels the 241 questions with the LLM (needs DEEPSEEK_API_KEY)

Expected Cohen's kappa:

| Pair | Set | Q0 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|
| A1 vs A2 | all 241 | 0.872 | 0.950 | 0.890 | 0.851 | 0.926 |
| A1 vs LLM | test (121) | 0.904 | 0.815 | 0.900 | 0.754 | 0.834 |
| A2 vs LLM | test (121) | 0.809 | 0.800 | 0.844 | 0.677 | 0.784 |
| A1 vs LLM | all 241 | 0.924 | 0.861 | 0.936 | 0.769 | 0.847 |
| A2 vs LLM | all 241 | 0.887 | 0.856 | 0.911 | 0.739 | 0.854 |

The LLM system prompt is built by `pipeline/m6_text/services.codebook_text()` from `codebook_llm_prompt.md` plus the
first 120 rows of `llm_fewshot_examples.xlsx`. Its sha256 is
`6769731f953724e6839c221ef019c17a7a354ae81d4dab400c5fc5169c10e352` (53,017 characters); if it changes, cached LLM labels
no longer match the prompt.

## Where the files came from

Copied on 2026-10-25 from the authors' working folders (the two Iteration 2 annotation files and the two codebook
files) and from the paper repository `data/s10/` (`irr_golden_standard.csv`, `irr_annotation_a1.xlsx` as the few-shot
examples, `codebook_prompt.md`, `corpus_annotations_v2.jsonl`, and `irr_sample_200.csv`, which has 241 rows despite
its name). Annotator names were removed from file names.
