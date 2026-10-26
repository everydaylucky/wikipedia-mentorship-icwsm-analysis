# Who Benefits from Wikipedia's Growth Mentorship?

Code for the paper *"Who Benefits from Wikipedia's Growth Mentorship? The Role of Newcomer Information Needs,"* accepted at ICWSM 2027.

English Wikipedia assigns every new account a volunteer mentor, and newcomers can send that mentor a question. This
repository rebuilds the full study from public Wikipedia data:

1. who was a mentor and when (the mentor roster),
2. every question sent through the mentorship module and the conversation that followed,
3. each newcomer's editing activity before and after the question,
4. which newcomers were randomly assigned a mentor,
5. the analysis sample, the treatment (did someone reply?) and the outcomes (did the newcomer keep editing?),
6. text features of questions and replies, including question types labelled with an LLM,
7. the propensity-score dataset,
8. all estimates in the paper (stratified ATT, subgroups, robustness checks, reply-feature models),
9. a table that compares every number in the paper with the rebuilt result.

Everything is frozen at **2026-04-20 14:00:00 UTC**. Questions start with the first mentorship question (2021-05-04).
Records after the cutoff are dropped at every step, so a rerun today gives the same data as the paper.

## Repository layout

| Path | What it holds |
|---|---|
| `config.py` | All settings in one place: the cutoff, file locations, model and analysis parameters |
| `pipeline/m1_roster/` ... `pipeline/m9_paper/` | One folder per step (M1 to M9, listed above) |
| `pipeline/common/` | Shared helpers: cached Wikipedia API client, text cleaning, user names, time handling |
| `run_all.py` | Runs all offline steps in order |
| `tests/` | Automatic checks (pytest) |
| `checks/` | One notebook per step that puts raw records next to processed values for random samples |
| `data/annotation/` | Codebooks, human labels and LLM labels for the question types ([details](data/annotation/README.md)) |
| `tools/` | Helpers that extract the numbers from the paper source for the M9 comparison |

Every Python file starts with a short description: what it does, why, its inputs and outputs, possible errors, how to
run it and how to check it.

## Setup

Python 3.11 is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Language detection uses fastText's language identification model. Download `lid.176.bin` from
<https://fasttext.cc/docs/en/language-identification.html> and save it as `data/raw/models/lid.176.bin`.

Copy `.env.example` to `.env` and fill it in:

| Setting | Needed for |
|---|---|
| `CONTACT_EMAIL` | Every step that goes online. It is sent in the User-Agent header, as Wikimedia's [User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy) asks. Use your own address |
| `DEEPSEEK_API_KEY` | M6: LLM question-type labels (DeepSeek `deepseek-v4-flash`) |
| `PERSPECTIVE_API_KEY` | M6: Perspective API scores |
| `DASHSCOPE_API_KEY` | M6: Qwen `text-embedding-v4` embeddings |

## Input data

The pipeline starts from data that was collected from Wikipedia before the cutoff and then checks and completes it
against Wikipedia as it was at the cutoff. These input files (mentor roster revisions, tagged question revisions,
newcomer profiles and contributions, and the raw API responses collected by this code) are too large for GitHub.
They are on Hugging Face: **[Hugging Face dataset link to be added]**.

After downloading:

- put the folders of the snapshot next to this repository, so that the folder names match the paths in `config.py`
  (section 2, "External data"), or set the `WIKI_BASE` environment variable to the folder that holds them;
- put the `raw/` folder of the snapshot at `data/raw/` in this repository.

Only the small annotation files are kept in this repository (`data/annotation/`). Everything else is computed by the
code here.

## Running the pipeline

Steps marked **online** call Wikipedia or the text services. Run them with `--online` the first time. Every API
response is cached in `data/cache/api/`, so later runs, with or without `--online`, read the cache and give identical
results. The Wikipedia API is called at most once per second.

```bash
# M1 mentor roster
python -m pipeline.m1_roster.collect_roster --online                       # online
python -m pipeline.m1_roster.build_roster

# M2 questions and conversations
python -m pipeline.m2_conversations.collect_topup --online                 # online: pages and DiscussionTools threads
python -m pipeline.m2_conversations.build_questions
python -m pipeline.m2_conversations.collect_topup --online --phase fallback  # online: threads deleted before the cutoff
python -m pipeline.m2_conversations.build_questions
python -m pipeline.m2_conversations.collect_topup --online --phase removals  # online: how removed sections were removed
python -m pipeline.m2_conversations.collect_topup --online --phase recheck
python -m pipeline.m2_conversations.build_questions

# M3 newcomer activity
python -m pipeline.m3_activity.index_s6
python -m pipeline.m3_activity.collect_topup --online                      # online: missing or incomplete histories
python -m pipeline.m3_activity.merge_activity
python -m pipeline.m3_activity.revert_times --online --phase all --window-days 30   # online: when reverted edits were reverted

# M4 to M9
python -m pipeline.m4_eligibility.build_eligibility --online               # online only for user-name lookups
python -m pipeline.m5_sample.build_sample
python -m pipeline.m6_text.build_text --online                             # online: LLM labels, Perspective, embeddings
python -m pipeline.m6_text.agreement                                       # annotation agreement (paper Table 3)
python -m pipeline.m7_dataset.build_dataset
python -m pipeline.m8_analysis.run_analysis
python -m pipeline.m9_paper.compare_new
```

After the online steps have run once, the whole offline chain can be rerun with:

```bash
python run_all.py              # all offline steps, M1 to M9
python run_all.py --from m5    # start from a later step
python run_all.py --list       # show which steps have run
```

Useful options:

- `python -m pipeline.m8_analysis.run_analysis --boot 100` runs a quick version with 100 bootstrap draws instead of
  500 and writes to `results/m8_b100/`, so the final results are not overwritten.
- `--jobs N` (M6 and M8) sets the number of parallel processes. The default is the number of CPU cores minus two.
- `--limit N` runs a step on the first N items only, for debugging.

Rough running times on a laptop: the first online run of M2 takes a few hours, revert times (M3) about ten hours,
and M6 rescoring with Perspective about nine hours (its default quota is one request per second). The offline chain
takes under an hour, most of it in M6 and M8.

Outputs go to `data/interim/` (one folder per step, each with a `manifest.json` and an `_issues/` folder listing
every edge case), `data/analytic/` (the propensity-score dataset and its data dictionary) and `results/`
(tables, logs and the paper comparison).

## Checking the results

```bash
pytest -q tests
python checks/build_m5_check.py --run      # same for m1 ... m8
```

Each `checks/build_mX_check.py` writes `checks/mX_check.ipynb`, and `--run` executes it from top to bottom. Every
notebook follows the same eight sections: what the step does, files and manifest, invariants, comparison with the
previous version, raw data next to processed values, edge cases, distributions, and a summary. The random samples
use `SEED = 0` at the top of each notebook; change it to draw a new sample. Values are recomputed with separate code
in the notebook and printed next to the stored values with a match column.

## Reproducing the annotation

The codebooks, the 241 human-labelled questions (two annotators, Iteration 2), the consensus labels and the LLM labels
are in `data/annotation/`. `python -m pipeline.m6_text.agreement` reproduces the agreement table in the paper. See
[`data/annotation/README.md`](data/annotation/README.md) for the file list and the expected values.

## Data and ethics

All data come from public Wikipedia pages and logs through the MediaWiki API. No private data are used. The released
annotation files do not contain user names. Please follow Wikimedia's API etiquette if you collect data again: keep
the one-request-per-second limit and send your own contact address.

## License

The code is released under the [MIT License](LICENSE). The annotation files in `data/annotation/` are released under
[CC BY-SA 4.0](data/annotation/LICENSE), because they contain Wikipedia text, which is licensed under CC BY-SA 4.0.
