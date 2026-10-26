"""
config.py: the single source of paths, dates, and constants for the whole project

Module: common (CODE_PLAN.md sections 2.3 and 4)
Rules: PLAN.md section 0.1 (fixed cutoff date), section 0.2 (use existing data first)

1. What it does
    Defines all paths, time constants, analysis parameters, and switches in one
    place. No script may hard-code a path, a date, or a threshold. Every script
    reads them from here. This way, a new reader only needs this one file to
    know where each module reads and writes, and which parameters it uses.

2. Why it is designed this way
    The old code spread paths and constants over more than twenty scripts, and
    they did not agree (for example, the cutoff date was called END_CUTOFF in
    s1 and DATA_CUTOFF in s12; the path to the embedding file was hard-coded in
    s12). With one central definition, a change in one place keeps the whole
    pipeline consistent.

    External data (raw data we already have) is not copied into this repo.
    Instead, its original location is registered here, and its sha256 is
    recorded in data/raw/SOURCES.yaml. This makes sure "we use exactly that copy
    of the data".

3. Inputs
    Environment variables (optional):
        WIKI_BASE      overrides the 0Wiki root folder (default: the parent folder of this repo)
        WIKI_ONLINE    set to 1 to allow network access (same as --online in each script)

4. Outputs
    None. It only provides constants.

5. Processing steps
    None.

6. Possible errors and how they are handled
    If an external path does not exist, nothing fails at import time (some
    modules do not need all external data). Instead,
    common.manifest.check_sources() checks the paths each module needs when
    that module starts.

7. How to run
    Not run on its own. Other modules do `import config`.

8. How to check
    tests/test_common.py::test_config_paths_exist checks that all registered
    external data exists.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# 1. Root folders
# ════════════════════════════════════════════════════════════════════════════

#: Root folder of this repo (the folder that holds config.py).
ROOT: Path = Path(__file__).resolve().parent

#: The 0Wiki main folder: all existing raw data lives under it. Can be overridden with the WIKI_BASE environment variable.
WIKI_BASE: Path = Path(os.environ.get("WIKI_BASE", str(ROOT.parent))).resolve()

# Data folders inside this repo
DATA: Path = ROOT / "data"
RAW: Path = DATA / "raw"            # raw data, written only once (newly collected)
INTERIM: Path = DATA / "interim"    # intermediate outputs of each module; <module>/_issues/ holds problem records
ANALYTIC: Path = DATA / "analytic"  # PSM dataset and data dictionary
RESULTS: Path = ROOT / "results"    # tables, figures, logs, reports
CACHE: Path = DATA / "cache"        # cache of API responses and text features
LEDGER: Path = ROOT / "ledger"

# ════════════════════════════════════════════════════════════════════════════
# 2. Fixed cutoff date (PLAN.md section 0.1)
# ════════════════════════════════════════════════════════════════════════════

#: Observation cutoff time (UTC). All data (questions, replies, edits, logs, rosters) is cut at this moment (PLAN section 0.1).
#: Reason: the paper version collected questions and conversations on 2026-04-20 at about 14:00 UTC. The last question is 2026-04-20T13:54:14Z
#: (paper s5; April 20 has only 14 questions, while the days before have 33 to 52 per day). A cutoff at 14:00 includes exactly the paper's last question.
#: Note: when the paper version s12 computed "is the follow-up complete", it used datetime(2026, 4, 24), only in that one place.
#: Here we use one single moment, and follow-up completeness is also judged by this moment (R7.4).
#: Rule: only records "earlier than" the cutoff enter the analysis; the cutoff moment itself is excluded (see common.timeutil).
DATA_CUTOFF: datetime = datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)
DATA_CUTOFF_ISO: str = "2026-04-20T14:00:00Z"

#: When the mentorship system went live on English Wikipedia; the first revision of the roster page is 2021-05-04.
#: It is also the start of the data: the first question with a question tag is 2021-05-05, and the paper sample starts in 2021-05
#: (the paper text says "June 2021", which is not accurate: 2021-05 is the reference group of the year-month dummies and has 29 people in the sample).
#: Decision D10 / R2.4: the main analysis keeps only accounts registered after this time.
MENTORSHIP_START: datetime = datetime(2021, 5, 4, tzinfo=timezone.utc)

#: Decision D10 / R2.2: the main analysis keeps only accounts whose account-creation log has one of these actions (self-registered).
#: Official definition in Manual:Log_actions: create = an anonymous user creates an account for themselves.
ELIGIBLE_CREATION_ACTIONS: tuple[str, ...] = ("create",)
#: Exclusion reason for each of the other actions
CREATION_EXCLUDE_REASON: dict[str, str] = {
    "autocreate": "autocreated_account",
    "create2": "created_by_other",
    "byemail": "created_by_other",
    # an admin forced creation of a local account with CentralAuth (Manual:Log_actions); the user did not register on this site themselves
    "forcecreatelocal": "forced_local_account",
}

#: Date the forceMentor parameter went live (T323038; the patch shipped with the 2023-02-02 release).
FORCEMENTOR_LIVE: datetime = datetime(2023, 2, 2, tzinfo=timezone.utc)

#: Moment the roster switched from wikitext to JSON (SOURCE_SWITCH in eval2 s1).
ROSTER_SOURCE_SWITCH: datetime = datetime(2022, 10, 26, 13, 19, 10, tzinfo=timezone.utc)

# ════════════════════════════════════════════════════════════════════════════
# 3. Input data (data/input/; download it from Hugging Face, see README "Input data")
#    Every input lives under data/input/, grouped by what it is. Keys are the names the modules use.
# ════════════════════════════════════════════════════════════════════════════

INPUT: Path = DATA / "input"
#: Git checkout of the paper-version code. Optional: only used to read the paper-version result tables for comparison.
PAPER_REPO: Path = WIKI_BASE / "1wiki-mentor-github-main" / "wikpedia-mentor-eval"
PAPER_COMMIT: str = "23fea07"          # git commit that matches the paper version's code and results

EXTERNAL: dict[str, Path] = {
    # Mentor roster: every revision of the roster page before the cutoff (wikitext period and JSON period)
    "roster_revisions_wikitext": INPUT / "roster/roster_revisions_wikitext.jsonl",
    "roster_revisions_json":     INPUT / "roster/roster_revisions_json.jsonl",
    "roster_uid_map":            INPUT / "roster/roster_uid_map.json",
    "eval2_mentor_list":         INPUT / "roster/earlier_frozen_mentor_list.jsonl",   # an earlier frozen roster, for comparison
    # Questions: every revision tagged as a mentorship question, and the conversations as first collected
    "eval2_tagged_raw":          INPUT / "questions/tagged_question_revisions.jsonl",
    "eval2_conversations":       INPUT / "questions/tagged_conversations.jsonl",
    "tag_scan_checkpoint":       INPUT / "questions/tag_scan_checkpoint.txt",
    # Newcomer activity as first collected (2026-04-23 to 24, stored by user name). Not released (8.7 GB): the
    # released data contains its extract, data/interim/m3/s6/, so index_s6 can be skipped.
    "s6_profiles":               INPUT / "activity_full/user_profiles.jsonl",
    "s6_contribs":               INPUT / "activity_full/user_contribs.jsonl",
    "s6_logevents":              INPUT / "activity_full/user_logevents.jsonl",
    "s6_abuselog":               INPUT / "activity_full/user_abuselog.jsonl",
    # Qwen text-embedding-v4 embeddings of questions and first replies (keyed by paper-version conversation id)
    "emb_q":                     INPUT / "embeddings/question_embeddings.npz",
    "emb_q_meta":                INPUT / "embeddings/question_embeddings_meta.jsonl",
    "emb_r":                     INPUT / "embeddings/reply_embeddings.npz",
    "emb_r_meta":                INPUT / "embeddings/reply_embeddings_meta.jsonl",
    # Paper-version files: reused where the text is unchanged, and used to replicate and compare with the paper
    "paper_s7":                  INPUT / "paper_version/conversations_cleaned.jsonl",
    "paper_s11":                 INPUT / "paper_version/question_features.jsonl",
    "paper_npz":                 INPUT / "paper_version/psm_dataset.npz",
    "paper_reply_features":      INPUT / "paper_version/reply_features.csv",
    "perspective_q":             INPUT / "paper_version/perspective_scores_misaligned.csv",   # misaligned, not reused (I-59)
    # Question-type annotation (kept in the git repository, see data/annotation/README.md)
    "llm_annotations":           DATA / "annotation/llm_labels_corpus_v2.jsonl",
    "codebook_prompt":           DATA / "annotation/codebook_llm_prompt.md",
    "llm_fewshot":               DATA / "annotation/llm_fewshot_examples.xlsx",
    "irr_gold":                  DATA / "annotation/irr_consensus_labels.csv",
    "irr_a1":                    DATA / "annotation/irr_iteration2_annotator_a1.xlsx",
    "irr_a2":                    DATA / "annotation/irr_iteration2_annotator_a2.csv",
    "irr_sample":                DATA / "annotation/irr_sample_241.csv",
    # Independent collection by an earlier project (June 2026). Optional, not released: the check notebooks use it
    # only to cross-check our edits, registration times and claim log, and skip the cells when it is missing.
    "val_mentee_registration":   INPUT / "validation/mentee_registration.jsonl",
    "val_mentee_contribs":       INPUT / "validation/mentee_contribs.jsonl",
    "setmentor_log_partial":     INPUT / "validation/manual_claims_log.jsonl",
    # Paper source text, used only by tools/build_paper_numbers.py (not released)
    "paper_md":                  WIKI_BASE / "2026-9/2026-9-24/paper.md",
}

#: Raw data already saved under data/raw/ in this repo (newly collected in 2026-10).
RAW_FILES: dict[str, Path] = {
    "claim_log":        RAW / "m4_claims/claim_log_full_20261006.json",
    "registration":     RAW / "m4_registration/mentee_registration_by_uid_20261007.json",
    "siteinfo":         RAW / "wiki_meta/siteinfo_enwiki.json",
    "ge_commit":        RAW / "wiki_meta/growthexperiments_commit.txt",
    "phabricator_dir":  RAW / "wiki_meta/phabricator",
}

# ════════════════════════════════════════════════════════════════════════════
# 4. Wikipedia mechanism constants (DATA_PROCESSING.md)
# ════════════════════════════════════════════════════════════════════════════

#: Question tags (GrowthExperiments includes/HomepageModules/Mentorship.php#L34-L35)
QUESTION_TAGS: tuple[str, ...] = ("mentorship module question", "mentorship panel question")
TAG_HOMEPAGE: str = "mentorship module question"
TAG_HELPPANEL: str = "mentorship panel question"

#: Mentor weights (IMentorWeights.php#L10-L13)
MENTOR_WEIGHTS: dict[str, int] = {"NONE": 0, "LOW": 1, "NORMAL": 2, "HIGH": 4}

#: Original English text of the bulk-reassignment log comments (i18n/mentorship/en.json#L99-L100).
#: $1 and $2 are user-name placeholders; when matching they are replaced by non-greedy wildcards (see m4_eligibility).
REASSIGN_TEMPLATES: tuple[str, ...] = (
    "$1 retired from mentorship.",
    "$2 removed $1 from mentorship",
)

#: Original English text of the "mentor is away" notice (i18n/homepage/en.json#L161-L162). A system-message wraps it as "Note: $1",
#: then MentorQuestionPoster::addAwayDisclaimer wraps it as ":''…''" and puts it on the first line of the body.
#: The real data also has an older form where both links are [[User talk:…]]. common.textclean.strip_away_note handles both forms.
AWAY_NOTE_TEMPLATE: str = (
    "[[User:$1|$1]]'s mentor <span class=\"plainlinks\">[{{fullurl:User:$2}} $2]</span> is away."
)

#: Known alternate accounts of mentors: alt signature name -> main user name (R5.2). Only accounts with evidence are listed.
#: Each entry must state its evidence. When you add one, also update the notes in checks/m2_check.ipynb.
MENTOR_ALT_ACCOUNTS: dict[str, str] = {
    "Yoshi24517 (mobile)": "Yoshi24517",   # evidence: revid 1242646265 and others, replies made as the mentor themselves
    "Pppery (alt)": "Pppery",              # evidence: registered on the roster under the alt name
}

# ════════════════════════════════════════════════════════════════════════════
# 5. Analysis parameters (PLAN.md M5, M8)
# ════════════════════════════════════════════════════════════════════════════

OUTCOME_WINDOWS_DAYS: tuple[int, ...] = (7, 14, 21, 28, 30, 60, 180)
PRIMARY_WINDOW_DAYS: int = 14
ACTIVE_DAYS_WINDOW: int = 30
PSEUDO_LAG_MAX_DAYS: float = 365.0   # pseudo start points are drawn only from real reply delays of 0 to 365 days
N_PSEUDO_DRAWS: int = 20             # number of draws for the "average over many draws" version
HOURS_48: float = 48.0

N_STRATA: int = 10
N_STRATA_SUBGROUP: int = 5
STRATA_SENSITIVITY: tuple[int, ...] = (5, 10, 20)
SUBGROUP_STRATA_SENSITIVITY: tuple[int, ...] = (3, 5, 7, 10)
TRIM_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15)
PS_FOLDS: int = 5
PS_C: float = 1.0                    # inverse of regularization strength for the L2 logistic regression
#: Number of bootstrap draws. Final results use 500. For debugging, set the environment variable WIKI_N_BOOT=100 (or run_analysis --boot 100);
#: output then goes to results/m8_b100/ and does not overwrite the final results.
N_BOOT_FINAL: int = 500
N_BOOT: int = int(os.environ.get("WIKI_N_BOOT", N_BOOT_FINAL))
SMD_THRESHOLD: float = 0.10
VIF_THRESHOLD: float = 10.0
LAG_WINSOR_PCT: float = 99.0

#: Number of parallel processes (M6 text features; M8 subgroups and interaction terms). By default 2 cores are left for the system and background jobs. The WIKI_N_JOBS environment variable can change it.
N_JOBS: int = int(os.environ.get("WIKI_N_JOBS", max(1, (os.cpu_count() or 2) - 2)))

#: Root seed for all random processes; other seeds are derived from it (common.io.derive_seed).
SEED: int = 42

# ════════════════════════════════════════════════════════════════════════════
# 6. Text processing (decisions D5, D6, D9)
# ════════════════════════════════════════════════════════════════════════════

LANGDETECT_SEED: int = 0             # langdetect.DetectorFactory.seed
SHORT_TEXT_MAX_WORDS: int = 2        # fewer than 3 words counts as short text and is not sent to langdetect
SPACY_MODEL: str = "en_core_web_sm"
LLM_MODEL: str = "deepseek-v4-flash"
EMBED_MODEL: str = "text-embedding-v4"
EMBED_DIM: int = 1024

# ════════════════════════════════════════════════════════════════════════════
# 7. Network access and API
# ════════════════════════════════════════════════════════════════════════════

WIKI_API: str = "https://en.wikipedia.org/w/api.php"


def _setting(name: str) -> str:
    """A value from the environment, or else from the .env file in the repo root (empty string if not set)."""
    if os.environ.get(name):
        return os.environ[name].strip()
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
    return ""


#: Your contact e-mail. Wikimedia's User-Agent policy asks every API client to give a way to contact its operator,
#: so set CONTACT_EMAIL in .env (see .env.example) before running anything with --online. Offline runs do not need it.
CONTACT_EMAIL: str = _setting("CONTACT_EMAIL")
USER_AGENT: str = f"MentorResearch/2.0 (academic research on Wikipedia mentorship; {CONTACT_EMAIL or 'contact not set'})"
API_MIN_INTERVAL_SEC: float = 1.0    # minimum gap between two requests
API_MAX_RETRIES: int = 6
API_MAXLAG: int = 5                  # MediaWiki maxlag parameter: back off on our own when the server is busy

#: Whether network access is allowed. The --online flag of each script overrides it.
ONLINE: bool = os.environ.get("WIKI_ONLINE", "0") == "1"

# ════════════════════════════════════════════════════════════════════════════
# 8. Fix switches (ledger/CHANGES.md records, item by item, what changes when each is turned on)
#    When LEGACY_MODE=True, M8 reads the paper-version npz and none of the fixes apply.
# ════════════════════════════════════════════════════════════════════════════

LEGACY_MODE: bool = False


def interim(module: str) -> Path:
    """Return the intermediate-output folder of a module (create it if missing).

    Args:
        module: module name, for example "m2".

    Returns:
        The path data/interim/<module>/.
    """
    p = INTERIM / module
    p.mkdir(parents=True, exist_ok=True)
    return p


def issues_dir(module: str) -> Path:
    """Return the problem-record folder of a module, data/interim/<module>/_issues/ (create it if missing)."""
    p = interim(module) / "_issues"
    p.mkdir(parents=True, exist_ok=True)
    return p
