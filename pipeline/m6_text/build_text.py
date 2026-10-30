"""
build_text.py: M6 main program: text features and labels for each sampled first question, and text features and
embeddings for the first reply in the treatment group

Module: M6 (PLAN.md M6; CODE_PLAN.md section 5, M6)
Rules: DATA_PROCESSING R4.4 to R4.6. Decisions: D6, D9. Issues: I-07, I-27, I-37

1. What it does
    1. Texts:
        - Input for LLM labels = the cleaned plain text of the question (M2 question_text; matches question_clean in
          the paper version).
        - Input for Perspective and embeddings = the same text with user names from the conversation removed and
          links replaced by [LINK] (make_emb_text; matches question_emb in the paper version).
        - Reply = the run of comments by the first replier, starting from their first reply (M2 turns).
    2. Reuse paper-version results (import_legacy): find the same question in the paper version by revid. If the
       new text and the paper-version text are identical after normalization, or their word Jaccard similarity is
       ≥ LEGACY_JACCARD, the paper version's LLM labels, Perspective scores, and embeddings are used as the result
       for this new text (source="legacy"). Otherwise (usually because the paper version mixed the reply into the
       question, or the cleaning rules differ), the services are called again.
    3. Texts still missing from the cache: with --online, call services (results go into the cache with
       source="api"). Without network access, they are recorded as missing.
    4. Local features (text_features): 37 question columns and 21 reply columns. Link redirects and policy
       categories come from the API (cached).
    5. Cosine similarity between the question and reply embeddings (I-27).

2. Why it is designed this way
    The paper version's labels, scores, and embeddings were computed on the paper-version texts. They can only be
    reused when the text is essentially the same; if the text changed, we compute again. This way we do not pay
    twice, and we do not carry over results from wrong texts (for example, a question with a reply mixed in).

3. Inputs
    data/interim/m5/sample.parquet; data/interim/m2/questions.parquet, turns.jsonl; paper-version s7, s10, and
    embedding files.

4. Outputs (data/interim/m6/)
    question_features.parquet  one row per sampled first question: 37 text features, 5 question types,
                               8 Perspective scores, and the source of each
    reply_features.parquet     one row per first reply in the treatment group: 21 reply features
    emb_q.npy + emb_q_index.parquet, emb_r.npy + emb_r_index.parquet  embeddings (float32; rows line up with the
                               index table)
    coverage.json              for each item, the counts reused, newly fetched, and missing

7. How to run
    python -m pipeline.m6_text.build_text --online      # call the three services to fill gaps (about 1 to 2 hours)
    python -m pipeline.m6_text.build_text               # use the cache only
    python -m pipeline.m6_text.build_text --jobs 4      # number of parallel processes for local text features (default config.N_JOBS)
    Local text features (the 37 question columns and the text part of the reply features) are also cached by the
    original text (tasks qfeat and rfeat, version text_features.FEATURE_VERSION). If only the treatment variables
    changed and the texts did not, a rerun takes almost no time. If you change a feature's method, add one to
    FEATURE_VERSION.

8. How to check
    checks/m6_check.ipynb; tests/test_m6.py.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import config
from pipeline.common.api import MediaWikiClient
from pipeline.common.io import iter_jsonl, write_json_atomic, write_table
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.common.textclean import make_emb_text
from pipeline.common.wikimeta import policy_status, resolve_titles
from pipeline.m6_text import services as S
from pipeline.m6_text.cache import TextCache, norm_text, text_sha
from pipeline.m6_text import text_features as TF
from pipeline.m6_text.text_features import QTEXT_COLS, reply_features

OUT = config.INTERIM / "m6"
LEGACY_JACCARD = 0.85
QTYPE_COLS = {"Q0": "q_substantive", "Q2": "q_referent", "Q3": "q_appraisal", "Q4": "q_normative", "Q5": "q_own_work"}
PERSP_COLS = {a: f"q_persp_{a.lower()}" for a in S.PERSP_ATTRS}


def similar(a: str, b: str) -> tuple[bool, float]:
    """True if the texts are identical after normalization (whitespace, case), or the Jaccard similarity of their word sets is ≥ LEGACY_JACCARD."""
    x, y = norm_text(a).lower(), norm_text(b).lower()
    if x == y:
        return True, 1.0
    sx, sy = set(x.split()), set(y.split())
    j = len(sx & sy) / max(1, len(sx | sy))
    return j >= LEGACY_JACCARD, round(j, 4)


def emb_text_from_plain(plain: str, names: list[str]) -> str:
    """For text that is already plain (replies): remove user names and replace links with [LINK], using the same rules as make_emb_text."""
    t = re.sub(r"https?://\S+", "[LINK]", plain or "")
    for n in sorted({n for n in names if n and len(n) >= 3}, key=len, reverse=True):
        t = re.sub(r"(?<!\w)" + re.escape(n).replace(r"\ ", r"[ _]") + r"(?!\w)", " ", t, flags=re.I)
    return re.sub(r"[ \t]+", " ", t).strip()


def load_inputs(log) -> tuple[pd.DataFrame, dict]:
    """Load the sample, the wikitext of each first question, and the thread comments."""
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet")
    s = s[s.in_sample | s.in_sample_unrestricted].copy()
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet", columns=["question_revid", "question_wikitext"])
    s = s.merge(q, left_on="first_question_revid", right_on="question_revid", how="left", suffixes=("", "_m2"))
    need = set(s.first_question_revid)
    turns = {r["question_revid"]: r["turns"] for r in iter_jsonl(config.INTERIM / "m2" / "turns.jsonl")
             if r["question_revid"] in need}
    log.info("sample %d; with turns %d", len(s), len(turns))
    return s, turns


def first_reply(turns: list[dict], q_ts) -> tuple[str, dict, list[str]]:
    """The run of comments by the first replier, starting from their first reply: (text, merged links, user names seen in the conversation)."""
    q_min = pd.Timestamp(q_ts).floor("min")
    names = [t["author"] for t in turns if t.get("author")]
    idx = next((i for i, t in enumerate(turns) if t["role"] != "mentee" and t.get("ts")
                and pd.Timestamp(t["ts"]) >= q_min), None)
    if idx is None:
        return "", {"wikilinks": [], "n_external": 0, "n_list_items": 0}, names
    who = turns[idx]["author"]
    parts, links = [], {"wikilinks": [], "n_external": 0, "n_list_items": 0}
    for t in turns[idx:]:
        if t["author"] != who:
            break
        parts.append(t["text"])
        lk = t.get("links") or {}
        links["wikilinks"] += lk.get("wikilinks", [])
        links["n_external"] += lk.get("n_external", 0)
        links["n_list_items"] += lk.get("n_list_items", 0)
    return "\n\n".join(parts), links, names


def text_table(s: pd.DataFrame, turns: dict) -> pd.DataFrame:
    """One row per first question: the plain question text, the text sent to Perspective and the embedding model, and the first reply."""
    rows = []
    for r in s.itertuples():
        ts = turns.get(r.first_question_revid, [])
        names = [t["author"] for t in ts if t.get("author")] + [r.mentee_name]
        rtext, rlinks, _ = first_reply(ts, r.q_ts) if r.treat else ("", {}, names)
        rows.append({"revid": int(r.first_question_revid), "q_plain": r.question_text or "",
                     "q_emb": make_emb_text(r.question_wikitext or "", names), "r_plain": rtext,
                     "r_emb": emb_text_from_plain(rtext, names), "r_links": rlinks})
    return pd.DataFrame(rows).set_index("revid")


def legacy() -> dict:
    """Paper-version texts and results, keyed by revid."""
    s7 = {int(r["revid"]): r for r in iter_jsonl(config.EXTERNAL["paper_s7"]) if r.get("revid")}
    cid2rev = {int(r["conversation_id"]): rv for rv, r in s7.items()}
    llm = {cid2rev.get(int(r["cid"])): {k: r[k] for k in S.DIMS} for r in iter_jsonl(config.EXTERNAL["llm_annotations"])
           if "Q0" in r}
    p = pd.read_csv(config.EXTERNAL["perspective_q"])
    persp = {cid2rev.get(int(r["conversation_id"])): {a: (None if pd.isna(r[a]) else float(r[a])) for a in S.PERSP_ATTRS}
             for _, r in p.iterrows()}
    eq = np.load(config.EXTERNAL["emb_q"])["embeddings"]
    er = np.load(config.EXTERNAL["emb_r"])["embeddings"]
    emb_q = {cid2rev.get(int(m["conversation_id"])): int(m["idx"]) for m in iter_jsonl(config.EXTERNAL["emb_q_meta"])}
    emb_r = {cid2rev.get(int(m["conversation_id"])): int(m["idx"]) for m in iter_jsonl(config.EXTERNAL["emb_r_meta"])}
    return {"s7": s7, "llm": llm, "persp": persp, "eq": eq, "er": er, "emb_q": emb_q, "emb_r": emb_r}


def fill(cache: TextCache, items: list[tuple[int, str, str]], legacy_vals: dict, task: str, model: str, version: str,
         call, online: bool, log, chunk: int = 200) -> dict[str, object]:
    """items: (revid, new text, paper-version text). First import paper-version results that can be reused, then fill gaps online. Returns {new text: result}."""
    reuse = {}
    for rv, new, old in items:
        if rv in legacy_vals and legacy_vals[rv] is not None and old is not None and similar(new, old)[0]:
            reuse[new] = legacy_vals[rv]
    have = cache.get_many([t for _, t, _ in items], task, model, version)
    to_put = {t: v for t, v in reuse.items() if t not in have}
    if to_put:
        cache.put_many(to_put, task, model, version, source="legacy")
    missing = cache.missing([t for _, t, _ in items if t.strip()], task, model, version)
    log.info("%s: %d texts, legacy reuse %d, already cached %d, missing %d", task, len(items), len(reuse), len(have),
             len(missing))
    if online and missing:
        n_got = 0
        for i in range(0, len(missing), chunk):          # save after each chunk, so a stop loses at most one chunk
            part = cache.missing(missing[i:i + chunk], task, model, version)   # skip texts filled meanwhile (e.g. imported from a second machine)
            if not part:
                continue
            got = call(part, log)
            cache.put_many(got, task, model, version, source="api")
            n_got += len(got)
            log.info("%s: %d/%d done", task, min(i + chunk, len(missing)), len(missing))
        log.info("%s: called service for %d, got %d", task, len(missing), n_got)
    return cache.get_many([t for _, t, _ in items], task, model, version)


def embeddings(items: list[tuple[int, str, str]], legacy_mat, legacy_idx: dict, name: str, online: bool, log):
    """Embeddings: take the paper-version vector when it can be reused, and fetch the rest online. Results are saved to emb_{name}.npy and an index table. Returns {revid: vector}."""
    path_npy, path_idx = OUT / f"emb_{name}.npy", OUT / f"emb_{name}_index.parquet"
    store: dict[str, np.ndarray] = {}
    if path_npy.exists():
        old = pd.read_parquet(path_idx)
        mat = np.load(path_npy)
        store = {sha: mat[i] for i, sha in enumerate(old.text_sha)}
    src = {}
    for rv, new, old in items:
        sha = text_sha(new)
        if sha in store:
            continue
        if rv in legacy_idx and old is not None and similar(new, old)[0]:
            store[sha] = legacy_mat[legacy_idx[rv]].astype(np.float32)
            src[sha] = "legacy"
    miss = [new for _, new, _ in items if new.strip() and text_sha(new) not in store]
    log.info("emb_%s: %d texts, have %d, missing %d", name, len(items), len(items) - len(miss), len(miss))
    if online and miss:
        got = S.embed(list(dict.fromkeys(miss)), log)
        for t, v in got.items():
            store[text_sha(t)] = np.asarray(v, dtype=np.float32)
            src[text_sha(t)] = "api"
    shas = list(store)
    np.save(path_npy, np.stack([store[k] for k in shas]) if shas else np.zeros((0, S.EMB_DIM), np.float32))
    prev = pd.read_parquet(path_idx).set_index("text_sha").source.to_dict() if path_idx.exists() else {}
    write_table(path_idx, pd.DataFrame({"text_sha": shas, "source": [src.get(k, prev.get(k, "legacy")) for k in shas]}))
    return {rv: store.get(text_sha(new)) for rv, new, _ in items}


def exact_key(*parts: str) -> str:
    """A key without whitespace normalization (text features depend on length and indentation, so whitespace cannot be merged as in the label cache)."""
    return "x:" + hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def cached_features(cache: TextCache, task: str, items: dict[str, object], worker, n_jobs: int, log,
                    chunk: int = 250) -> dict[str, dict]:
    """Local text features: use the cache, and compute the rest with several processes.

    Args:
        items: {exact_key: one input for worker}.
        worker: a *_batch function from text_features (takes a list, returns a list in the same order).
    How it works:
        Items already in the cache (task + FEATURE_VERSION) are taken from there. The missing ones are split into
        blocks of `chunk` items and computed in parallel by n_jobs processes. Each finished block is written to
        the cache right away, so if the run is interrupted, the work already done is not lost.
    """
    keys = list(items)
    have = cache.get_many(keys, task, "local", TF.FEATURE_VERSION)
    todo = [k for k in keys if k not in have]
    log.info("%s: %d texts, cached %d, computing %d with %d processes", task, len(keys), len(have), len(todo), n_jobs)
    if todo:
        blocks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        t0, done = time.time(), 0
        if n_jobs <= 1:
            results = (worker([items[k] for k in b]) for b in blocks)
        else:
            ex = ProcessPoolExecutor(n_jobs)
            results = ex.map(worker, [[items[k] for k in b] for b in blocks])
        for b, res in zip(blocks, results):
            got = dict(zip(b, res))
            cache.put_many(got, task, "local", TF.FEATURE_VERSION, source="local")
            have.update(got)
            done += len(b)
            if done % (chunk * 20) < chunk or done == len(todo):
                log.info("  %s %d/%d (%.0f s)", task, done, len(todo), time.time() - t0)
        if n_jobs > 1:
            ex.shutdown()
    return have


def main(argv: list[str] | None = None) -> None:
    ap = base_parser("M6: text features, LLM labels, Perspective, embeddings.")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS, help="number of parallel processes for local text features")
    args = ap.parse_args(argv)
    log = setup_logging("m6")
    t0 = time.time()
    check_upstream("m6", ["m5"])
    OUT.mkdir(parents=True, exist_ok=True)
    s, turns = load_inputs(log)
    if args.limit:
        s = s.head(args.limit)
        log.warning("--limit %d: dry run on a subset", args.limit)
    L = legacy()
    cache = TextCache()

    # 1. Texts
    T = text_table(s, turns)

    # 2. LLM, Perspective, embeddings (reuse, then fill gaps)
    s7 = L["s7"]
    old = lambda rv, f: (s7.get(rv) or {}).get(f)     # noqa: E731
    llm = fill(cache, [(rv, x.q_plain, old(rv, "question_clean")) for rv, x in T.iterrows()], L["llm"],
               "llm_qtype", S.LLM_MODEL, S.LLM_VERSION, S.annotate_llm, args.online, log)
    # Perspective scores are never copied from the paper version (I-59): the paper file is keyed by a different set of
    # conversation ids than s7, so most scores belong to another text. Every question is scored again.
    persp = fill(cache, [(rv, x.q_emb, old(rv, "question_emb")) for rv, x in T.iterrows()], {},
                 "perspective_q", S.PERSP_MODEL, S.PERSP_VERSION, S.perspective, args.online, log)
    eq = embeddings([(rv, x.q_emb, old(rv, "question_emb")) for rv, x in T.iterrows()], L["eq"], L["emb_q"], "q",
                    args.online, log)
    tr = T[T.r_plain.str.len() > 0]
    er = embeddings([(rv, x.r_emb, old(rv, "reply_emb")) for rv, x in tr.iterrows()], L["er"], L["emb_r"], "r",
                    args.online, log)

    # 3. Question features (computed locally, cached by text, several processes)
    wt = s.set_index("first_question_revid").question_wikitext.fillna("").to_dict()
    qkey = {rv: exact_key(wt.get(rv, ""), x.q_plain) for rv, x in T.iterrows()}
    qfeat = cached_features(cache, "qfeat", {k: (wt.get(rv, ""), T.at[rv, "q_plain"]) for rv, k in qkey.items()},
                            TF.question_features_batch, args.jobs, log)
    qrows = []
    for rv, x in T.iterrows():
        f = dict(qfeat[qkey[rv]])
        lab = llm.get(x.q_plain)
        f.update({c: (int(lab[k] == "Y") if lab else np.nan) for k, c in QTYPE_COLS.items()})
        sc = persp.get(x.q_emb)
        if sc is not None and all(v is None for v in sc.values()):
            sc = None                                    # Perspective answered but gave no scores (e.g. one-letter text): treat as missing
        f.update({c: (sc.get(a) if sc else np.nan) for a, c in PERSP_COLS.items()})
        f.update({"first_question_revid": rv, "has_llm": lab is not None, "has_persp": sc is not None,
                  "has_emb_q": eq.get(rv) is not None})
        qrows.append(f)
    QF = pd.DataFrame(qrows)
    write_table(OUT / "question_features.parquet", QF)
    np.save(OUT / "emb_q_sample.npy", np.stack([eq[rv] if eq.get(rv) is not None else np.full(S.EMB_DIM, np.nan, np.float32)
                                                 for rv in QF.first_question_revid]))
    log.info("question features: %d rows", len(QF))

    # 4. Reply features (link redirects and policy categories: from the API, cached)
    client = MediaWikiClient(online=args.online)
    titles = {l["title"] for x in tr.r_links for l in x.get("wikilinks", [])}
    resolved = resolve_titles(titles, client)
    finals = {v["final"] for v in resolved.values() if v.get("ns") == 4}
    pol = {t for t, v in policy_status(finals, client).items() if v}
    rkey = {rv: exact_key(x.r_plain) for rv, x in tr.iterrows()}
    rfeat = cached_features(cache, "rfeat", {k: tr.at[rv, "r_plain"] for rv, k in rkey.items()},
                            TF.reply_text_features_batch, args.jobs, log)
    lag = s.set_index("first_question_revid").reply_lag_days.to_dict()
    rrows = []
    for rv, x in tr.iterrows():
        q_words = len(x.q_plain.split())
        f = reply_features(x.r_plain, x.r_links, q_words, float(lag[rv]) * 24, pol, resolved, base=rfeat[rkey[rv]])
        a, b = eq.get(rv), er.get(rv)
        if a is not None and b is not None:
            f["r_qr_cosine_sim"] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        f["first_question_revid"] = rv
        rrows.append(f)
    RF = pd.DataFrame(rrows)
    write_table(OUT / "reply_features.parquet", RF)
    cov = {"questions": len(QF), "llm": int(QF.has_llm.sum()), "persp": int(QF.has_persp.sum()), "emb_q": int(QF.has_emb_q.sum()),
           "replies": len(RF), "emb_r": int(sum(er.get(rv) is not None for rv in tr.index)),
           "qr_cosine": int(RF.r_qr_cosine_sim.notna().sum()), "cache": cache.stats}
    write_json_atomic(OUT / "coverage.json", cov)
    log.info("done: %s", cov)
    write_manifest("m6", inputs=[config.INTERIM / "m5" / "sample.parquet"],
                   outputs=[OUT / "question_features.parquet", OUT / "reply_features.parquet"],
                   params={"legacy_jaccard": LEGACY_JACCARD, "online": args.online}, counts=cov, started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
