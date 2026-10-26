"""
services.py: three outside text services: LLM question types (DeepSeek), Perspective scores, Qwen embeddings

Module: M6 (PLAN.md M6 items 1, 3, 4; CODE_PLAN.md section 5, M6)
Rules: Issues I-07 (embedding model name), I-31; PLAN M6 "missing is not filled with 0"

1. What it does
    annotate_llm(texts): DeepSeek deepseek-v4-flash, non-thinking mode, temperature 0. The system prompt is the paper
        version's codebook_prompt.md plus the first 120 human labels from data/annotation/llm_fewshot_examples.xlsx as examples (exactly
        the same as the paper version s10_2). It outputs five Y/N labels: Q0, Q2, Q3, Q4, Q5. If the reply cannot be
        parsed, we retry up to 3 times. If it still fails, the text is recorded as missing (not filled with N).
    perspective(texts): Perspective API v1alpha1, languages=["en"], doNotStore=True. The paper version requested 23
        attributes. The model uses 8 of them (TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, PROFANITY, THREAT,
        SEXUALLY_EXPLICIT, FLIRTATION). If the service fails, the text is recorded as missing, not filled with 0.
    embed(texts): the OpenAI-compatible endpoint of Alibaba Cloud DashScope, model text-embedding-v4, 1024 dimensions
        (the paper text must be changed to use this model name, I-07).

2. Why it is designed this way
    All three services only process texts that are missing from the cache (cache.TextCache; embeddings are stored
    in an npy file, see build_text). They use the same model, prompt, and settings as the paper version. This way,
    reused paper-version results and newly fetched results can be mixed.
    API keys are read from the .env file in the repository root (they are never written into the code).

6. Possible errors and how they are handled
    - No key, or --online is not turned on: no call is made and an empty result is returned. The caller records
      these texts as missing and lists them.
    - Rate limit (429): retry with exponential backoff. Other errors: retry 3 times, then record as missing.

8. How to check
    checks/m6_check.ipynb: counts of cache hits, misses, and new calls; re-call 3 random texts and compare them
    with the cache (embeddings).
"""
from __future__ import annotations

import json
import os
import re
import time

import logging

import config

logging.getLogger("httpx").setLevel(logging.WARNING)

DIMS = ["Q0", "Q2", "Q3", "Q4", "Q5"]
LLM_MODEL, LLM_VERSION = "deepseek-v4-flash", "codebook_v2_fewshot120"
PERSP_MODEL, PERSP_VERSION = "perspective", "v1alpha1"
EMB_MODEL, EMB_DIM = "text-embedding-v4", 1024
PERSP_ATTRS = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT",
               "SEXUALLY_EXPLICIT", "FLIRTATION"]
MAX_RETRIES = 3


def env(name: str) -> str | None:
    """Read a value from the environment or from the .env file in the repository root (never printed)."""
    if os.environ.get(name):
        return os.environ[name]
    p = config.ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ── LLM ─────────────────────────────────────────────────────────────────────
def codebook_text() -> str:
    """The paper version's system prompt: codebook plus 120 human-labelled examples (USE_FEWSHOT and FEWSHOT_N in s10_2)."""
    import openpyxl
    text = config.EXTERNAL["codebook_prompt"].read_text("utf-8")
    wb = openpyxl.load_workbook(config.EXTERNAL["llm_fewshot"], read_only=True)   # not irr_a1: the prompt used an earlier round
    headers, examples = None, []
    for row in wb.active.iter_rows(values_only=True):
        if headers is None:
            headers = [str(h) for h in row]
            continue
        e = dict(zip(headers, row))
        if int(str(e["idx"])) > 120:
            continue
        labels = {d: str(e.get(d, "")).strip().upper() for d in DIMS}
        if all(v in ("Y", "N") for v in labels.values()):
            examples.append(f"Message: {e['question']}\nAnswer: {json.dumps(labels)}")
    wb.close()
    return text + "\n\n---\n\n## Examples\n\n" + "\n\n".join(examples)


def parse_labels(raw: str) -> dict | None:
    """Turn the model reply into {Q0..Q5: "Y"/"N"}. Return None if it cannot be parsed (same as s10_2 parse_json_result)."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw or "").strip()
    m = re.search(r"\{[^}]*\}", cleaned)
    if m:
        try:
            d = json.loads(m.group())
            r = {k: d.get(k, "?") for k in DIMS}
            if all(v in ("Y", "N") for v in r.values()):
                return r
        except json.JSONDecodeError:
            pass
    r = {}
    for k in DIMS:
        m2 = re.search(rf'"{k}"\s*:\s*"([YN])"', cleaned)
        if m2:
            r[k] = m2.group(1)
    return r if len(r) == len(DIMS) else None


def annotate_llm(texts: list[str], log, max_workers: int = 8) -> dict[str, dict]:
    """Call DeepSeek on texts and return {text: labels}. Texts that failed are left out of the result."""
    key = env("DEEPSEEK_API_KEY")
    if not key or not texts:
        return {}
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=key, timeout=120)
    system = codebook_text()

    errors: dict[str, int] = {}                          # error type -> count, written to the log at the end
    fatal: list[str] = []                                # out of balance or login failed: retrying will not help, so stop

    def one(t):
        if fatal:
            return t, None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL, temperature=0, max_completion_tokens=200,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": f"Annotate the following message for dimensions Q0, Q2, Q3, Q4, Q5.\n\nMessage:\n{t}"}])
                r = parse_labels(resp.choices[0].message.content or "")
                if r:
                    return t, r
                errors["unparseable_reply"] = errors.get("unparseable_reply", 0) + 1
            except Exception as e:                       # noqa: BLE001  retry on service errors; in the end record as missing
                msg = str(e)
                kind = msg[:80]
                errors[kind] = errors.get(kind, 0) + 1
                if errors[kind] == 1:
                    log.warning("  llm error (first of its kind): %s", msg[:300])
                if any(code in msg for code in ("402", "401", "Insufficient Balance")):
                    fatal.append(msg[:300])
                    return t, None
                if "429" in msg:
                    time.sleep(2 ** attempt)
        return t, None

    out = {}
    with ThreadPoolExecutor(max_workers) as ex:
        for i, (t, r) in enumerate(ex.map(one, texts), 1):
            if r:
                out[t] = r
            if i % 500 == 0:
                log.info("  llm %d/%d (ok %d)", i, len(texts), len(out))
    if errors:
        log.warning("  llm errors: %s", errors)
    if fatal:                                            # labels we already got are returned and cached as usual; the missing ones show up in coverage.json
        log.error("  DeepSeek stopped early, %d of %d texts unlabelled: %s", len(texts) - len(out), len(texts), fatal[0])
    return out


# ── Perspective ─────────────────────────────────────────────────────────────
def perspective(texts: list[str], log) -> dict[str, dict]:
    """Call Perspective on texts and return {text: {attribute: score}}. Texts that failed are left out of the result."""
    import requests
    key = env("PERSPECTIVE_API_KEY")
    if not key or not texts:
        return {}
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={key}"
    out = {}
    for i, t in enumerate(texts, 1):
        payload = {"comment": {"text": t[:20000]}, "languages": ["en"], "doNotStore": True,
                   "requestedAttributes": {a: {} for a in PERSP_ATTRS}}
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(url, json=payload, timeout=20)
                if r.status_code == 429:
                    time.sleep(2 ** attempt + 1)
                    continue
                if r.ok:
                    sc = r.json().get("attributeScores", {})
                    out[t] = {a: sc.get(a, {}).get("summaryScore", {}).get("value") for a in PERSP_ATTRS}
                break
            except requests.RequestException:
                time.sleep(1)
        time.sleep(1.05)                                 # the default quota is 1 query per second
        if i % 200 == 0:
            log.info("  perspective %d/%d (ok %d)", i, len(texts), len(out))
    return out


# ── Embeddings ────────────────────────────────────────────────────────────────────
def embed(texts: list[str], log, batch: int = 10) -> dict[str, list[float]]:
    """Qwen text-embedding-v4 (1024 dimensions). Return {text: vector}. Batches that failed are left out of the result."""
    key = env("DASHSCOPE_API_KEY")
    if not key or not texts:
        return {}
    from openai import OpenAI
    client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=key, timeout=120)
    out = {}
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] for t in texts[i:i + batch]]
        for attempt in range(MAX_RETRIES):
            try:
                r = client.embeddings.create(model=EMB_MODEL, input=chunk, dimensions=EMB_DIM, encoding_format="float")
                for t, d in zip(texts[i:i + batch], r.data):
                    out[t] = d.embedding
                break
            except Exception:                            # noqa: BLE001
                time.sleep(2 ** attempt)
        if (i // batch) % 100 == 0:
            log.info("  embed %d/%d", min(i + batch, len(texts)), len(texts))
    return out
