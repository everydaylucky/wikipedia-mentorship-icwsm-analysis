"""
api.py: one shared MediaWiki API client

Module: common layer (CODE_PLAN.md sections 2.3 and 4)

1. What it does
    All online data collection (M2 extra scans, DiscussionTools parsing, move
    logs; M3 extra fetches; M4 claim logs; namespace and redirect lookups) sends
    its requests through MediaWikiClient. The client takes care of:
        1. A fixed User-Agent (the Wikimedia User-Agent policy requires contact details);
        2. Rate limiting: at least config.API_MIN_INTERVAL_SEC seconds between two requests;
        3. Retries: on HTTP 429, 5xx, or a maxlag error, wait longer and longer
           (exponential backoff), at most config.API_MAX_RETRIES times;
        4. Disk cache: each response is saved in data/cache/api/, keyed by "a hash
           of the request parameters". The same request is only ever sent once;
           a rerun reads straight from the cache;
        5. Offline mode: without --online, a request that is not in the cache
           fails right away. This makes sure an "offline run" really does not
           go online, so results are fully reproducible;
        6. Paging: query_all() handles the API's continue parameter for you.

2. Why it is designed this way
    In the old code, each collection script wrote its own requests, retries,
    and resume-from-checkpoint logic. They all behaved differently, and some even
    turned off SSL certificate checks. With one shared client, all collection
    is equally polite to the server, handles errors the same way, and is
    equally reproducible.

    The cache key is the full set of parameters, so a cache hit happens only
    when the parameters are exactly the same (including cutoff parameters such
    as rvstart). If the cutoff date changes, the request is sent again, so old
    results are never reused by mistake.

3. Inputs
    A dict of request parameters (same as in the MediaWiki API docs).

4. Outputs
    Parsed JSON (a dict). Cache file: data/cache/api/<first two chars>/<sha256>.json.
    It holds the request parameters, the fetch time, and the response, and can
    be opened and read on its own.

6. Possible errors and how they are handled
    - Offline and not in the cache: raise OfflineCacheMiss. The message shows the
      parameters, which helps list "what still needs to be fetched";
    - The API returns an error field: raise ApiError (except maxlag, which is retried);
    - All retries used up: raise the exception from the last try.

8. How to check
    tests/test_common.py::test_api_cache_offline tests the cache and offline
    behavior in a temporary folder, without going online.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import config
from pipeline.common.io import write_json_atomic

log = logging.getLogger(__name__)


class OfflineCacheMiss(RuntimeError):
    """In offline mode, a request asked for something that is not in the cache."""


class ApiError(RuntimeError):
    """The MediaWiki API returned an error. The attribute code is the API error code."""

    def __init__(self, code: str, info: str, params: dict):
        super().__init__(f"API error {code}: {info} (params={params})")
        self.code = code


class MediaWikiClient:
    """MediaWiki API client with rate limiting, retries, a cache, and an offline mode.

    Args:
        online: whether network access is allowed. Default is config.ONLINE (the --online flag of each script overrides it).
        cache_dir: cache folder, default data/cache/api.
        api_url: API address, default English Wikipedia.

    Usage:
        client = MediaWikiClient(online=args.online)
        data = client.get({"action": "query", "list": "users", "ususers": "Mujinga"})
        for page in client.query_all({...}, result_path=("query", "logevents")):
            ...
    """

    def __init__(self, online: bool | None = None, cache_dir: Path | None = None,
                 api_url: str = config.WIKI_API):
        self.online = config.ONLINE if online is None else online
        self.cache_dir = cache_dir or (config.CACHE / "api")
        self.api_url = api_url
        self._last = 0.0
        self.stats = {"cache_hit": 0, "network": 0, "retries": 0}

    # ── Cache ──────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(params: dict[str, Any]) -> dict[str, str]:
        """Put the parameters in a standard form: turn all values into strings, add format=json, and sort by key."""
        p = {k: ("|".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))
             for k, v in params.items() if v is not None}
        p.setdefault("format", "json")
        p.setdefault("formatversion", "2")
        return dict(sorted(p.items()))

    def _cache_path(self, params: dict[str, str]) -> Path:
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    # ── Requests ──────────────────────────────────────────────────────────────
    def _wait(self) -> None:
        """Rate limit: make sure at least API_MIN_INTERVAL_SEC seconds pass between two network requests."""
        dt = time.time() - self._last
        if dt < config.API_MIN_INTERVAL_SEC:
            time.sleep(config.API_MIN_INTERVAL_SEC - dt)
        self._last = time.time()

    def _fetch(self, params: dict[str, str], max_retries: int | None = None) -> dict:
        """Actually send the request and handle retries. Only called when online=True.

        max_retries: the most tries allowed, default config.API_MAX_RETRIES. For requests that
        are sure to time out (for example, DiscussionTools parsing of a very large page), set it
        low, so that each failure does not cost several minutes of waiting.
        """
        if not config.CONTACT_EMAIL:
            raise RuntimeError("Set CONTACT_EMAIL=<your e-mail> in .env (see .env.example) before going online: "
                               "Wikimedia's User-Agent policy requires a contact address.")
        tries = max_retries or config.API_MAX_RETRIES
        p = dict(params)
        p.setdefault("maxlag", str(config.API_MAXLAG))
        url = f"{self.api_url}?{urllib.parse.urlencode(p)}"
        req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
        last_exc: Exception | None = None
        for attempt in range(tries):
            self._wait()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read())
                err = data.get("error")
                if err:
                    if err.get("code") == "maxlag":       # server is busy: back off, then retry
                        raise _Retryable(f"maxlag: {err.get('info')}")
                    raise ApiError(err.get("code", "?"), err.get("info", ""), params)
                self.stats["network"] += 1
                return data
            except (_Retryable, urllib.error.URLError, TimeoutError, ConnectionError) as e:
                code = getattr(e, "code", None)
                if isinstance(e, urllib.error.HTTPError) and code not in (429, 500, 502, 503, 504):
                    raise
                last_exc = e
                self.stats["retries"] += 1
                wait = min(2 ** attempt * 2, 120)
                log.warning("API retry %d/%d after %ss: %s", attempt + 1, tries, wait, e)
                if attempt + 1 < tries:
                    time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def get(self, params: dict[str, Any], *, refresh: bool = False, max_retries: int | None = None) -> dict:
        """Send one API request (or read it from the cache).

        Args:
            params: API parameters, for example {"action": "query", "list": "users", "ususers": "X"}.
            refresh: if True, ignore the cache and request again (needs online).
            max_retries: the most tries allowed (see _fetch).

        Returns:
            The response JSON.

        Raises:
            OfflineCacheMiss: offline and not in the cache.
            ApiError: the API returned an error.
        """
        p = self._normalize(params)
        path = self._cache_path(p)
        if path.exists() and not refresh:
            self.stats["cache_hit"] += 1
            return json.loads(path.read_text())["response"]
        if not self.online:
            raise OfflineCacheMiss(f"not in cache and offline: {p}")
        data = self._fetch(p, max_retries)
        write_json_atomic(path, {"params": p, "fetched_at": datetime.now(timezone.utc).isoformat(),
                                 "response": data}, indent=0)
        return data

    def query_all(self, params: dict[str, Any], result_path: tuple[str, ...]) -> Iterator[Any]:
        """Handle paging (continue) for you, and yield the items of the result list one by one.

        Args:
            params: parameters for the first page.
            result_path: where the results sit in the response, for example ("query", "logevents").
                         For prop=revisions, use ("query", "pages") and let the caller unpack them.

        Yields:
            Each item in the result list.
        """
        p = dict(params)
        while True:
            data = self.get(p)
            node: Any = data
            for k in result_path:
                node = node.get(k, []) if isinstance(node, dict) else []
            yield from (node or [])
            cont = data.get("continue")
            if not cont:
                break
            p = {**params, **cont}


class _Retryable(Exception):
    """For internal use: an error that can be retried (for example maxlag)."""
