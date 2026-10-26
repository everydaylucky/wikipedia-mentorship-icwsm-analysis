"""
wikimeta.py: the namespace table and page lookups (redirects, categories)

Module: common layer (CODE_PLAN.md section 4)
Rules: DATA_PROCESSING R4.5 (link types follow the official namespace table and
       redirects)

1. What it does
    1. NamespaceTable: reads the official namespace and alias table of English
       Wikipedia (API:Siteinfo, stored in data/raw/wiki_meta/siteinfo_enwiki.json)
       and splits any link target into "namespace + page name". For example,
       "WP:RS" -> (4, "RS").
    2. resolve_titles(): uses the API to follow redirects and find the page and
       namespace a link finally points to. Pseudo-namespaces (for example
       "H:PICTURE") live in the article namespace, so this is the only way to
       resolve them (Wikipedia:Shortcut).
    3. policy_status(): decides whether a page is a policy or guideline page. It
       uses the official categories Category:Wikipedia policies and
       Category:Wikipedia guidelines, not a hand-made list.

2. Why it is designed this way
    The old code used regexes on prefixes like "WP:" and "Help:" to decide the link
    type. This had two problems:
        - it missed aliases and case variants (for example "Project:", "wp:");
        - it treated pseudo-namespaces ("H:", "MOS:") as real namespaces.
    The official table plus API lookups avoid both problems completely. API results
    are cached by MediaWikiClient, so offline reruns do not go online again.

3. Inputs
    siteinfo_enwiki.json; MediaWikiClient when lookups are needed.

4. Outputs
    Function return values.

6. Possible errors and how they are handled
    - Unknown prefix: treated as part of an ordinary page name in the article
      namespace (ns 0). This is what MediaWiki itself does.
    - API returns missing: the page does not exist, recorded as exists=False
      (a red link).

8. How to check
    tests/test_common.py::test_namespace_table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import config
from pipeline.common.api import MediaWikiClient

# Namespace numbers this study cares about (Manual:Namespace)
NS_MAIN, NS_TALK, NS_USER, NS_USER_TALK = 0, 1, 2, 3
NS_PROJECT, NS_FILE, NS_TEMPLATE, NS_HELP, NS_CATEGORY = 4, 6, 10, 12, 14
NS_DRAFT, NS_SPECIAL = 118, -1

POLICY_CATEGORIES = ("Category:Wikipedia policies", "Category:Wikipedia guidelines")


@dataclass(frozen=True)
class TitleParts:
    """The result of splitting a link target.

    Attributes:
        ns: namespace number.
        ns_name: the standard namespace name (empty string for the article namespace).
        page: the page name without the namespace (with any #anchor removed).
        full: the standard full title, for example "Wikipedia:Reliable sources".
    """
    ns: int
    ns_name: str
    page: str
    full: str


class NamespaceTable:
    """The official namespace and alias table (API:Siteinfo).

    Usage:
        nst = NamespaceTable.load()
        nst.split("WP:RS")          # TitleParts(ns=4, ns_name="Wikipedia", page="RS", ...)
        nst.split(":Help:Contents") # a leading colon means "link, do not embed"; it is removed when parsing
    """

    def __init__(self, siteinfo: dict):
        q = siteinfo["query"]
        ns_list = q["namespaces"].values() if isinstance(q["namespaces"], dict) else q["namespaces"]
        self.names: dict[int, str] = {}
        self._lookup: dict[str, int] = {}
        for ns in ns_list:
            nid = int(ns["id"])
            name = ns.get("name", "")
            self.names[nid] = name
            for alias in {name, ns.get("canonical", "")}:
                if alias:
                    self._lookup[alias.lower()] = nid
        for a in q.get("namespacealiases", []):
            self._lookup[a["alias"].lower()] = int(a["id"])

    @classmethod
    def load(cls, path: Path | None = None) -> "NamespaceTable":
        """Load from data/raw/wiki_meta/siteinfo_enwiki.json."""
        p = path or config.RAW_FILES["siteinfo"]
        return cls(json.loads(Path(p).read_text()))

    def split(self, target: str) -> TitleParts:
        """Split a link target into namespace and page name.

        Args:
            target: the target of a wikilink, for example "WP:RS", "User talk:Foo",
                "Article#Section".

        Returns:
            TitleParts. If the prefix is not a known namespace or alias, the whole
            string is a page name in the article namespace.

        Notes:
            Namespace names are case insensitive. The first letter of the page name
            is capitalized, following $wgCapitalLinks. Underscores and spaces are
            the same. The "#anchor" part is removed.
        """
        t = target.strip().lstrip(":").split("#", 1)[0].replace("_", " ").strip()
        ns, page = NS_MAIN, t
        if ":" in t:
            prefix, rest = t.split(":", 1)
            nid = self._lookup.get(prefix.strip().lower())
            if nid is not None:
                ns, page = nid, rest.strip()
        page = (page[:1].upper() + page[1:]) if page else page
        ns_name = self.names.get(ns, "")
        full = f"{ns_name}:{page}" if ns_name else page
        return TitleParts(ns, ns_name, page, full)


@lru_cache(maxsize=1)
def namespace_table() -> NamespaceTable:
    """The shared namespace table for the whole project (loaded only once)."""
    return NamespaceTable.load()


def resolve_titles(titles: Iterable[str], client: MediaWikiClient) -> dict[str, dict]:
    """Follow redirects for a batch of titles. Finds the final target, its
    namespace, and whether it exists (R4.5).

    Args:
        titles: standard full titles (NamespaceTable.split(...).full).
        client: API client; when offline, only the cache can be used.

    Returns:
        {original title: {"final": final title, "ns": final namespace,
                          "exists": bool, "redirected": bool}}

    Notes:
        Sends 50 titles per batch (the API limit), with redirects=1. Results are
        cached by client.
    """
    titles = sorted({t for t in titles if t})
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = client.get({"action": "query", "titles": "|".join(batch), "redirects": 1})
        q = data.get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for t in batch:
            cur = norm.get(t, t)
            final = redir.get(cur, cur)
            pg = pages.get(final, {})
            out[t] = {"final": final, "ns": pg.get("ns"), "exists": not pg.get("missing", False),
                      "redirected": final != cur}
    return out


def policy_status(titles: Iterable[str], client: MediaWikiClient) -> dict[str, str | None]:
    """Decide whether each page is a policy, a guideline, or neither, using the
    official categories (not a hand-made list).

    Args:
        titles: final titles (redirects already followed).

    Returns:
        {title: "policy" | "guideline" | None}
    """
    titles = sorted({t for t in titles if t})
    out: dict[str, str | None] = {t: None for t in titles}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = client.get({"action": "query", "titles": "|".join(batch), "prop": "categories",
                           "clcategories": "|".join(POLICY_CATEGORIES), "cllimit": "max"})
        for p in data.get("query", {}).get("pages", []):
            cats = {c["title"] for c in p.get("categories", [])}
            if POLICY_CATEGORIES[0] in cats:
                out[p["title"]] = "policy"
            elif POLICY_CATEGORIES[1] in cats:
                out[p["title"]] = "guideline"
    return out
