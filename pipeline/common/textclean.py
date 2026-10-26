"""
textclean.py: parsing and cleaning wikitext (built on mwparserfromhell)

Module: common layer (CODE_PLAN.md section 4)
Rules: DATA_PROCESSING R4.4 (remove the away note), R4.5 (classify links by the
       official namespaces), R4.6 (remove templates entirely);
Decisions: D9 item 5 (no more hand-written name lists)

1. What it does
    1. strip_away_note(): removes the away note that the system inserts on the
       first line of a question, and reads the primary mentor and mentee names
       from it.
    2. parse(): uses mwparserfromhell to parse wikitext into a syntax tree.
    3. to_plain_text(): removes signatures, user links, all templates, HTML tags
       and comments, and keeps the display text of links. The result is plain text
       for people to read and for sentiment analysis and the LLM.
    4. make_emb_text(): starts from the plain text and also removes the user names
       that appear in it. The result is the text used for embeddings.
    5. extract_links() / extract_templates() / count_list_items() /
       count_external_links(): pull structural features out of the syntax tree,
       which M6 uses to compute wiki formatting features.

2. Why it is designed this way
    In the paper version, s7 cleaned text with dozens of template regexes and a
    hand-written name list. The list dealt only with the signature styles of 18
    users (Qwerfjkl, Fritzmann, Polygnotus, and others). That approach was neither
    complete nor general: new signature styles slip through, and there is no way
    to justify why those particular people were chosen.

    Here we do this instead:
        - wikitext is parsed by mwparserfromhell (the standard parser library of
          the Wikipedia bot community);
        - all templates are removed (R4.6): templates on talk pages are mostly
          formatting, notices and pings, not part of what people say;
        - signatures are removed by their structure ("user link + timestamp"), not
          by a list of names;
        - the user names removed from the embedding text are the speakers who
          actually appear in this conversation (passed in by the caller), not a
          fixed list written in advance.

    The paper version's s7 cleaning functions are used only to reproduce stage A
    (loaded from git 23fea07). The new pipeline no longer uses them.

3. Inputs
    A wikitext string.

4. Outputs
    A plain text string, or structural features.

6. Possible errors and how they are handled
    mwparserfromhell can parse any string (invalid markup is treated as plain
    text), so it does not raise exceptions. If the text is empty after cleaning
    (for example, a question that is only a signature), the caller handles it
    under the R4 and R5 rules.

8. How to check
    tests/test_common.py::test_textclean_*, including the away note in revid
    1343488016.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import mwparserfromhell as mwp

from pipeline.common.timeutil import SIG_TS_RE
from pipeline.common.users import norm_user
from pipeline.common.wikimeta import NamespaceTable, namespace_table

# ── Away note (R4.4) ─────────────────────────────────────────────────────────
# Form seen in the real data: :''Note: [[User talk:Mentee|Mentee]]'s mentor [[User talk:PrimaryMentor|PrimaryMentor]] is away.''
# Default i18n form: Note: [[User:Mentee|Mentee]]'s mentor <span class="plainlinks">[{{fullurl:User:PrimaryMentor}} PrimaryMentor]</span> is away.
_LINK = r"\[\[\s*User(?:[ _]talk)?\s*:\s*(?P<{g}>[^|\]]+)(?:\|[^\]]*)?\]\]"
_EXT = r"(?:<span[^>]*>)?\s*\[\S+\s+(?P<{g}>[^\]]+)\]\s*(?:</span>)?"
AWAY_NOTE_RE = re.compile(
    r"^\s*:?\s*(?:''|<i>)?\s*Note:\s*" + _LINK.format(g="mentee") +
    r"\s*'s\s+mentor\s+(?:" + _LINK.format(g="primary") + "|" + _EXT.format(g="primary2") + r")"
    r"\s+is\s+away(?:\s+until\s+[^.'\n<]*)?\.?\s*(?:''|</i>)?[ \t]*\n?",   # "is away." or "is away until October 2026."
    re.I | re.M,
)
#: The away note only appears at the very start of a question. But sometimes a
#: template line comes before it (for example {{Main|...}}). So we search for it
#: at the start of any line within the first AWAY_SEARCH_CHARS characters, not
#: only at the first character.
AWAY_SEARCH_CHARS = 1000

# Leftovers of signatures: after user links are removed, fragments like
# "-- ( talk) 15:55, 27 March 2024 (UTC)" are often left behind
_SIG_REMNANT_RE = re.compile(r"(?:[-\u2013\u2014~]{1,3}\s*)?\(\s*(?:talk|contribs?|t|c|·|\s|/|\|)*\)\s*")
_WS_RE = re.compile(r"[ \t ]+")
_BLANKLINES_RE = re.compile(r"\n\s*\n+")
_USER_NS = {2, 3}           # User, User talk
_SPECIAL_NS = -1


@dataclass(frozen=True)
class AwayNote:
    """Information read from the away note."""
    mentee: str | None
    primary_mentor: str | None


def strip_away_note(text: str | None) -> tuple[str, AwayNote | None]:
    """Remove the system away note from the first line of a question (R4.4), and
    read the primary mentor from it.

    Args:
        text: the raw wikitext of the question.

    Returns:
        (text with the note removed, AwayNote or None). If there is no note, the
        text is returned unchanged together with None.

    Example:
        revid 1343488016 starts with
        ":''Note: [[User talk:SashankInAnitquity|SashankInAnitquity]]'s mentor
          [[User talk:Dr vulpes|Dr vulpes]] is away.''"
        -> AwayNote(mentee="SashankInAnitquity", primary_mentor="Dr vulpes")
    """
    if not text:
        return text or "", None
    m = AWAY_NOTE_RE.search(text[:AWAY_SEARCH_CHARS])
    if not m:
        return text, None
    primary = m.group("primary") or m.group("primary2")
    return text[:m.start()] + text[m.end():], AwayNote(norm_user(m.group("mentee")), norm_user(primary))


def parse(text: str | None) -> mwp.wikicode.Wikicode:
    """Parse wikitext with mwparserfromhell. Empty input gives an empty syntax tree."""
    return mwp.parse(text or "")


def _is_user_link(link, nst: NamespaceTable) -> bool:
    """Whether a link points to a user: User:, User talk:, Special:Contributions/."""
    parts = nst.split(str(link.title))
    if parts.ns in _USER_NS:
        return True
    return parts.ns == _SPECIAL_NS and parts.page.lower().startswith(("contributions/", "contribs/"))


def to_plain_text(text: str | None, nst: NamespaceTable | None = None) -> str:
    """Turn wikitext into plain text (R4.6).

    Steps:
        1. parse into a syntax tree;
        2. remove all templates (removing a top-level template also removes any
           templates nested inside it);
        3. remove links that point to users (the core of signatures and pings), as
           well as file/image links, category links and <gallery> (these do not
           show up as text on the page);
        4. strip_code: other links keep their display text, HTML tags keep only
           their content, comments are removed;
        5. remove signature timestamps and signature leftovers, and merge
           whitespace.

    Args:
        text: wikitext.
        nst: namespace table; defaults to the shared table.

    Returns:
        Plain text. Paragraph breaks are kept (several blank lines become one).
    """
    nst = nst or namespace_table()
    code = parse(text)
    for t in code.filter_templates(recursive=False):
        try:
            code.remove(t)
        except ValueError:      # already removed together with its parent
            pass
    for link in code.filter_wikilinks(recursive=True):
        title = str(link.title).strip()
        # File/image (ns 6) and category (ns 14) links do not show as text on the page,
        # so remove them entirely (links that start with ":" are ordinary links and are kept)
        hidden = not title.startswith(":") and nst.split(title).ns in (6, 14)
        if _is_user_link(link, nst) or hidden:
            try:
                code.remove(link)
            except ValueError:
                pass
    for tag in code.filter_tags(recursive=False):
        if str(tag.tag).strip().lower() == "gallery":
            try:
                code.remove(tag)
            except ValueError:
                pass
    plain = code.strip_code(normalize=True, collapse=False)
    plain = SIG_TS_RE.sub("", plain)
    plain = _SIG_REMNANT_RE.sub(" ", plain)
    plain = _WS_RE.sub(" ", plain)
    plain = "\n".join(line.strip() for line in plain.splitlines())
    plain = _BLANKLINES_RE.sub("\n\n", plain).strip()
    # Talk page indent marks (: * # at the start of a line) carry no content
    plain = re.sub(r"(?m)^[:*#]+\s*", "", plain)
    # Dash left over before a signature ("-- [[User:X]]" leaves "--" once the user link is removed)
    plain = re.sub(r"(?m)\s*[-\u2013\u2014~]+\s*$", "", plain)
    # "Hi there ," left after a pinged name is removed -> "Hi there,"
    plain = re.sub(r"[ \t]+([,.;:!?])(?=\s|$)", r"\1", plain)
    # Links like [[:File:X]] display as "File:X", but strip_code leaves the leading colon
    plain = re.sub(r"(?<!\S):(?=[A-Z][\w ]*:)", "", plain)
    return plain.strip()


def make_emb_text(text: str | None, user_names: list[str] | tuple[str, ...] = (),
                  nst: NamespaceTable | None = None) -> str:
    """Make the text used for embeddings and LLM labeling: the plain text, with the
    user names that appear in the conversation also removed.

    Args:
        text: wikitext.
        user_names: user names that actually appear in this conversation (mentee,
                    mentor, other repliers). The caller gets them from the list of
                    speakers. No fixed list written in advance is used.

    Returns:
        Plain text with the user names removed.

    Notes:
        Names are replaced from longest to shortest, so that replacing a short name
        first does not break a longer name that contains it. Matching is on whole
        words and ignores case.
    """
    plain = to_plain_text(text, nst)
    for name in sorted({n for n in user_names if n and len(n) >= 3}, key=len, reverse=True):
        pat = r"(?<!\w)" + re.escape(name).replace(r"\ ", r"[ _]") + r"(?!\w)"
        plain = re.sub(pat, " ", plain, flags=re.I)
    plain = _WS_RE.sub(" ", plain)
    return re.sub(r"\s+\n", "\n", plain).strip()


def extract_links(text: str | None, nst: NamespaceTable | None = None) -> list[dict]:
    """Extract wikilinks and classify them with the official namespace table (R4.5).

    Returns:
        [{"target": raw target, "full": standard title, "ns": namespace number,
          "display": display text}]
        Links that point to users are not included (they are part of signatures).
        File embeds (File:) are also excluded, because they are not links.
    """
    nst = nst or namespace_table()
    out = []
    for link in parse(text).filter_wikilinks(recursive=True):
        if _is_user_link(link, nst):
            continue
        parts = nst.split(str(link.title))
        if parts.ns == 6 and not str(link.title).lstrip().startswith(":"):
            continue
        out.append({"target": str(link.title).strip(), "full": parts.full, "ns": parts.ns,
                    "display": str(link.text).strip() if link.text else parts.page})
    return out


def extract_templates(text: str | None) -> list[str]:
    """Extract template names (normalized, including nested templates)."""
    return [norm_user(str(t.name).strip()) or "" for t in parse(text).filter_templates(recursive=True)]


def count_external_links(text: str | None) -> int:
    """Number of external links ([http://... text] and bare URLs)."""
    return len(parse(text).filter_external_links(recursive=True))


def count_list_items(text: str | None) -> int:
    """Number of list items: items whose line starts with * or # (Help:Lists).
    The : used for talk page indenting does not count.

    Notes:
        mwparserfromhell parses list items as Tag nodes whose wiki_markup is "*"
        or "#".
    """
    n = 0
    for tag in parse(text).filter_tags(recursive=True):
        if getattr(tag, "wiki_markup", None) in ("*", "#"):
            n += 1
    return n
