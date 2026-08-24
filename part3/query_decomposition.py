"""Structural clause splitting for comparison and multi-part questions --
e.g. "Compare footwear and electronics returns" or "I paid COD, it arrived
late, and it's damaged, can I return it?". This splits on generic English
coordination (commas, "and", "but", "compare X and Y", "what about"), never
on a domain word list -- the splitter has no idea what footwear or COD is.
Each resulting clause is later embedded and retrieved independently by
part3/retrieval.py; clause count and content decide multi-clause mode, not
any keyword lookup.
"""
import re

from part3.config import MIN_CLAUSE_WORDS

_COMPARE_PATTERN = re.compile(
    r"\bcompare\b\s+(.+?)\s+(?:and|with|vs\.?|versus)\s+(.+)", re.IGNORECASE
)
_SPLIT_PATTERN = re.compile(r",|\band\b|\bbut\b|\bversus\b|\bvs\.?\b", re.IGNORECASE)
_LEADING_FILLER = re.compile(
    r"^(what about|and|also|how about|what if|is it|can it|can i)\b[\s,]*", re.IGNORECASE
)


def split_clauses(text: str) -> list[str]:
    """Split a message into candidate clauses. Returns a single-item list
    (the original text, trimmed) when the message doesn't structurally
    decompose into multiple parts -- ordinary single-topic questions are
    passed through unchanged.
    """
    text = text.strip()
    if not text:
        return []

    compare_match = _COMPARE_PATTERN.search(text)
    if compare_match:
        # "compare X and Y" already names two topics directly -- X/Y are
        # legitimately short (often one word, e.g. "footwear"), so the
        # generic word-count filter below doesn't apply to this path.
        left, right = compare_match.group(1).strip(" ?."), compare_match.group(2).strip(" ?.")
        clauses = [_LEADING_FILLER.sub("", c).strip() for c in (left, right) if c.strip()]
        return clauses if len(clauses) >= 2 else [text]

    clauses = [c.strip(" ?.") for c in _SPLIT_PATTERN.split(text)]
    clauses = [_LEADING_FILLER.sub("", c).strip() for c in clauses if c.strip()]
    clauses = [c for c in clauses if len(c.split()) >= MIN_CLAUSE_WORDS]

    return clauses if len(clauses) >= 2 else [text]


def is_short_followup(text: str) -> bool:
    """Heuristic for a pronoun-style follow-up ("What about if they're
    damaged?", "And how long do I have?") -- short messages that open with a
    continuation word rather than naming a topic. Used to decide whether to
    retry retrieval with the previous turn's message prepended for context;
    this is context-concatenation, not real coreference resolution.
    """
    words = text.strip().split()
    return bool(words) and len(words) <= 8 and bool(_LEADING_FILLER.match(text.strip()))
