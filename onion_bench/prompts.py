"""Prompt construction. Only `title` and `text` from a dataset row are used.

Prompt variants (recorded as `prompt_variant` on every response):
- full:          headline + full body text
- headline_lead: headline + the article's opening (whole sentences from the
                 start of the body until LEAD_MIN_WORDS is reached), verbatim
                 and built identically for every source
- headline_only: headline alone
"""

from __future__ import annotations

import json
import re

from .assemble import PROMPT_FIELDS

PROMPT_VERSION = "v1"

_SHOWN_FULL = "You will be shown one news article: its headline and body text."
_SHOWN = {
    "full": _SHOWN_FULL,
    "headline_lead": "You will be shown one news article: its headline and the opening of its body text.",
    "headline_only": "You will be shown the headline of one news article.",
}
VARIANTS = tuple(_SHOWN)
LEAD_MIN_WORDS = 40

SYSTEM_PROMPT = """\
You are taking part in a news-verification study. You will be shown one news article: its headline and body text. Nothing else about it (outlet, author, URL, date) is given.

Decide whether the article is REAL or FAKE:
- real: a genuine news report about events that actually happened.
- fake: satire, parody, or otherwise fabricated; the events it reports did not happen as described.

The article may be about events after your training data ends, and you have no internet or tool access, so you cannot look anything up. Judge from the content alone and always commit to your best guess.

Answer with a JSON object: "verdict" is "real" or "fake", and "p_fake" is your probability, as an integer from 0 to 100, that the article is fake."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["real", "fake"]},
        "p_fake": {"type": "integer"},
    },
    "required": ["verdict", "p_fake"],
    "additionalProperties": False,
}


def system_prompt(variant: str = "full") -> str:
    if variant not in _SHOWN:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {VARIANTS}")
    return SYSTEM_PROMPT.replace(_SHOWN_FULL, _SHOWN[variant])


# A sentence ends at . ! ? or … (optionally followed by closing quotes/brackets)
# when whitespace and an upper-case letter, digit or opening quote follow.
_NEXT = r"(?=[\"'“‘«(\[]?[A-ZÀ-ÖØ-Þ0-9])"
_SENT_END = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’»)\]]))\s+" + _NEXT)


# Pieces ending in an initial or a common abbreviation are not sentence ends.
_ABBREV_END = re.compile(
    r"(?:\b[A-Z]|\b(?:Mr|Mrs|Ms|Dr|St|Sen|Rep|Gov|Pres|Jr|Sr|Sra|Dra|Lt|Gen|Col|Capt|Sgt|Prof|Rev|Inc|Co|Corp|Ltd|No|Nº|vs|etc|Av|Eng|Mt|Ft|Jan|Feb|Mar|Apr|Aug|Sept?|Oct|Nov|Dec))\.$"
)


def _sentences(para: str) -> list[str]:
    out: list[str] = []
    for piece in (x.strip() for x in _SENT_END.split(para) if x.strip()):
        if out and _ABBREV_END.search(out[-1]):
            out[-1] = out[-1] + " " + piece
        else:
            out.append(piece)
    return out or [para]


def lead(text: str, min_words: int = LEAD_MIN_WORDS) -> str:
    """Whole sentences from the start of the body until min_words is reached.
    Never cuts inside a sentence; keeps paragraph breaks; verbatim."""
    out: list[str] = []
    n = 0
    for para in text.split("\n\n"):
        # re.split drops the separators; rebuild with single spaces (whitespace is
        # already normalised to single spaces inside a paragraph).
        kept = []
        for sent in _sentences(para):
            kept.append(sent)
            n += len(sent.split())
            if n >= min_words:
                break
        out.append(" ".join(kept))
        if n >= min_words:
            break
    return "\n\n".join(out)


def user_message(item: dict, variant: str = "full") -> str:
    title, text = (item[f] for f in PROMPT_FIELDS)
    if variant == "full":
        return f"<headline>\n{title}\n</headline>\n\n<article>\n{text}\n</article>"
    if variant == "headline_lead":
        return f"<headline>\n{title}\n</headline>\n\n<article_opening>\n{lead(text)}\n</article_opening>"
    if variant == "headline_only":
        return f"<headline>\n{title}\n</headline>"
    raise ValueError(f"unknown prompt variant {variant!r}; choose from {VARIANTS}")


def parse_answer(text: str) -> tuple[str | None, float | None]:
    """Return (verdict, p_fake in [0, 1]) or (None, None) when unparseable."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        start, end = (text or "").find("{"), (text or "").rfind("}")
        if start < 0 or end <= start:
            return None, None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None, None
    verdict = obj.get("verdict") if isinstance(obj, dict) else None
    if verdict not in ("real", "fake"):
        return None, None
    p = obj.get("p_fake")
    p_fake = min(max(float(p), 0.0), 100.0) / 100.0 if isinstance(p, (int, float)) else None
    return verdict, p_fake
