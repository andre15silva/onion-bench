"""Prompt construction. Only `title` and `text` from a dataset row are used."""

from __future__ import annotations

import json

from .assemble import PROMPT_FIELDS

PROMPT_VERSION = "v1"

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


def user_message(item: dict) -> str:
    title, text = (item[f] for f in PROMPT_FIELDS)
    return f"<headline>\n{title}\n</headline>\n\n<article>\n{text}\n</article>"


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
