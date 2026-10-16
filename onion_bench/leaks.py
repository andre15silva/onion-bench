"""Detect source-identifying strings in the model-facing fields.

The prompt only ever contains `title` and `text`, so URL, outlet name, byline,
section, tags etc. can never reach the model. But article text itself sometimes
names the outlet ("the BBC understands...", "o Imprensa Falsa sabe..."). We do
not edit the text; instead such items are flagged and, by default, excluded
from the eval split (`eval_eligible=False`) while kept in the dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LeakRule:
    source: str  # which source the pattern identifies
    pattern: str
    flags: int = 0  # re flags; patterns are case-sensitive unless re.I is given

    def compile(self) -> re.Pattern:
        return re.compile(self.pattern, self.flags)


# Generic satire markers. These don't identify an outlet, but they would give
# the answer away if they appeared as a disclaimer, so they are reported too.
HINT_RULES = [
    LeakRule("_satire_hint", r"\b(?:satire|satirical|parody|sátira|satíric[oa]s?|paródia)\b", re.I),
]


def rules_from_sources(sources: list[dict]) -> list[LeakRule]:
    rules: list[LeakRule] = []
    for s in sources:
        for p in s.get("leak_patterns", []):
            if isinstance(p, dict):
                rules.append(LeakRule(s["name"], p["pattern"], re.I if p.get("ignore_case") else 0))
            else:
                rules.append(LeakRule(s["name"], p))
    return rules


def find_leaks(title: str, text: str, rules: list[LeakRule]) -> list[dict]:
    """Return one entry per (rule, field) match: {"source", "field", "match"}."""
    hits = []
    for rule in rules + HINT_RULES:
        rx = rule.compile()
        for field_name, value in (("title", title or ""), ("text", text or "")):
            m = rx.search(value)
            if m:
                hits.append({"source": rule.source, "field": field_name, "match": m.group(0)})
    return hits


def is_blocking(hits: list[dict]) -> bool:
    """Outlet mentions block eval eligibility; generic satire hints only do so in the title."""
    return any(h["source"] != "_satire_hint" or h["field"] == "title" for h in hits)
