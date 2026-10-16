"""Synthetic counterfactual fakes: a real post-cutoff article with one
plausible factual change.

The generator (a model that is NOT evaluated; default claude-haiku-5-5 via
the claude-cli backend) does not rewrite the article. It returns exact
find/replace edits, which are applied programmatically, so every other word
stays verbatim and the diff is explicit. The change must alter the article's
main claim, so it has to show up in the headline and in the opening
(`prompts.lead`), and therefore in every prompt variant.

Automatic checks decide `eligible`; a human can confirm or reject each item
with `review-counterfactuals` (recorded in provenance.human_verified).

To limit harm, the generator is told never to attribute crimes, scandals,
wrongdoing or embarrassing statements to real, named people, and to change
numbers, places, dates, outcomes of decisions, or institutional facts
instead. Records are clearly labelled synthetic (`sourcing_mode`).
"""

from __future__ import annotations

import asyncio
import difflib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .assemble import item_id
from .cli_backend import call_structured
from .leaks import find_leaks, rules_from_sources
from .prompts import lead
from .scrape import _append_jsonl, _read_jsonl, load_sources

GENERATOR_MODEL = "claude-haiku-5-5"
GENERATOR_VERSION = "cf-v2"
MAX_CHANGED_WORD_SHARE = 0.15

SYSTEM = """You help build a research benchmark that measures whether AI models can detect false news. The output is never published as news: every item is stored with a label saying it was deliberately altered.

You will receive a real news article (headline and body). Produce a counterfactual version with ONE plausible factual change to its main claim, so that the edited article reports something that did not happen, while reading exactly like the original outlet's reporting.

Rules:
- Change one core fact: a number or amount, a place, a date or time, the outcome of a vote, decision or event, or which institution/organisation did something. Propagate it consistently: edit every sentence that states or depends on that fact (headline, opening, later mentions) so the article stays internally consistent.
- The change MUST appear in the headline and in the first sentence or paragraph of the body.
- Make it material: someone who knew the true story should recognise the edited article as reporting a different, false event. Good: an outcome reversed (approved -> rejected, won -> lost, rose -> fell), a different institution or country acting, a number changed substantially (e.g. 4 dead -> 14 dead, 2% -> 7%), a different place. Bad: off-by-one numbers, synonyms, minor descriptive details.
- Keep it plausible and sober: no exaggeration, humour, or implausible values. A careful reader should not be able to tell from the text alone.
- Do NOT attribute crimes, scandals, wrongdoing, illnesses, deaths or embarrassing statements to real, named people, and do not invent quotes. Do not target private individuals.
- Keep the article's language. Change as few words as possible; everything else stays word-for-word.

Return the edits as exact find/replace pairs. `find` must be copied character-for-character from the given field (headline -> "title", body -> "text") and be long enough to be unambiguous (prefer a whole phrase over a single short word); every occurrence of `find` in that field will be replaced."""

SCHEMA = {
    "type": "object",
    "properties": {
        "replacements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": ["title", "text"]},
                    "find": {"type": "string"},
                    "replace": {"type": "string"},
                },
                "required": ["field", "find", "replace"],
                "additionalProperties": False,
            },
        },
        "change_type": {"type": "string", "enum": ["number", "place", "date_or_time", "outcome", "actor_institution", "other"]},
        "change_summary": {"type": "string"},
        "why_plausible": {"type": "string"},
    },
    "required": ["replacements", "change_type", "change_summary", "why_plausible"],
    "additionalProperties": False,
}


def select_bases(items: list[dict], per_language: dict[str, int], seed: int = 0) -> list[dict]:
    """Eligible real control articles with body text, round-robin over sources."""
    rng = random.Random(seed)
    out = []
    for lang, n in per_language.items():
        by_src: dict[str, list[dict]] = defaultdict(list)
        for it in sorted(items, key=lambda r: r["id"]):
            if it["language"] == lang and it["subset"] == "real_control" and it.get("has_text", True) and it["eval_eligible"]:
                by_src[it["source"]].append(it)
        for v in by_src.values():
            rng.shuffle(v)
        picked = []
        while len(picked) < n and any(by_src.values()):
            for s in sorted(by_src):
                if by_src[s] and len(picked) < n:
                    picked.append(by_src[s].pop())
        out += picked
    return out


def apply_replacements(title: str, text: str, reps: list[dict]) -> tuple[str, str, list[dict]]:
    fields = {"title": title, "text": text}
    applied = []
    for r in reps:
        f, find, rep = r.get("field"), r.get("find", ""), r.get("replace", "")
        n = fields[f].count(find) if f in fields and find else 0
        if n and find != rep:
            fields[f] = fields[f].replace(find, rep)
        applied.append({"field": f, "find": find, "replace": rep, "occurrences": n})
    return fields["title"], fields["text"], applied


def changed_word_share(a: str, b: str) -> float:
    wa, wb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    same = sum(blk.size for blk in sm.get_matching_blocks())
    return 1 - same / max(len(wa), 1)


def check(base: dict, new_title: str, new_text: str, applied: list[dict], rules) -> dict[str, bool]:
    before = {(h["source"], h["match"]) for h in find_leaks(base["title"], base["text"], rules)}
    after = {(h["source"], h["match"]) for h in find_leaks(new_title, new_text, rules)}
    return {
        "has_edits": bool(applied),
        "all_finds_matched": bool(applied) and all(a["occurrences"] > 0 for a in applied),
        "no_ambiguous_short_finds": all(not (len(a["find"]) < 4 and a["occurrences"] > 2) for a in applied),
        "title_changed": new_title != base["title"],
        "opening_changed": lead(new_text) != lead(base["text"]),
        "small_edit": 0 < changed_word_share(base["title"] + "\n" + base["text"], new_title + "\n" + new_text) <= MAX_CHANGED_WORD_SHARE,
        "no_new_outlet_mentions": after <= before,
    }


async def generate(
    bases: list[dict], out_path: Path, *, model: str = GENERATOR_MODEL, effort: str = "medium", concurrency: int = 4,
) -> dict:
    done = {r["provenance"]["base_id"] for r in _read_jsonl(out_path)}
    todo = [b for b in bases if b["id"] not in done]
    rules = rules_from_sources(load_sources())
    sem = asyncio.Semaphore(concurrency)
    counts: dict[str, int] = defaultdict(int)

    async def one(base: dict) -> None:
        async with sem:
            user = f"<headline>\n{base['title']}\n</headline>\n\n<article>\n{base['text']}\n</article>"
            res = await call_structured(model, effort, SYSTEM, user, SCHEMA)
        o = res["output"] or {}
        new_title, new_text, applied = apply_replacements(base["title"], base["text"], o.get("replacements") or [])
        checks = check(base, new_title, new_text, applied, rules) if res["setup_ok"] and o else {"generated": False}
        eligible = bool(res["setup_ok"] and o and all(checks.values()))
        counts["eligible" if eligible else "failed_checks"] += 1
        _append_jsonl(out_path, {
            "source": "counterfactual",
            "url": base["url"] + f"#counterfactual-{GENERATOR_VERSION}",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "title": new_title,
            "original_title": None,
            "text": new_text,
            "published_at": base["published_at"],
            "modified_at": None,
            "authors": [],
            "section": base.get("section"),
            "keywords": [],
            "page_language": None,
            "site_name": None,
            "extraction_method": "llm_find_replace_edit",
            "word_count": len(new_text.split()),
            "language": base["language"],
            "label": "fake",
            "subset": "counterfactual",
            "sourcing_mode": "synthetic_counterfactual",
            "selection_method": f"counterfactual_of:{base['source']}",
            "eligible": eligible,
            "provenance": {
                "base_id": base["id"],
                "base_url": base["url"],
                "base_source": base["source"],
                "generator_model": model,
                "generator_effort": effort,
                "generator_version": GENERATOR_VERSION,
                "generator_backend": "claude_cli",
                "model_served": res["model_served"],
                "replacements": applied,
                "change_type": o.get("change_type"),
                "change_summary": o.get("change_summary"),
                "why_plausible": o.get("why_plausible"),
                "auto_checks": checks,
                "generation_error": res["error"] if not o else None,
                "human_verified": None,
            },
        })

    await asyncio.gather(*(one(b) for b in todo))
    return dict(counts)


def review(path: Path) -> None:
    """Interactive y/n review of eligible, not yet reviewed counterfactuals."""
    rows = _read_jsonl(path)
    try:
        for r in rows:
            p = r["provenance"]
            if not r["eligible"] or p.get("human_verified") is not None:
                continue
            print(f"\n[{r['language']}] {p['change_type']}: {p['change_summary']}")
            for a in p["replacements"]:
                print(f"  {a['field']}: «{a['find']}» → «{a['replace']}» (x{a['occurrences']})")
            print(f"  NEW HEADLINE: {r['title']}\n  OPENING: {lead(r['text'])[:300]}")
            ans = input("Plausible, false, harmless? [y/n/q] ").strip().lower()
            if ans == "q":
                break
            if ans in ("y", "n"):
                p["human_verified"] = ans == "y"
                if ans == "n":
                    r["eligible"] = False
    finally:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
