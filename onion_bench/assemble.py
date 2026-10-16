"""Merge raw per-source files into the benchmark dataset (one row per article)."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

from .leaks import find_leaks, is_blocking, rules_from_sources
from .scrape import _read_jsonl, load_sources, reject_reason

# The ONLY fields that are ever shown to a model. Everything else is metadata
# kept for traceability and analysis.
PROMPT_FIELDS = ("title", "text")

DATASET_COLUMNS = [
    "id", "title", "text", "has_text", "language", "label", "subset", "original_title",
    "source", "source_name", "url", "published_at", "modified_at", "authors",
    "section", "keywords", "fetched_at", "extraction_method",
    "sourcing_mode", "provenance",
    "word_count", "content_sha256", "leak_flags", "eval_eligible", "selection_method",
    "implausibility_score",
]

# How an item got into the dataset (column `sourcing_mode`); details are in `provenance`.
SOURCING_MODES = {
    "publisher_scrape": "article scraped from the publisher's site (RSS/sitemap/listing discovery)",
    "reddit_linked_scrape": "article linked from a subreddit post, scraped from its publisher",
    "manual_html_import": "article page saved by hand from a browser and imported",
    "manual_index_import": "headline taken from a section/index page saved by hand (no body text)",
    "fact_check_claimreview": "claim text and verdict from a fact-checker's schema.org ClaimReview (no body text)",
    "synthetic_counterfactual": "real article with one plausible factual change made by an LLM (not an evaluated model)",
}


def sourcing_mode(rec: dict, src: dict) -> str:
    if rec.get("sourcing_mode"):
        return rec["sourcing_mode"]
    legacy = {"manual_html_import": "manual_html_import", "manual_index_import": "manual_index_import"}
    return legacy.get(rec.get("collected_via", ""), src.get("sourcing_mode", "publisher_scrape"))


def item_id(url: str) -> str:
    return "ob-" + hashlib.sha256(url.encode()).hexdigest()[:12]


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).lower().split())


def content_hash(title: str, text: str) -> str:
    return hashlib.sha256((_norm(title) + "\n" + _norm(text)).encode()).hexdigest()


def assemble(
    raw_dir: Path,
    out_path: Path,
    *,
    since: datetime,
    until: datetime | None = None,
    min_words: int = 40,
    implausible_file: Path | None = None,
) -> dict:
    sources = load_sources()
    rules = rules_from_sources(sources)

    selected: dict[str, dict] = {}
    if implausible_file and implausible_file.exists():
        for r in _read_jsonl(implausible_file):
            if r.get("approved") is True:
                selected[r["id"]] = r

    rows, seen_hash = [], set()
    stats: Counter = Counter()
    for src in sources:
        for rec in _read_jsonl(raw_dir / f"{src['name']}.jsonl"):
            if reject_reason(rec, src, since, until, min_words):
                stats["filtered"] += 1
                continue
            h = content_hash(rec["title"], rec["text"])
            if h in seen_hash:
                stats["duplicate"] += 1
                continue
            seen_hash.add(h)
            hits = find_leaks(rec["title"], rec["text"], rules)
            rid = item_id(rec["url"])
            mode = sourcing_mode(rec, src)
            provenance = dict(rec.get("provenance") or {})
            if rec.get("collected_via") and "collected_via" not in provenance:
                provenance["collected_via"] = rec["collected_via"]
            row = {
                "id": rid,
                "title": rec["title"],
                "text": rec["text"],
                "has_text": not rec.get("headline_only", False),
                "original_title": rec.get("original_title"),
                # Record-level values win for sources whose items differ
                # (fact-check verdicts, counterfactuals of articles in several languages).
                "language": rec.get("language") or src["language"],
                "label": rec.get("label") or src["label"],
                "subset": rec.get("subset") or src["subset"],
                "source": src["name"],
                "source_name": src["display_name"],
                "url": rec["url"],
                "published_at": rec["published_at"],
                "modified_at": rec.get("modified_at"),
                "authors": rec.get("authors") or [],
                "section": rec.get("section"),
                "keywords": rec.get("keywords") or [],
                "fetched_at": rec["fetched_at"],
                "extraction_method": rec["extraction_method"],
                "sourcing_mode": mode,
                "provenance": provenance,
                "word_count": rec["word_count"],
                "content_sha256": h,
                "leak_flags": hits,
                # Items can also be held out by their own checks (e.g. counterfactual validation).
                "eval_eligible": not is_blocking(hits) and rec.get("eligible", True),
                "selection_method": rec.get("selection_method") or f"source:{src['name']}",
                "implausibility_score": None,
            }
            if rid in selected and row["label"] == "real":
                row["subset"] = "real_implausible"
                row["selection_method"] = selected[rid].get("selection_method", "llm_implausibility+human_review")
                row["implausibility_score"] = selected[rid].get("score")
            rows.append(row)
            stats[f"{row['subset']}/{row['source']}/{'eligible' if row['eval_eligible'] else 'not_eligible'}"] += 1
            stats[f"sourcing_mode/{mode}"] += 1

    rows.sort(key=lambda r: (r["published_at"], r["id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in DATASET_COLUMNS}, ensure_ascii=False) + "\n")
    stats["total"] = len(rows)
    stats["eval_eligible"] = sum(r["eval_eligible"] for r in rows)
    summary = {"since": since.isoformat(), "until": until.isoformat() if until else None, "counts": dict(sorted(stats.items()))}
    out_path.with_suffix(".stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
