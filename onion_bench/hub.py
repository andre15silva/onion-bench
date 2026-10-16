"""Publish the dataset to the Hugging Face Hub (private by default).

The article texts are copyrighted by their publishers. Unless you have
permission to redistribute them, publish with --text-mode metadata (URLs,
labels, dates, hashes; texts re-scraped by users with `onion-bench scrape`)
or keep the repo private/gated.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

CARD = """\
---
language: [en, pt]
task_categories: [text-classification]
tags: [satire, fake-news, misinformation, llm-evaluation, knowledge-cutoff]
pretty_name: onion-bench
configs:
  - config_name: default
    data_files: onion_bench.jsonl
---

# onion-bench

Fake (satire, fact-checked misinformation, synthetic counterfactuals) and real news
published **after {since}**, i.e. after the training cutoff of the evaluated models,
for testing whether LLMs can tell fake from real news without web access.

## Fields

Model-facing (the only fields ever placed in a prompt): `title`, `text`.
Text is verbatim from the publisher page (whitespace-normalised, page chrome removed),
except for `synthetic_counterfactual` items (see below). `has_text` is false for
headline-only items (fact-check claims, headlines from index pages), which are used
only with the headline-only prompt variant.

Labels: `label` (`fake` | `real`) and `subset`:

| subset | label | what it is |
|---|---|---|
| `satire` | fake | satirical outlets |
| `misinformation` | fake | viral claims rated false by a fact-checker |
| `counterfactual` | fake | real article with one plausible factual change made by an LLM |
| `real_control` | real | ordinary reporting from mainstream outlets |
| `real_implausible` | real | real stories that sound fake (curated or scored + reviewed) |
| `fact_checked_true` | real | viral claims rated true by a fact-checker |

**Sourcing** (`sourcing_mode`, with details in `provenance`):

{modes}

Traceability: `source`, `source_name`, `url`, `published_at`, `modified_at`,
`authors`, `section`, `keywords`, `fetched_at`, `extraction_method`,
`original_title` (headline before outlet branding was stripped), `content_sha256`,
`selection_method`, `implausibility_score`, `provenance`. For fact-check claims,
`provenance` holds the fact-checker, verdict, review URL and dates. For
counterfactuals, it holds the base article's id and URL, the generator model and
version, the exact find/replace edits, the change summary, the automatic checks and
`human_verified`.

Leakage control: `leak_flags` lists outlet names/domains (and satire
disclaimers) found in `title`/`text`; `eval_eligible` is false when any were found
or when an item failed its own checks.

## Counts

{counts}

## Text mode

{text_mode_note}

## Licensing

Articles remain the copyright of their publishers ({publishers}). This dataset
is intended for non-commercial research use. `counterfactual` items are deliberately
false versions of real reports, made for this benchmark. They are labelled as such,
avoid attributing wrongdoing to named people, and must not be republished as news.
"""


def push(dataset: Path, repo_id: str, *, private: bool = True, text_mode: str = "full", since: str = "") -> str:
    from huggingface_hub import HfApi

    rows = [json.loads(l) for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if text_mode == "metadata":
        for r in rows:
            r["title"] = None
            r["text"] = None
        note = ("Texts are omitted (`title`/`text` are null). Re-create them from `url` with the onion-bench scraper and verify with "
                "`content_sha256`. For counterfactuals, re-scrape `provenance.base_url` and apply `provenance.replacements`.")
    else:
        note = "Full texts included."
    from .assemble import SOURCING_MODES

    modes = "\n".join(f"- `{k}`: {v}" for k, v in SOURCING_MODES.items())
    counts = Counter((r["subset"], r["language"], r["source"]) for r in rows)
    table = "| subset | language | source | items |\n|---|---|---|---|\n" + "\n".join(
        f"| {s} | {l} | {src} | {n} |" for (s, l, src), n in sorted(counts.items())
    )
    publishers = ", ".join(sorted({r["source_name"] for r in rows}))
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "onion_bench.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (d / "README.md").write_text(CARD.format(since=since, counts=table, text_mode_note=note, publishers=publishers, modes=modes))
        api.upload_folder(folder_path=str(d), repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"
