"""Find real articles that *sound* fake ("real_implausible" subset).

Two complementary routes:
1. Curated sources: outlets' weird-news sections and r/nottheonion (see
   config/sources.yaml); their items already land in `real_implausible`.
2. This module: score every real control article for how unbelievable it
   sounds with a scorer model that is NOT one of the evaluated models, then
   have a human confirm the top candidates. Using an evaluated model (or its
   eval responses) to pick these items would bias the subset toward exactly
   the items that model gets wrong, so the scorer is kept separate and its
   picks still require human approval (or an explicit --auto-approve-top).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anthropic

from .prompts import user_message

SCORER_MODEL = "claude-haiku-5-5"

SCORER_SYSTEM = """\
You help build a dataset of real news stories that sound unbelievable. You will see one article. Assume it is true; do not judge whether it is real. Rate how surprising, absurd or hard to believe the story would sound to an ordinary newspaper reader encountering it with no context, from 0 (completely mundane) to 10 (sounds like a joke or satire)."""

SCORER_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}


async def score_items(items: list[dict], out_path: Path, model: str = SCORER_MODEL, concurrency: int = 8) -> int:
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    client = anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(concurrency)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("a", encoding="utf-8")

    async def one(it: dict) -> None:
        async with sem:
            try:
                msg = await client.messages.create(
                    model=model,
                    max_tokens=4000,
                    system=SCORER_SYSTEM,
                    messages=[{"role": "user", "content": user_message(it)}],
                    output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCORER_SCHEMA}},
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                obj = json.loads(text) if msg.stop_reason != "refusal" else {"score": None, "reason": "refusal"}
            except (anthropic.APIError, json.JSONDecodeError) as e:
                obj = {"score": None, "reason": f"error: {e}"}
            f.write(json.dumps({
                "id": it["id"], "source": it["source"], "title": it["title"], "url": it["url"],
                "score": obj.get("score"), "reason": obj.get("reason"), "scorer_model": model,
                "approved": None,
            }, ensure_ascii=False) + "\n")
            f.flush()

    try:
        await asyncio.gather(*(one(it) for it in todo))
    finally:
        f.close()
    return len(todo)


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def auto_approve_top(path: Path, k: int, per_language: dict[str, str] | None = None, min_score: int = 7) -> int:
    rows = _load(path)
    ranked = sorted((r for r in rows if isinstance(r.get("score"), int) and r["score"] >= min_score), key=lambda r: -r["score"])
    chosen = {r["id"] for r in ranked[:k]}
    for r in rows:
        if r["id"] in chosen and r["approved"] is None:
            r["approved"] = True
            r["selection_method"] = f"llm_implausibility({r['scorer_model']}, auto_top{k})"
    _save(path, rows)
    return len(chosen)


def review(path: Path, min_score: int = 6) -> None:
    """Interactive y/n review of the highest-scored unreviewed candidates."""
    rows = _load(path)
    queue = sorted(
        (r for r in rows if r["approved"] is None and isinstance(r.get("score"), int) and r["score"] >= min_score),
        key=lambda r: -r["score"],
    )
    try:
        for r in queue:
            print(f"\n[{r['score']}/10] {r['title']}\n  {r['url']}\n  scorer: {r['reason']}")
            ans = input("Sounds fake but is real? [y/n/q] ").strip().lower()
            if ans == "q":
                break
            if ans in ("y", "n"):
                r["approved"] = ans == "y"
                r["selection_method"] = f"llm_implausibility({r['scorer_model']})+human_review"
    finally:
        _save(path, rows)
