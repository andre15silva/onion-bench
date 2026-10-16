"""Diagnostic probes for shortcut use and contamination (claude-cli backend).

- cues: the eval task plus self-reported cues and whether the model recognises
  the article or the event from training. Asking for reasons can change
  behaviour, so these runs are kept apart from the scored eval.
- memorization: headline only; the model is asked for the article's opening
  sentence. High overlap with the real lead suggests the text (or an earlier
  version of it) was in training data despite the post-cutoff date.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
import shutil
import tempfile
from pathlib import Path

from .cli_backend import child_env, command
from .prompts import SYSTEM_PROMPT
from .runner import Job, ModelSpec

CUES_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["real", "fake"]},
        "p_fake": {"type": "integer"},
        "main_cues": {"type": "array", "items": {"type": "string"}},
        "recognized_from_training": {"type": "string", "enum": ["no", "this_article", "the_event_or_facts"]},
    },
    "required": ["verdict", "p_fake", "main_cues", "recognized_from_training"],
    "additionalProperties": False,
}
CUES_SYSTEM = SYSTEM_PROMPT + """

For this study we also record your reasoning. In "main_cues", list the (at most three) most decisive cues for your verdict, each in one short sentence. In "recognized_from_training", say whether you recognise this specific article ("this_article"), or the events/facts it reports ("the_event_or_facts"), from your training data, or neither ("no")."""

MEMO_SCHEMA = {
    "type": "object",
    "properties": {"knows_article": {"type": "boolean"}, "first_sentence": {"type": "string"}},
    "required": ["knows_article", "first_sentence"],
    "additionalProperties": False,
}
MEMO_SYSTEM = """You will be shown the headline of a published news article. If you have seen this exact article before, write its opening sentence as close to verbatim as you can; if you have not, write the most likely opening sentence. Set knows_article to true only if you recognise the specific article."""


async def _call(sem, claude_bin, env, job, system, schema, user) -> dict:
    async with sem:
        wd = tempfile.mkdtemp(prefix="session-")
        proc = await asyncio.create_subprocess_exec(
            *command(job, claude_bin, system=system, schema=schema, user=user), cwd=wd, env=env,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), 900)
        init = result = None
        for line in out.decode("utf-8", "replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "system" and e.get("subtype") == "init":
                init = e
            elif e.get("type") == "result":
                result = e
        ok = bool(init and init.get("tools") == ["StructuredOutput"] and not init.get("mcp_servers")
                  and init.get("apiKeySource") in ("none", None)
                  and result and set((result.get("modelUsage") or {}).keys()) == {job.model.id})
        return {"model": job.model.id, "item_id": job.item["id"], "setup_ok": ok,
                "output": (result or {}).get("structured_output"), "error": None if result else "no result"}


async def run_probe(kind: str, items: list[dict], models: list[ModelSpec], out_path: Path, concurrency: int = 4) -> None:
    claude_bin = shutil.which("claude")
    env = child_env()
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for m in models:
        for it in items:
            job = Job(m, it, 0)
            if kind == "cues":
                tasks.append(_call(sem, claude_bin, env, job, CUES_SYSTEM, CUES_SCHEMA, None))
            else:
                tasks.append(_call(sem, claude_bin, env, job, MEMO_SYSTEM, MEMO_SCHEMA, f"<headline>\n{it['title']}\n</headline>"))
    recs = await asyncio.gather(*tasks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


def lead_sentence(text: str) -> str:
    first = text.split("\n\n")[0]
    m = re.match(r"(.+?[.!?])(?:\s|$)", first)
    return m.group(1) if m else first


def overlap(a: str, b: str) -> dict:
    """Similarity of a generated lead to the real one: character ratio and the
    longest shared word run."""
    wa, wb = a.lower().split(), b.lower().split()
    sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    longest = max((blk.size for blk in sm.get_matching_blocks()), default=0)
    return {"ratio": round(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio(), 3), "longest_word_run": longest}


# ------------------------------------------------------------ style swap
_DATELINE = re.compile(r"^[A-Z][A-Z .,'’-]{2,40}—")
_SMALL = {"a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "by", "in", "of", "up", "as", "is", "it"}
_FAKE_DATELINE = {"bbc_news": "LONDON—", "cbs_news": "WASHINGTON—"}


def sentence_case(t: str) -> str:
    w = t.split(" ")
    return " ".join([w[0]] + [x if (x.isupper() and len(x) > 1) or not x[:1].isalpha() else x.lower() for x in w[1:]])


def title_case(t: str) -> str:
    return " ".join(x if i and x.lower() in _SMALL else x[:1].upper() + x[1:] for i, x in enumerate(t.split(" ")))


def style_swap(items: list[dict]) -> list[dict]:
    """Counterfactual copies: The Onion loses its house style (dateline,
    Title Case headline); BBC/CBS articles get it. Only for diagnostics; the
    dataset itself is never edited."""
    out = []
    for it in items:
        it = dict(it)
        if it["source"] == "the_onion":
            it["title"], it["text"] = sentence_case(it["title"]), _DATELINE.sub("", it["text"], count=1)
        elif it["source"] in _FAKE_DATELINE:
            it["title"], it["text"] = title_case(it["title"]), _FAKE_DATELINE[it["source"]] + it["text"]
        else:
            continue
        out.append(it)
    return out
