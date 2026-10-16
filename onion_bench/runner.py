"""Query models: n independent samples per (model, item). Resumable.

Design choices that matter for validity:
- No `tools` are passed, so the model has no web search/fetch or any other
  way to reach the internet during the eval.
- No refusal `fallbacks`: a fallback would silently answer with a different
  model. Refusals are recorded as their own outcome instead.
- Sampling parameters (temperature etc.) are not settable on these models;
  the 5 samples use the default sampling of each model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import anthropic
import yaml

from .prompts import OUTPUT_SCHEMA, PROMPT_VERSION, SYSTEM_PROMPT, parse_answer, user_message

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    id: str
    knowledge_cutoff: date | None = None
    effort: str = "high"


@dataclass
class EvalConfig:
    models: list[ModelSpec]
    n_samples: int = 5
    max_tokens: int = 16000
    thinking_display: str = "summarized"
    include_leak_flagged: bool = False
    concurrency: int = 8

    @classmethod
    def load(cls, path: Path) -> "EvalConfig":
        raw = yaml.safe_load(path.read_text())
        models = [
            ModelSpec(
                id=m["id"],
                knowledge_cutoff=date.fromisoformat(str(m["knowledge_cutoff"])) if m.get("knowledge_cutoff") else None,
                effort=m.get("effort", "high"),
            )
            for m in raw.pop("models")
        ]
        return cls(models=models, **raw)


@dataclass(frozen=True)
class Job:
    model: ModelSpec
    item: dict = field(hash=False, compare=False)
    sample: int = 0

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.model.id, self.item["id"], self.sample)

    @property
    def custom_id(self) -> str:
        return f"{self.model.id}__{self.item['id']}__s{self.sample}"


def load_items(dataset: Path, include_leak_flagged: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if include_leak_flagged or r["eval_eligible"]]


def after_cutoff(item: dict, spec: ModelSpec) -> bool:
    if spec.knowledge_cutoff is None:
        return True
    return datetime.fromisoformat(item["published_at"]).date() > spec.knowledge_cutoff


def done_keys(responses: Path) -> set[tuple[str, str, int]]:
    if not responses.exists():
        return set()
    keys = set()
    for line in responses.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # Transient failures are retried on the next run; final outcomes are kept.
        if r["status"] != "api_error":
            keys.add((r["model"], r["item_id"], r["sample"]))
    return keys


def plan_jobs(cfg: EvalConfig, items: list[dict], responses: Path, models: list[str] | None = None) -> list[Job]:
    done = done_keys(responses)
    jobs = []
    for spec in cfg.models:
        if models and spec.id not in models:
            continue
        for it in items:
            if not after_cutoff(it, spec):
                continue
            for k in range(cfg.n_samples):
                j = Job(spec, it, k)
                if j.key not in done:
                    jobs.append(j)
    return jobs


def request_params(cfg: EvalConfig, job: Job) -> dict:
    return {
        "model": job.model.id,
        "max_tokens": cfg.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message(job.item)}],
        "thinking": {"type": "adaptive", "display": cfg.thinking_display},
        "output_config": {"effort": job.model.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }


def record_from_message(job: Job, msg, latency_s: float | None) -> dict:
    text = "".join(b.text for b in msg.content if b.type == "text")
    thinking = "\n".join(getattr(b, "thinking", "") or "" for b in msg.content if b.type == "thinking").strip()
    if msg.stop_reason == "refusal":
        status, verdict, p_fake = "refusal", None, None
    else:
        verdict, p_fake = parse_answer(text)
        status = "ok" if verdict else ("max_tokens" if msg.stop_reason == "max_tokens" else "parse_error")
    stop_details = getattr(msg, "stop_details", None)
    return {
        "model": job.model.id,
        "model_served": msg.model,
        "item_id": job.item["id"],
        "sample": job.sample,
        "prompt_version": PROMPT_VERSION,
        "effort": job.model.effort,
        "status": status,
        "verdict": verdict,
        "p_fake": p_fake,
        "raw_text": text,
        "thinking_summary": thinking or None,
        "stop_reason": msg.stop_reason,
        "refusal_category": getattr(stop_details, "category", None) if stop_details else None,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "request_id": getattr(msg, "_request_id", None),
        "latency_s": latency_s,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def error_record(job: Job, status: str, err: str) -> dict:
    return {
        "model": job.model.id, "item_id": job.item["id"], "sample": job.sample,
        "prompt_version": PROMPT_VERSION, "effort": job.model.effort,
        "status": status, "verdict": None, "p_fake": None, "error": err,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class _Writer:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


async def run_async(cfg: EvalConfig, jobs: list[Job], responses: Path, client: anthropic.AsyncAnthropic | None = None) -> dict:
    client = client or anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(cfg.concurrency)
    out = _Writer(responses)
    counts: dict[str, int] = {}

    async def one(job: Job) -> None:
        async with sem:
            t0 = time.monotonic()
            try:
                msg = await client.messages.create(**request_params(cfg, job))
                rec = record_from_message(job, msg, round(time.monotonic() - t0, 2))
            except anthropic.BadRequestError as e:
                # Deterministic for this input: record it so it is not retried forever.
                rec = error_record(job, "bad_request", str(e))
            except anthropic.APIError as e:
                rec = error_record(job, "api_error", f"{type(e).__name__}: {e}")
            out.write(rec)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            n = sum(counts.values())
            if n % 50 == 0 or n == len(jobs):
                log.info("%d/%d done %s", n, len(jobs), counts)

    try:
        await asyncio.gather(*(one(j) for j in jobs))
    finally:
        out.close()
    return counts


# ------------------------------------------------------------------ batches
def submit_batches(cfg: EvalConfig, jobs: list[Job], state_file: Path, client: anthropic.Anthropic | None = None, chunk: int = 10_000) -> list[str]:
    """Submit jobs through the Message Batches API (50% cheaper, async)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = client or anthropic.Anthropic()
    state = json.loads(state_file.read_text()) if state_file.exists() else {"batches": []}
    ids = []
    for i in range(0, len(jobs), chunk):
        part = jobs[i : i + chunk]
        batch = client.messages.batches.create(
            requests=[Request(custom_id=j.custom_id, params=MessageCreateParamsNonStreaming(**request_params(cfg, j))) for j in part]
        )
        ids.append(batch.id)
        state["batches"].append({"id": batch.id, "n": len(part), "collected": False})
        log.info("submitted batch %s (%d requests)", batch.id, len(part))
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))
    return ids


def collect_batches(cfg: EvalConfig, items: list[dict], state_file: Path, responses: Path, client: anthropic.Anthropic | None = None, wait: bool = True) -> dict:
    client = client or anthropic.Anthropic()
    state = json.loads(state_file.read_text())
    by_item = {it["id"]: it for it in items}
    by_model = {m.id: m for m in cfg.models}
    out = _Writer(responses)
    counts: dict[str, int] = {}
    try:
        for b in state["batches"]:
            if b["collected"]:
                continue
            while True:
                info = client.messages.batches.retrieve(b["id"])
                if info.processing_status == "ended" or not wait:
                    break
                log.info("batch %s: %s", b["id"], info.request_counts)
                time.sleep(60)
            if info.processing_status != "ended":
                continue
            for res in client.messages.batches.results(b["id"]):
                model_id, item_id, s = res.custom_id.split("__")
                job = Job(by_model[model_id], by_item[item_id], int(s[1:]))
                if res.result.type == "succeeded":
                    rec = record_from_message(job, res.result.message, None)
                elif res.result.type == "errored" and res.result.error.error.type == "invalid_request_error":
                    rec = error_record(job, "bad_request", str(res.result.error))
                else:
                    rec = error_record(job, "api_error", res.result.type)
                out.write(rec)
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            b["collected"] = True
            state_file.write_text(json.dumps(state, indent=2))
    finally:
        out.close()
    return counts
