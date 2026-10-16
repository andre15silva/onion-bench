"""onion-bench command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache" / "html"
DATASET = ROOT / "data" / "dataset" / "onion_bench.jsonl"
IMPLAUSIBLE = ROOT / "data" / "raw" / "implausible_candidates.jsonl"
EVAL_CFG = ROOT / "config" / "eval.yaml"
DEFAULT_SINCE = "2026-07-01"


def _dt(s: str | None) -> datetime | None:
    if not s:
        return None
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def cmd_scrape(a) -> None:
    from .fetch import DEFAULT_USER_AGENT, Fetcher
    from .scrape import load_sources, scrape_source

    sources = load_sources()
    names = a.source or [s["name"] for s in sources if s["name"] not in ("inimigo_publico", "r_nottheonion")]
    fetcher = Fetcher(
        user_agent=a.user_agent or DEFAULT_USER_AGENT,
        robots_agents=tuple(a.robots_agent or ["onion-bench"]),
        min_delay_s=a.delay,
        cache_dir=CACHE,
    )
    for s in sources:
        if s["name"] in names:
            stats = scrape_source(s, fetcher, since=_dt(a.since), until=_dt(a.until), raw_dir=RAW, max_items=a.max_items, min_words=a.min_words, seed=a.seed)
            print(s["name"], json.dumps(stats))


def cmd_reextract(a) -> None:
    from .scrape import load_sources, reextract

    for s in load_sources():
        if not a.source or s["name"] in a.source:
            print(s["name"], json.dumps(reextract(s, RAW, CACHE)))


def cmd_import_html(a) -> None:
    from .scrape import get_source, import_html

    files = [p for x in a.paths for p in (sorted(Path(x).glob("*.htm*")) if Path(x).is_dir() else [Path(x)])]
    print(f"imported {import_html(get_source(a.source), files, RAW)} pages into {RAW / (a.source + '.jsonl')}")


def cmd_assemble(a) -> None:
    from .assemble import assemble

    summary = assemble(RAW, Path(a.out), since=_dt(a.since), until=_dt(a.until), min_words=a.min_words, implausible_file=IMPLAUSIBLE)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_score_implausible(a) -> None:
    from .implausible import score_items
    from .runner import load_items

    items = [it for it in load_items(Path(a.dataset)) if it["subset"] == "real_control"]
    n = asyncio.run(score_items(items, IMPLAUSIBLE, model=a.model, concurrency=a.concurrency))
    print(f"scored {n} new items -> {IMPLAUSIBLE}")


def cmd_review_implausible(a) -> None:
    from .implausible import auto_approve_top, review

    if a.auto_approve_top:
        print(f"auto-approved {auto_approve_top(IMPLAUSIBLE, a.auto_approve_top, min_score=a.min_score)} items")
    else:
        review(IMPLAUSIBLE, min_score=a.min_score)
    print("now re-run `onion-bench assemble` to move approved items into real_implausible")


def _cfg_items(a):
    from .runner import EvalConfig, balance_items, load_items, stratified_sample

    cfg = EvalConfig.load(Path(a.config))
    if a.samples:
        cfg.n_samples = a.samples
    items = load_items(Path(a.dataset), include_leak_flagged=cfg.include_leak_flagged)
    if cfg.balance_per_language:
        items = balance_items(items, seed=cfg.seed)
    if a.sample:
        items = stratified_sample(items, a.sample, seed=cfg.seed)
    if a.limit:
        items = items[: a.limit]
    return cfg, items


def cmd_run(a) -> None:
    from .runner import plan_jobs, run_async

    cfg, items = _cfg_items(a)
    out = Path(a.out) / "responses.jsonl"
    jobs = plan_jobs(cfg, items, out, models=a.model)
    print(f"{len(items)} items, {len(jobs)} calls to make -> {out}")
    if a.dry_run or not jobs:
        return
    if a.backend == "claude-cli":
        from .cli_backend import run_cli

        counts = asyncio.run(run_cli(cfg, jobs, out, concurrency=a.concurrency or 4, pause_at_utilization=a.pause_at))
    else:
        if a.concurrency:
            cfg.concurrency = a.concurrency
        counts = asyncio.run(run_async(cfg, jobs, out))
    print(json.dumps(counts))


def cmd_submit_batch(a) -> None:
    from .runner import plan_jobs, submit_batches

    cfg, items = _cfg_items(a)
    out = Path(a.out)
    jobs = plan_jobs(cfg, items, out / "responses.jsonl", models=a.model)
    print(f"submitting {len(jobs)} requests")
    if jobs and not a.dry_run:
        print(submit_batches(cfg, jobs, out / "batches.json"))


def cmd_collect_batch(a) -> None:
    from .runner import collect_batches

    cfg, items = _cfg_items(a)
    out = Path(a.out)
    print(json.dumps(collect_batches(cfg, items, out / "batches.json", out / "responses.jsonl", wait=not a.no_wait)))


def cmd_report(a) -> None:
    from .report import build_report

    cfg, items = _cfg_items(a)
    out = Path(a.out)
    build_report(cfg, items, out / "responses.jsonl", out, n_boot=a.n_boot)
    print((out / "report.md").read_text())


def cmd_probe(a) -> None:
    from .diagnostics import run_probe, style_swap

    cfg, items = _cfg_items(a)
    out = Path(a.out) / "diagnostics"
    models = [m for m in cfg.models if not a.model or m.id in a.model]
    if a.kind == "style-swap":
        from .cli_backend import run_cli
        from .runner import plan_jobs

        cfg.n_samples = 1
        swapped = style_swap(items)
        resp = out / "style_swap" / "responses.jsonl"
        print(json.dumps(asyncio.run(run_cli(cfg, plan_jobs(cfg, swapped, resp, models=a.model), resp, concurrency=a.concurrency or 4))))
    else:
        asyncio.run(run_probe(a.kind, items, models, out / f"{a.kind}.jsonl", concurrency=a.concurrency or 4))
    print(f"wrote {out}")


def cmd_show_prompt(a) -> None:
    from .prompts import SYSTEM_PROMPT, user_message
    from .runner import load_items

    items = load_items(Path(a.dataset), include_leak_flagged=True)
    it = next(i for i in items if i["id"] == a.id) if a.id else items[0]
    print("=== SYSTEM ===\n" + SYSTEM_PROMPT + "\n\n=== USER ===\n" + user_message(it))


def cmd_push_hub(a) -> None:
    from .hub import push

    print(push(Path(a.dataset), a.repo_id, private=not a.public, text_mode=a.text_mode, since=a.since))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="onion-bench")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("scrape", help="collect raw articles (respects robots.txt, stops on bot walls)")
    s.add_argument("--source", action="append", help="source name (repeatable); default: all automatic sources")
    s.add_argument("--since", default=DEFAULT_SINCE)
    s.add_argument("--until")
    s.add_argument("--max-items", type=int, help="override per-source cap")
    s.add_argument("--min-words", type=int, default=40)
    s.add_argument("--delay", type=float, default=1.5, help="seconds between requests to one host")
    s.add_argument("--user-agent")
    s.add_argument("--robots-agent", action="append", help="robots.txt agent token(s) that must all be allowed")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_scrape)

    s = sub.add_parser("reextract", help="re-run extraction on cached HTML (no network)")
    s.add_argument("--source", action="append")
    s.set_defaults(fn=cmd_reextract)

    s = sub.add_parser("import-html", help="import article pages saved by hand for a source")
    s.add_argument("source")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_import_html)

    s = sub.add_parser("assemble", help="build data/dataset/onion_bench.jsonl from raw files")
    s.add_argument("--since", default=DEFAULT_SINCE)
    s.add_argument("--until")
    s.add_argument("--min-words", type=int, default=40)
    s.add_argument("--out", default=str(DATASET))
    s.set_defaults(fn=cmd_assemble)

    s = sub.add_parser("score-implausible", help="LLM-score real control items for how fake they sound")
    s.add_argument("--dataset", default=str(DATASET))
    s.add_argument("--model", default="claude-haiku-5-5", help="scorer; must not be an evaluated model")
    s.add_argument("--concurrency", type=int, default=8)
    s.set_defaults(fn=cmd_score_implausible)

    s = sub.add_parser("review-implausible", help="human review of scored candidates")
    s.add_argument("--min-score", type=int, default=6)
    s.add_argument("--auto-approve-top", type=int, help="skip review and approve the top K (recorded as such)")
    s.set_defaults(fn=cmd_review_implausible)

    for name, fn, hlp in (
        ("run", cmd_run, "run the eval with concurrent API calls"),
        ("submit-batch", cmd_submit_batch, "submit the eval via the Batches API (50%% cheaper)"),
        ("collect-batch", cmd_collect_batch, "collect Batches API results"),
        ("report", cmd_report, "compute metrics and write report.md"),
    ):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--config", default=str(EVAL_CFG))
        s.add_argument("--dataset", default=str(DATASET))
        s.add_argument("--out", default=str(ROOT / "results" / "main"))
        s.add_argument("--model", action="append", help="restrict to model id(s)")
        s.add_argument("--samples", type=int, help="override n_samples")
        s.add_argument("--limit", type=int, help="only the first N items (smoke tests)")
        s.add_argument("--sample", type=int, help="stratified subset of ~N items (same language/label mix)")
        s.add_argument("--backend", choices=["api", "claude-cli"], default="api",
                       help="api: Anthropic API key; claude-cli: `claude -p` on a Pro/Max subscription")
        s.add_argument("--concurrency", type=int)
        s.add_argument("--pause-at", type=float, default=0.9,
                       help="claude-cli: pause when the plan's usage window reaches this utilization")
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--no-wait", action="store_true")
        s.add_argument("--n-boot", type=int, default=1000)
        s.set_defaults(fn=fn)

    s = sub.add_parser("probe", help="diagnostics via claude-cli: self-reported cues, memorization, style swap")
    s.add_argument("kind", choices=["cues", "memorization", "style-swap"])
    s.add_argument("--config", default=str(EVAL_CFG))
    s.add_argument("--dataset", default=str(DATASET))
    s.add_argument("--out", default=str(ROOT / "results" / "main"))
    s.add_argument("--model", action="append")
    s.add_argument("--samples", type=int)
    s.add_argument("--sample", type=int, default=40)
    s.add_argument("--limit", type=int)
    s.add_argument("--concurrency", type=int)
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("show-prompt", help="print the exact prompt for an item")
    s.add_argument("--dataset", default=str(DATASET))
    s.add_argument("--id")
    s.set_defaults(fn=cmd_show_prompt)

    s = sub.add_parser("push-hub", help="upload the dataset to the Hugging Face Hub")
    s.add_argument("repo_id")
    s.add_argument("--dataset", default=str(DATASET))
    s.add_argument("--public", action="store_true", help="default is a private repo")
    s.add_argument("--text-mode", choices=["full", "metadata"], default="full")
    s.add_argument("--since", default=DEFAULT_SINCE)
    s.set_defaults(fn=cmd_push_hub)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
