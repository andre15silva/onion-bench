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
    names = a.source or [s["name"] for s in sources if s["name"] not in ("inimigo_publico", "r_nottheonion") and s.get("kind") != "counterfactual"]
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
            if s.get("kind") == "counterfactual":
                continue  # generated, not scraped: nothing to re-extract
            print(s["name"], json.dumps(reextract(s, RAW, CACHE, since=_dt(DEFAULT_SINCE))))


def cmd_import_html(a) -> None:
    from .scrape import get_source, import_html

    files = [p for x in a.paths for p in (sorted(Path(x).glob("*.htm*")) if Path(x).is_dir() else [Path(x)])]
    print(f"imported {import_html(get_source(a.source), files, RAW)} pages into {RAW / (a.source + '.jsonl')}")


def cmd_import_index(a) -> None:
    from .scrape import get_source, import_index

    src = get_source(a.source)
    n, urls = import_index(src, Path(a.page), RAW, a.url_pattern or src["index_url_pattern"])
    print(f"imported {n} headline-only items from {len(urls)} article links into {RAW / (a.source + '.jsonl')}")
    if a.urls_out:
        Path(a.urls_out).write_text("\n".join(urls) + "\n")
        print(f"article URLs written to {a.urls_out} (save these pages and run import-html for full texts)")


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
    if getattr(a, "prompt_variant", None):
        cfg.prompt_variant = a.prompt_variant
    if a.samples:
        cfg.n_samples = a.samples
    items = load_items(Path(a.dataset), include_leak_flagged=cfg.include_leak_flagged, prompt_variant=cfg.prompt_variant)
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


def cmd_compare(a) -> None:
    from .report import compare_runs

    md = compare_runs([Path(d) for d in a.runs])
    if a.out:
        Path(a.out).write_text(md)
    print(md)


def cmd_make_counterfactuals(a) -> None:
    from .counterfactual import generate, select_bases
    from .runner import load_items

    items = load_items(Path(a.dataset))
    bases = select_bases(items, {"en": a.en, "pt": a.pt}, seed=a.seed)
    print(f"{len(bases)} base articles; generating with {a.model} -> {RAW / 'counterfactual.jsonl'}")
    if a.dry_run:
        return
    print(json.dumps(asyncio.run(generate(bases, RAW / "counterfactual.jsonl", model=a.model, effort=a.effort, concurrency=a.concurrency))))
    print("re-run `onion-bench assemble`; optionally `onion-bench review-counterfactuals` first")


def cmd_review_counterfactuals(a) -> None:
    from .counterfactual import review

    review(RAW / "counterfactual.jsonl")


def cmd_figures(a) -> None:
    from .figures import make_figures, summarize

    out = Path(a.out)
    summary = summarize(Path(a.dataset), Path(a.full), Path(a.headline_only))
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for p in make_figures(summary, out / "figures"):
        print("wrote", p)
    print("wrote", out / "results" / "summary.json")


def cmd_show_prompt(a) -> None:
    from .prompts import system_prompt, user_message
    from .runner import load_items

    items = load_items(Path(a.dataset), include_leak_flagged=True)
    it = next(i for i in items if i["id"] == a.id) if a.id else items[0]
    print("=== SYSTEM ===\n" + system_prompt(a.prompt_variant) + "\n\n=== USER ===\n" + user_message(it, a.prompt_variant))


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

    s = sub.add_parser("import-index", help="import headlines from a saved section/index page (headline_only items)")
    s.add_argument("source")
    s.add_argument("page")
    s.add_argument("--url-pattern", help="regex for article links (default: the source's index_url_pattern)")
    s.add_argument("--urls-out", help="also write the article URLs to this file")
    s.set_defaults(fn=cmd_import_index)

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
        s.add_argument("--prompt-variant", choices=["full", "headline_lead", "headline_only"],
                       help="what the model sees (default from config: full)")
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

    s = sub.add_parser("compare", help="side-by-side metrics of several report directories")
    s.add_argument("runs", nargs="+", help="directories containing metrics.json")
    s.add_argument("--out", help="write the markdown table here")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("make-counterfactuals", help="generate synthetic counterfactual fakes from real control articles (claude-cli)")
    s.add_argument("--dataset", default=str(DATASET))
    s.add_argument("--en", type=int, default=60, help="number of English base articles")
    s.add_argument("--pt", type=int, default=60, help="number of Portuguese base articles")
    s.add_argument("--model", default="claude-haiku-5-5", help="generator; must not be an evaluated model")
    s.add_argument("--effort", default="medium")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_make_counterfactuals)

    s = sub.add_parser("review-counterfactuals", help="human y/n review of generated counterfactuals")
    s.set_defaults(fn=cmd_review_counterfactuals)

    s = sub.add_parser("figures", help="README figures (light/dark SVG) and results/summary.json from two report dirs")
    s.add_argument("--dataset", default=str(DATASET))
    s.add_argument("--full", default=str(ROOT / "results" / "v3_full"), help="report dir of the full-text run")
    s.add_argument("--headline-only", default=str(ROOT / "results" / "v3_headline_only"), help="report dir of the headline-only run")
    s.add_argument("--out", default=str(ROOT / "docs"))
    s.set_defaults(fn=cmd_figures)

    s = sub.add_parser("show-prompt", help="print the exact prompt for an item")
    s.add_argument("--dataset", default=str(DATASET))
    s.add_argument("--id")
    s.add_argument("--prompt-variant", choices=["full", "headline_lead", "headline_only"], default="full")
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
