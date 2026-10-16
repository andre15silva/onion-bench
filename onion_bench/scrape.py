"""Collect raw articles for one source into data/raw/<source>.jsonl (resumable)."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

from . import discover
from .extract import ExtractConfig, extract
from .fetch import BlockedError, Fetcher, FetchError, RobotsDisallowed

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = ROOT / "config" / "sources.yaml"


def load_sources(path: Path = SOURCES_FILE) -> list[dict]:
    return yaml.safe_load(path.read_text())["sources"]


def get_source(name: str, sources: list[dict] | None = None) -> dict:
    for s in sources or load_sources():
        if s["name"] == name:
            return s
    raise KeyError(f"unknown source {name!r}")


def canonical_url(url: str) -> str:
    p = urlsplit(url.strip())
    return urlunsplit((p.scheme, p.netloc.lower(), p.path, "", ""))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _url_ok(url: str, src: dict) -> bool:
    inc = src.get("url_include") or []
    exc = src.get("url_exclude") or []
    if inc and not any(re.search(p, url) for p in inc):
        return False
    return not any(re.search(p, url) for p in exc)


def stratified_order(cands: list[discover.Candidate], seed: int) -> list[discover.Candidate]:
    """Round-robin over months (random within month) so that a capped crawl
    spreads evenly over the collection window instead of taking only the newest."""
    rng = random.Random(seed)
    buckets: dict[str, list] = defaultdict(list)
    for c in cands:
        buckets[c.date_hint.strftime("%Y-%m") if c.date_hint else "unknown"].append(c)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    out = []
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop())
    return out


def discover_candidates(fetcher: Fetcher, src: dict, since: datetime) -> list[discover.Candidate]:
    seen: dict[str, discover.Candidate] = {}
    for d in src.get("discovery", []):
        try:
            if d["type"] == "rss":
                found = discover.from_rss(fetcher, d["url"])
            elif d["type"] == "listing":
                found = discover.from_listing(fetcher, d["url"], d["link_pattern"], d.get("max_pages", 10))
            elif d["type"] == "reddit":
                found = discover.from_reddit(d["subreddit"], since=since, user_agent=fetcher.user_agent, t=d.get("t", "year"))
            elif d["type"] == "sitemap":
                found = discover.from_sitemap(
                    fetcher, d["url"], since=since,
                    max_child_sitemaps=d.get("max_child_sitemaps"),
                    child_include=d.get("child_include"),
                )
            else:
                raise ValueError(f"unknown discovery type {d['type']}")
        except (BlockedError, RobotsDisallowed) as e:
            log.error("[%s] discovery via %s stopped: %s", src["name"], d["url"], e)
            continue
        except FetchError as e:
            log.warning("[%s] discovery via %s failed: %s", src["name"], d["url"], e)
            continue
        log.info("[%s] %s -> %d urls", src["name"], d["url"], len(found))
        for c in found:
            u = canonical_url(c.url)
            if not _url_ok(u, src):
                continue
            if c.date_hint is not None and c.date_hint < since:
                continue
            if u not in seen or (seen[u].date_hint is None and c.date_hint is not None):
                seen[u] = discover.Candidate(u, c.date_hint)
    return list(seen.values())


def build_record(src: dict, url: str, html: str, cfg: ExtractConfig) -> dict:
    ex = extract(html, cfg)
    text = ex.text
    return {
        "source": src["name"],
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": ex.title,
        "text": text,
        "published_at": ex.published_at.isoformat() if ex.published_at else None,
        "modified_at": ex.modified_at.isoformat() if ex.modified_at else None,
        "authors": ex.authors,
        "section": ex.section,
        "keywords": ex.keywords,
        "page_language": ex.language,
        "site_name": ex.site_name,
        "extraction_method": ex.method,
        "word_count": len(text.split()),
        "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
    }


def reject_reason(rec: dict, src: dict, since: datetime, until: datetime | None, min_words: int) -> str | None:
    if not rec["title"]:
        return "no_title"
    if rec["word_count"] < min_words:
        return f"too_short({rec['word_count']})"
    if not rec["published_at"]:
        return "no_date"
    pub = datetime.fromisoformat(rec["published_at"])
    if pub < since:
        return "before_since"
    if until is not None and pub > until:
        return "after_until"
    sec = (rec.get("section") or "").lower()
    if any(x.lower() in sec for x in src.get("exclude_sections", [])):
        return f"section({rec['section']})"
    return None


def scrape_source(
    src: dict,
    fetcher: Fetcher,
    *,
    since: datetime,
    until: datetime | None = None,
    raw_dir: Path,
    max_items: int | None = None,
    min_words: int = 40,
    seed: int = 0,
) -> dict:
    out_path = raw_dir / f"{src['name']}.jsonl"
    rej_path = raw_dir / f"{src['name']}.rejected.jsonl"
    kept = _read_jsonl(out_path)
    done = {r["url"] for r in kept} | {r["url"] for r in _read_jsonl(rej_path)}
    n_kept = sum(1 for r in kept if reject_reason(r, src, since, until, min_words) is None)
    cap = max_items if max_items is not None else src.get("max_items", 200)
    stats = defaultdict(int)
    stats["kept_before"] = n_kept

    cands = discover_candidates(fetcher, src, since)
    stats["candidates"] = len(cands)
    cfg = ExtractConfig(**(src.get("extract") or {}))
    for c in stratified_order(cands, seed):
        if n_kept >= cap:
            break
        if c.url in done:
            continue
        try:
            html = fetcher.get(c.url)
        except (BlockedError, RobotsDisallowed) as e:
            log.error("[%s] stopping: %s", src["name"], e)
            stats["stopped_blocked"] += 1
            break
        except FetchError as e:
            log.warning("[%s] %s", src["name"], e)
            _append_jsonl(rej_path, {"url": c.url, "reason": f"fetch:{e}"})
            stats["fetch_error"] += 1
            continue
        rec = build_record(src, c.url, html, cfg)
        reason = reject_reason(rec, src, since, until, min_words)
        if reason:
            _append_jsonl(rej_path, {"url": c.url, "reason": reason, "title": rec["title"], "published_at": rec["published_at"]})
            stats[f"rejected:{reason.split('(')[0]}"] += 1
            continue
        _append_jsonl(out_path, rec)
        n_kept += 1
        stats["kept_new"] += 1
        if n_kept % 25 == 0:
            log.info("[%s] %d/%d kept", src["name"], n_kept, cap)
    stats["kept_total"] = n_kept
    return dict(stats)


def import_html(src: dict, files: list[Path], raw_dir: Path) -> int:
    """Import article pages saved by hand (e.g. from a browser) for sources that
    block automated collection. The canonical URL is read from the page."""
    from bs4 import BeautifulSoup

    out_path = raw_dir / f"{src['name']}.jsonl"
    done = {r["url"] for r in _read_jsonl(out_path)}
    cfg = ExtractConfig(**(src.get("extract") or {}))
    n = 0
    for f in files:
        html = f.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        link = soup.find("link", rel="canonical")
        og = soup.find("meta", attrs={"property": "og:url"})
        url = (link and link.get("href")) or (og and og.get("content")) or f"file://{f.name}"
        url = canonical_url(url)
        if url in done:
            continue
        rec = build_record(src, url, html, cfg)
        rec["collected_via"] = "manual_html_import"
        _append_jsonl(out_path, rec)
        done.add(url)
        n += 1
    return n


def reextract(src: dict, raw_dir: Path, cache_dir: Path) -> dict:
    """Re-run extraction on cached HTML (after changing selectors); no network."""
    path = raw_dir / f"{src['name']}.jsonl"
    cfg = ExtractConfig(**(src.get("extract") or {}))
    out, stats = [], defaultdict(int)
    for rec in _read_jsonl(path):
        cache = cache_dir / (hashlib.sha256(rec["url"].encode()).hexdigest() + ".html")
        if not cache.exists():
            out.append(rec)
            stats["no_cache"] += 1
            continue
        new = build_record(src, rec["url"], cache.read_text(encoding="utf-8"), cfg)
        new["fetched_at"] = rec["fetched_at"]
        if "collected_via" in rec:
            new["collected_via"] = rec["collected_via"]
        stats["changed" if new["text"] != rec["text"] or new["title"] != rec["title"] else "unchanged"] += 1
        out.append(new)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    return dict(stats)
