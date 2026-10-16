"""Extract title/body/date from an article page without rewriting its text.

Text is taken verbatim from the publisher's HTML. The only normalisation is the
unavoidable HTML->text step: collapsing runs of whitespace inside a paragraph
and joining paragraphs with a blank line. No words are added, removed or
reordered, except that page chrome (ads, share buttons, newsletter boxes,
"related articles" lists, captions) is excluded via per-source selectors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bs4 import BeautifulSoup, Tag

from .discover import parse_date

_WS = re.compile(r"\s+")
_ARTICLE_TYPES = {
    "Article", "NewsArticle", "ReportageNewsArticle", "AnalysisNewsArticle",
    "BlogPosting", "SatiricalArticle", "OpinionNewsArticle", "BackgroundNewsArticle",
}


@dataclass
class ExtractConfig:
    title_selector: str | None = "h1"
    body_selector: str | None = None
    # Optional standfirst/lead paragraph that sits outside the body container.
    lead_selector: str | None = None
    paragraph_selector: str = "p"
    exclude_selectors: list[str] = field(default_factory=list)
    # Paragraphs that are exactly one of these strings (site boilerplate that
    # cannot be targeted with a CSS selector) are dropped.
    drop_paragraphs: list[str] = field(default_factory=list)
    # Same, by regex (for templated chrome such as "Follow Suffolk news on ...").
    drop_paragraph_patterns: list[str] = field(default_factory=list)
    min_words_selector: int = 15
    # Allow trafilatura when the selectors find no text. Off for sources with
    # reliable selectors: on pages without a body (headline-only posts) it
    # returns sidebars, e.g. lists of the outlet's other headlines.
    generic_fallback: bool = True
    # Fallback when the page has no machine-readable date: an element whose
    # `datetime` attribute or text holds the publication date.
    date_selector: str | None = None
    date_formats: list[str] = field(default_factory=lambda: ["%B %d, %Y", "%d %B %Y", "%Y-%m-%d"])


@dataclass
class Extracted:
    title: str | None
    paragraphs: list[str]
    published_at: datetime | None
    modified_at: datetime | None
    authors: list[str]
    section: str | None
    keywords: list[str]
    language: str | None
    site_name: str | None
    method: str

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _node_text(el: Tag) -> str:
    for br in el.find_all("br"):
        br.replace_with("\n")
    # No separator: inline tags (<a>, <em>) carry their own surrounding spaces.
    return _clean(el.get_text())


def _iter_ld(soup: BeautifulSoup):
    for sc in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(sc.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, list):
                stack.extend(x)
            elif isinstance(x, dict):
                if "@graph" in x:
                    stack.append(x["@graph"])
                yield x


def _types(x: dict) -> set[str]:
    t = x.get("@type")
    return set(t) if isinstance(t, list) else {t} if isinstance(t, str) else set()


def _names(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [v["name"]] if isinstance(v.get("name"), str) else []
    if isinstance(v, list):
        return [n for i in v for n in _names(i)]
    return []


def _meta(soup: BeautifulSoup, *keys: str) -> str | None:
    for k in keys:
        m = soup.find("meta", attrs={"property": k}) or soup.find("meta", attrs={"name": k})
        if m and m.get("content"):
            return m["content"].strip()
    return None


def _parse_text_date(text: str, formats: list[str]) -> datetime | None:
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def extract(html: str, cfg: ExtractConfig) -> Extracted:
    soup = BeautifulSoup(html, "lxml")
    ld = next((x for x in _iter_ld(soup) if _types(x) & _ARTICLE_TYPES), {})

    published = parse_date(ld.get("datePublished")) or parse_date(
        _meta(soup, "article:published_time", "og:article:published_time", "datePublished", "pubdate")
    )
    if published is None and cfg.date_selector and (d := soup.select_one(cfg.date_selector)):
        published = parse_date(d.get("datetime")) or _parse_text_date(_node_text(d), cfg.date_formats)
    modified = parse_date(ld.get("dateModified")) or parse_date(_meta(soup, "article:modified_time"))
    authors = _names(ld.get("author")) or ([a] if (a := _meta(soup, "author")) else [])
    section = ld.get("articleSection") or _meta(soup, "article:section")
    if isinstance(section, list):
        section = ", ".join(map(str, section))
    kw = ld.get("keywords") or _meta(soup, "keywords") or []
    keywords = [k.strip() for k in kw.split(",")] if isinstance(kw, str) else [str(k) for k in kw]
    html_tag = soup.find("html")
    language = ld.get("inLanguage") or (html_tag.get("lang") if html_tag else None)
    site_name = _meta(soup, "og:site_name")

    title = None
    if cfg.title_selector and (t := soup.select_one(cfg.title_selector)):
        title = _node_text(t) or None
    if not title:
        title = ld.get("headline") or _meta(soup, "og:title")

    paragraphs: list[str] = []
    method = "none"
    root = soup.select_one(cfg.body_selector) if cfg.body_selector else None
    if root is not None:
        for sel in cfg.exclude_selectors:
            for el in root.select(sel):
                el.decompose()
        drop = {_clean(d) for d in cfg.drop_paragraphs}
        paragraphs = [p for el in root.select(cfg.paragraph_selector) if (p := _node_text(el)) and p not in drop]
        if cfg.lead_selector and (lead := soup.select_one(cfg.lead_selector)) and (lt := _node_text(lead)):
            if lt not in paragraphs:
                paragraphs.insert(0, lt)
        method = "selector"
    if sum(len(p.split()) for p in paragraphs) < cfg.min_words_selector:
        body = ld.get("articleBody")
        if isinstance(body, str) and len(body.split()) >= cfg.min_words_selector:
            paragraphs = [_clean(p) for p in re.split(r"\n\s*\n|\r?\n", body) if _clean(p)]
            method = "jsonld"
        elif root is not None or not cfg.generic_fallback:
            # The page has the source's body container but (almost) no text,
            # e.g. a headline-only post. Generic extraction would pick up
            # sidebars and related-article lists, so keep what we have.
            pass
        else:
            import trafilatura

            txt = trafilatura.extract(html, favor_precision=True, include_comments=False, include_tables=False)
            paragraphs = [_clean(p) for p in (txt or "").split("\n") if _clean(p)]
            method = "trafilatura" if paragraphs else "none"

    if cfg.drop_paragraph_patterns:
        rxs = [re.compile(p) for p in cfg.drop_paragraph_patterns]
        paragraphs = [p for p in paragraphs if not any(rx.search(p) for rx in rxs)]

    return Extracted(
        title=_clean(title) if title else None,
        paragraphs=paragraphs,
        published_at=published,
        modified_at=modified,
        authors=authors,
        section=section,
        keywords=keywords,
        language=language,
        site_name=site_name,
        method=method,
    )
