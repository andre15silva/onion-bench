"""Find candidate article URLs from RSS feeds and XML sitemaps."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from lxml import etree

from .fetch import Fetcher, FetchError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    url: str
    # Date hint from the feed/sitemap (pubDate, lastmod, or date in the URL).
    # The authoritative publication date is re-read from the article page.
    date_hint: datetime | None


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


_URL_DATE = re.compile(r"/(20\d\d)/(\d\d)/(\d\d)/")


def date_from_url(url: str) -> datetime | None:
    m = _URL_DATE.search(url)
    if not m:
        return None
    try:
        return datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
    except ValueError:
        return None


def _xml(text: str) -> etree._Element:
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(text.strip().encode("utf-8"), parser=parser) if text.strip() else None
    if root is None:
        raise FetchError("response is not XML")
    return root


def _local(tag: object) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def from_rss(fetcher: Fetcher, url: str) -> list[Candidate]:
    root = _xml(fetcher.get(url, use_cache=False))
    out: list[Candidate] = []
    for item in root.iter():
        name = _local(item.tag)
        if name not in ("item", "entry"):
            continue
        link, date = None, None
        for child in item:
            cname = _local(child.tag)
            if cname == "link":
                link = (child.text or child.get("href") or "").strip() or link
            elif cname in ("pubDate", "published", "updated", "date") and date is None:
                date = parse_date(child.text)
        if link:
            out.append(Candidate(link, date or date_from_url(link)))
    return out


def from_sitemap(
    fetcher: Fetcher,
    url: str,
    *,
    since: datetime,
    max_child_sitemaps: int | None = None,
    child_include: str | None = None,
    _depth: int = 0,
) -> list[Candidate]:
    """Walk a sitemap (index), keeping URLs whose lastmod/URL date is >= since.

    Child sitemaps with a lastmod older than `since` are skipped. When children
    carry no lastmod, only the last `max_child_sitemaps` are visited (WordPress
    and most CMSs list the newest sitemap last).
    """
    try:
        root = _xml(fetcher.get(url, use_cache=False))
    except FetchError as e:
        log.warning("sitemap %s: %s", url, e)
        return []
    kind = _local(root.tag)
    out: list[Candidate] = []
    if kind == "sitemapindex":
        children: list[tuple[str, datetime | None]] = []
        for sm in root:
            loc = lastmod = None
            for c in sm:
                if _local(c.tag) == "loc":
                    loc = (c.text or "").strip()
                elif _local(c.tag) == "lastmod":
                    lastmod = parse_date(c.text)
            if not loc or (child_include and not re.search(child_include, loc)):
                continue
            if lastmod is not None and lastmod < since:
                continue
            children.append((loc, lastmod))
        if max_child_sitemaps is not None and not any(lm for _, lm in children):
            children = children[-max_child_sitemaps:]
        for loc, _ in children:
            out.extend(
                from_sitemap(
                    fetcher, loc, since=since, max_child_sitemaps=max_child_sitemaps,
                    child_include=child_include, _depth=_depth + 1,
                )
            )
        return out
    for u in root:
        loc = lastmod = None
        for c in u.iter():
            n = _local(c.tag)
            if n == "loc" and loc is None:
                loc = (c.text or "").strip()
            elif n in ("lastmod", "publication_date") and c.text:
                d = parse_date(c.text)
                # Prefer the news:publication_date over lastmod when both exist.
                if lastmod is None or n == "publication_date":
                    lastmod = d
        if not loc:
            continue
        hint = date_from_url(loc) or lastmod
        # lastmod >= publication date, so lastmod < since means "too old" for sure.
        if hint is not None and hint < since:
            continue
        out.append(Candidate(loc, hint))
    return out


_NON_ARTICLE_HOSTS = ("reddit.com", "redd.it", "youtube.com", "youtu.be", "twitter.com", "x.com", "imgur.com", "tiktok.com", "instagram.com", "facebook.com")


def from_reddit(subreddit: str, *, since: datetime, user_agent: str, listing: str = "top", t: str = "year", max_pages: int = 10) -> list[Candidate]:
    """Outbound article links from a subreddit (e.g. r/nottheonion) via the
    official Reddit API. Needs REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET (an app
    registered at https://www.reddit.com/prefs/apps). The linked articles are
    then fetched from their publishers like any other source (robots.txt applies)."""
    import os

    import requests

    cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
    if not cid or not secret:
        raise FetchError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set; skipping reddit discovery")
    tok = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(cid, secret), data={"grant_type": "client_credentials"},
        headers={"User-Agent": user_agent}, timeout=30,
    )
    tok.raise_for_status()
    headers = {"Authorization": f"bearer {tok.json()['access_token']}", "User-Agent": user_agent}
    out, after = [], None
    for _ in range(max_pages):
        r = requests.get(
            f"https://oauth.reddit.com/r/{subreddit}/{listing}",
            params={"t": t, "limit": 100, "after": after}, headers=headers, timeout=30,
        )
        r.raise_for_status()
        data = r.json()["data"]
        for ch in data["children"]:
            p = ch["data"]
            created = datetime.fromtimestamp(p["created_utc"], tz=timezone.utc)
            url = p.get("url_overridden_by_dest") or p.get("url") or ""
            if created < since or p.get("is_self") or any(h in url for h in _NON_ARTICLE_HOSTS):
                continue
            out.append(Candidate(url, created))
        after = data.get("after")
        if not after:
            break
    return out


def from_listing(fetcher: Fetcher, url_template: str, link_pattern: str, max_pages: int = 10) -> list[Candidate]:
    """Article links from paginated listing pages ("{n}" = page number), newest first."""
    rx = re.compile(link_pattern)
    out: list[Candidate] = []
    seen: set[str] = set()
    empty = 0
    for n in range(1, max_pages + 1):
        try:
            html = fetcher.get(url_template.format(n=n), use_cache=False)
        except FetchError as e:
            log.warning("listing page %d: %s", n, e)
            break
        links = [m.group(0) for m in rx.finditer(html) if m.group(0) not in seen]
        if not links:
            # Front pages sometimes use a different layout; stop after two empty pages.
            empty += 1
            if empty >= 2:
                break
            continue
        empty = 0
        for l in dict.fromkeys(links):
            seen.add(l)
            out.append(Candidate(l, date_from_url(l)))
    return out
