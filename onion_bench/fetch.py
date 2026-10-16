"""Polite HTTP fetching: robots.txt compliance, per-host rate limiting, disk cache.

This module deliberately does NOT try to get around bot protection (CAPTCHAs,
JS challenges, DataDome/Cloudflare walls). If a site blocks automated access,
`BlockedError` is raised and the source has to be collected another way (e.g.
`onion-bench import-html` on pages saved by hand, or by asking the publisher).
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import requests

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "onion-bench/0.1 (academic research benchmark)"

# Markers of anti-bot interstitials. We stop when we see one; we never solve them.
_BLOCK_MARKERS = (
    "captcha-delivery.com",  # DataDome
    "cf-chl-",  # Cloudflare challenge
    "challenges.cloudflare.com",
    "Please enable JS and disable any ad blocker",
)


class FetchError(RuntimeError):
    pass


class RobotsDisallowed(FetchError):
    pass


class BlockedError(FetchError):
    """The site served a bot-protection challenge instead of content."""


@dataclass
class Fetcher:
    user_agent: str = DEFAULT_USER_AGENT
    # robots.txt is checked for *every* agent listed here; all must be allowed.
    robots_agents: tuple[str, ...] = ("onion-bench",)
    min_delay_s: float = 1.0
    timeout_s: float = 30.0
    cache_dir: Path | None = None
    session: requests.Session = field(default_factory=requests.Session)
    _robots: dict[str, urllib.robotparser.RobotFileParser | None] = field(default_factory=dict)
    _last_hit: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en,pt;q=0.8",
            }
        )
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- robots ---------------------------------------------------------------
    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._throttle(parts.netloc)
                r = self.session.get(origin + "/robots.txt", timeout=self.timeout_s)
                if r.status_code >= 400:
                    # No robots.txt (4xx) means everything is allowed, per RFC 9309.
                    rp.parse([])
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException as e:
                log.warning("could not read robots.txt for %s (%s); treating as disallowed", origin, e)
                rp = None
            self._robots[origin] = rp
        return self._robots[origin]

    def allowed(self, url: str) -> bool:
        rp = self._robots_for(url)
        if rp is None:
            return False
        return all(rp.can_fetch(agent, url) for agent in self.robots_agents)

    # -- fetching -------------------------------------------------------------
    def _throttle(self, host: str) -> None:
        last = self._last_hit.get(host)
        if last is not None:
            wait = self.min_delay_s - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".html")

    def get(self, url: str, *, use_cache: bool = True, retries: int = 2) -> str:
        cache = self._cache_path(url)
        if use_cache and cache is not None and cache.exists():
            return cache.read_text(encoding="utf-8")
        if not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url} for {self.robots_agents}")
        host = urlsplit(url).netloc
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            self._throttle(host)
            try:
                r = self.session.get(url, timeout=self.timeout_s)
            except requests.RequestException as e:
                last_exc = e
                time.sleep(2**attempt)
                continue
            body = r.text
            if any(m in body[:5000] for m in _BLOCK_MARKERS):
                raise BlockedError(f"{url}: served a bot-protection challenge (HTTP {r.status_code})")
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = FetchError(f"{url}: HTTP {r.status_code}")
                time.sleep(int(r.headers.get("retry-after", 2**attempt)) if r.headers.get("retry-after", "").isdigit() else 2**attempt)
                continue
            if r.status_code >= 400:
                raise FetchError(f"{url}: HTTP {r.status_code}")
            declared = "charset" in r.headers.get("content-type", "").lower()
            if not r.encoding or (r.encoding.lower() == "iso-8859-1" and not declared):
                r.encoding = r.apparent_encoding
                body = r.text
            if cache is not None:
                cache.write_text(body, encoding="utf-8")
            return body
        raise FetchError(f"{url}: giving up after {retries + 1} attempts ({last_exc})")
