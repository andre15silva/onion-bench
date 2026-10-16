"""Claims from fact-checkers, via their schema.org ClaimReview markup.

Each fact-check page yields one headline-only item: the claim as the
fact-checker recorded it (`claimReviewed`) and a label derived from the
verdict. Only clear-cut verdicts are kept (source config `rating_map`):
false/fake claims become `misinformation` (label fake); true claims become
`fact_checked_true` (label real; viral claims that sound false but are true).
Everything else (mixed, misleading, unproven, miscaptioned...) is dropped.

Also dropped:
- statements attributed to a named person: whether "X said Y" is real news
  and whether Y is true are different questions, so the label is ambiguous;
- claims that are only about a photo or video, which can't be judged
  without the media;
- claims first seen before the collection window (if the page says when).
"""

from __future__ import annotations

import hashlib
import html as htmllib
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from .discover import parse_date
from .extract import _iter_ld, _types

_MEDIA_ONLY = re.compile(
    r"^\s*(?:an?|the|this|these)?\s*(?:photo(?:graph)?s?|images?|pictures?|videos?|clips?|footage|screenshots?"
    r"|fotografias?|fotos?|imagens?|v[íi]deos?)\b",
    re.I,
)


# Claim "authors" that are anonymous posters rather than a named person.
_GENERIC_AUTHOR = re.compile(
    r"utilizador|usu[aá]rio|user|users|redes sociais|social media|viral|internet|online|facebook|instagram|tiktok|"
    r"whatsapp|telegram|twitter|\bx\b|youtube|multiple|v[aá]rios|various|unknown|desconhecid|an[oó]nim|anonymous|posts?\b|"
    r"publica[cç][aãõo]|website|site|blog|meme",
    re.I,
)


def _clean(s: str) -> str:
    return " ".join(htmllib.unescape(s or "").split()).strip().strip('"“”').strip()


def find_claim_review(soup: BeautifulSoup) -> dict | None:
    for x in _iter_ld(soup):
        if "ClaimReview" in _types(x):
            return x
    return None


def build_claim_record(src: dict, url: str, page_html: str, cfg=None) -> dict:
    soup = BeautifulSoup(page_html, "lxml")
    cr = find_claim_review(soup) or {}
    rating = cr.get("reviewRating") or {}
    verdict = _clean(rating.get("alternateName") or rating.get("name") or "")
    item = cr.get("itemReviewed") or {}
    author = item.get("author") or {}
    author_type = author.get("@type") if isinstance(author, dict) else None
    author_name = author.get("name") if isinstance(author, dict) else None
    claim = _clean(cr.get("claimReviewed") or "")
    review_date = parse_date(cr.get("datePublished"))
    claim_date = parse_date(item.get("datePublished"))
    appearance = item.get("appearance") or []
    if isinstance(appearance, dict):
        appearance = [appearance]

    rmap = src.get("rating_map", {})
    label = next((lab for lab, names in rmap.items() if verdict.lower() in {n.lower() for n in names}), None)
    reject = None
    if not cr:
        reject = "no_claimreview"
    elif not claim:
        reject = "no_claim_text"
    elif label is None:
        reject = f"rating({verdict or 'none'})"
    elif author_type == "Person" and author_name and not _GENERIC_AUTHOR.search(author_name):
        reject = "attributed_statement"
    elif _MEDIA_ONLY.match(claim):
        reject = "media_only_claim"
    elif claim_date is not None and src.get("_since") and claim_date < src["_since"]:
        reject = "claim_before_since"

    return {
        "source": src["name"],
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": claim,
        "original_title": None,
        "text": "",
        "headline_only": True,
        "published_at": review_date.isoformat() if review_date else None,
        "modified_at": None,
        "authors": [],
        "section": None,
        "keywords": [],
        "page_language": None,
        "site_name": None,
        "extraction_method": "claimreview",
        "word_count": 0,
        "label": label,
        "subset": {"fake": "misinformation", "real": "fact_checked_true"}.get(label),
        "sourcing_mode": "fact_check_claimreview",
        "selection_method": f"fact_check:{src['name']}:{verdict}",
        "reject": reject,
        "provenance": {
            "fact_checker": src["display_name"],
            "review_url": url,
            "verdict": verdict,
            "rating_value": rating.get("ratingValue"),
            "review_published_at": review_date.isoformat() if review_date else None,
            "claim_first_seen_at": claim_date.isoformat() if claim_date else None,
            "claim_author": author_name,
            "claim_author_type": author_type,
            "claim_appearances": [a.get("url") for a in appearance if isinstance(a, dict) and a.get("url")][:5],
            "claim_text_by": "fact-checker (ClaimReview.claimReviewed)",
        },
        "html_sha256": hashlib.sha256(page_html.encode("utf-8")).hexdigest(),
    }
