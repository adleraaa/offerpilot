from datetime import datetime, timezone

import requests
from offerpilot.models import NormalizedJob
from offerpilot.collectors.base import canonicalize_url

API = "https://api.lever.co/v0/postings/{slug}?mode=json"


def fetch(slug: str) -> list:
    resp = requests.get(API.format(slug=slug), timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse(payload: list, company_id: str) -> list[NormalizedJob]:
    out = []
    for p in payload:
        url = p["hostedUrl"]
        created = p.get("createdAt")
        posted_at = (datetime.fromtimestamp(created / 1000, tz=timezone.utc).isoformat()
                     if created else None)
        out.append(NormalizedJob(
            source="lever",
            external_id=p["id"],
            company_id=company_id,
            title=p["text"],
            location=(p.get("categories") or {}).get("location", ""),
            url=url,
            canonical_url=canonicalize_url(url),
            description_text=p.get("descriptionPlain", ""),
            posted_at=posted_at,
        ))
    return out
