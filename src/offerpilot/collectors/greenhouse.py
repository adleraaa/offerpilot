import requests
from offerpilot.models import NormalizedJob
from offerpilot.collectors.base import canonicalize_url, strip_html

API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"


def fetch(slug: str) -> dict:
    resp = requests.get(API.format(slug=slug), timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse(payload: dict, company_id: str) -> list[NormalizedJob]:
    out = []
    for j in payload.get("jobs", []):
        url = j["absolute_url"]
        try:
            canonical = canonicalize_url(url)
        except ValueError as e:
            # One poisoned posting costs one job, not the whole board.
            print(f"[greenhouse] skipping {company_id} job {j.get('id')}: {e}")
            continue
        out.append(NormalizedJob(
            source="greenhouse",
            external_id=str(j["id"]),
            company_id=company_id,
            title=j["title"],
            location=(j.get("location") or {}).get("name", ""),
            url=url,
            canonical_url=canonical,
            description_text=strip_html(j.get("content", "")),
            posted_at=j.get("updated_at"),
        ))
    return out
