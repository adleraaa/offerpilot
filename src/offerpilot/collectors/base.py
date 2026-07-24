import html
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

TRACKING_PREFIXES = ("utm_", "gh_src", "lever-origin", "ref", "fbclid",
                     "gclid")


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not any(k.lower().startswith(p) for p in TRACKING_PREFIXES)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path,
                       urlencode(kept), ""))


def strip_html(text: str) -> str:
    unescaped = html.unescape(text)
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    collapsed = re.sub(r"\s+", " ", no_tags).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", collapsed)
