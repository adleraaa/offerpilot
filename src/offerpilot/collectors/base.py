import html
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

TRACKING_PREFIXES = ("utm_", "gh_src", "lever-origin", "ref", "fbclid",
                     "gclid")


ALLOWED_SCHEMES = ("http", "https")


def canonicalize_url(url: str) -> str:
    """Normalize a posting URL, rejecting any scheme we would not follow.

    Posting URLs are untrusted input. A `javascript:` or `data:` URL that
    reaches `jobs.canonical_url` is a stored XSS payload aimed at whatever
    renders it later, so it is refused at the boundary rather than sanitized
    at each render site. Raising (not returning "") keeps the bad job visible:
    `cmd_collect` isolates per-job failures and reports the count.
    """
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"refusing non-http(s) posting URL with scheme "
            f"{parts.scheme.lower()!r}")
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not any(k.lower().startswith(p) for p in TRACKING_PREFIXES)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path,
                       urlencode(kept), ""))


def strip_html(text: str) -> str:
    # Greenhouse serves HTML-escaped HTML, so an entity inside it arrives
    # escaped twice: "&amp;nbsp;" -> "&nbsp;" -> "\xa0". Two passes decode
    # that and stop. A third pass would also decode "&amp;amp;nbsp;" -- what a
    # posting that literally discusses `&nbsp;` looks like on the wire -- into
    # whitespace, silently deleting prose; unbounded fixed-point unescaping
    # deletes arbitrarily deep nestings. Two is the smallest bound that
    # handles the real wire format, so it is the bound that loses least.
    unescaped = text
    for _ in range(2):
        once = html.unescape(unescaped)
        if once == unescaped:
            break
        unescaped = once
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    collapsed = re.sub(r"\s+", " ", no_tags).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", collapsed)
