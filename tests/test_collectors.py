import json
from pathlib import Path

import pytest
from offerpilot.collectors import base, greenhouse, lever
from offerpilot.collectors.base import strip_html

FIX = Path(__file__).parent / "fixtures"


def test_canonicalize_url_strips_tracking_and_slash():
    u = "https://Boards.Greenhouse.io/x/jobs/1/?gh_src=a&utm_source=b&x=1"
    assert base.canonicalize_url(u) == \
        "https://boards.greenhouse.io/x/jobs/1?x=1"


def test_canonicalize_preserves_query_value_slashes():
    u = "https://x.co/jobs?next=/careers/"
    assert base.canonicalize_url(u) == "https://x.co/jobs?next=%2Fcareers%2F"


@pytest.mark.parametrize("url", [
    "javascript:alert(document.cookie)",
    "JavaScript:alert(1)",
    "  javascript:alert(1)",
    "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
    "vbscript:msgbox(1)",
    "file:///etc/passwd",
])
def test_canonicalize_url_rejects_non_http_schemes(url):
    """A posting URL is untrusted input; only http(s) may reach the DB."""
    with pytest.raises(ValueError):
        base.canonicalize_url(url)


def test_canonicalize_url_rejects_a_schemeless_url():
    with pytest.raises(ValueError):
        base.canonicalize_url("//evil.example/x")


@pytest.mark.parametrize("url", [
    "https://boards.greenhouse.io/x/jobs/1",
    "HTTP://Example.com/jobs/2",
])
def test_canonicalize_url_still_accepts_http_and_https(url):
    assert base.canonicalize_url(url).startswith(("http://", "https://"))


def test_greenhouse_skips_a_job_whose_url_is_not_http(capsys):
    """One poisoned posting must cost one job, not the whole board."""
    payload = {"jobs": [
        {"id": 1, "title": "Real", "location": {"name": "Remote"},
         "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
         "content": "Good"},
        {"id": 2, "title": "Poisoned", "location": {"name": "Remote"},
         "absolute_url": "javascript:alert(1)", "content": "Bad"},
    ]}
    jobs = greenhouse.parse(payload, company_id="acme")
    assert [j.title for j in jobs] == ["Real"]
    assert "skipping" in capsys.readouterr().out


def test_lever_skips_a_job_whose_url_is_not_http(capsys):
    payload = [
        {"id": "a", "text": "Real", "categories": {"location": "Remote"},
         "hostedUrl": "https://jobs.lever.co/acme/a", "descriptionPlain": "ok"},
        {"id": "b", "text": "Poisoned", "categories": {"location": "Remote"},
         "hostedUrl": "data:text/html,<script>alert(1)</script>",
         "descriptionPlain": "bad"},
    ]
    jobs = lever.parse(payload, company_id="acme")
    assert [j.title for j in jobs] == ["Real"]
    assert "skipping" in capsys.readouterr().out


def test_strip_html_unescapes_and_removes_tags():
    assert base.strip_html("&lt;p&gt;Build &lt;b&gt;agents&lt;/b&gt;.&lt;/p&gt;") == \
        "Build agents."


def test_strip_html_separates_adjacent_blocks():
    assert base.strip_html("<ul><li>BS degree</li><li>3+ years</li></ul>") == \
        "BS degree 3+ years"


def test_greenhouse_parse():
    payload = json.loads((FIX / "greenhouse_jobs.json").read_text())
    jobs = greenhouse.parse(payload, company_id="examplecorp")
    assert len(jobs) == 1
    j = jobs[0]
    assert j.source == "greenhouse" and j.external_id == "4011001"
    assert j.title == "AI Engineer Intern"
    assert "agents" in j.description_text and "<" not in j.description_text
    assert "utm_source" not in j.canonical_url


def test_lever_parse():
    payload = json.loads((FIX / "lever_postings.json").read_text())
    jobs = lever.parse(payload, company_id="samplestartup")
    j = jobs[0]
    assert j.source == "lever" and j.external_id == "ab12-cd34"
    assert j.location == "New York City"
    assert j.canonical_url.endswith("/ab12-cd34")
    assert j.posted_at.startswith("2025-07-01")


def test_strip_html_handles_double_escaped_greenhouse_content():
    """Greenhouse serves escaped HTML: one unescape leaves entities behind."""
    raw = "&lt;p&gt;Build things&amp;nbsp;with us&lt;/p&gt;"
    out = strip_html(raw)
    assert "&nbsp;" not in out
    assert "&amp;" not in out
    assert "<p>" not in out
    assert "Build things with us" in out


def test_strip_html_is_idempotent_on_plain_text():
    assert strip_html("Plain text, no markup.") == "Plain text, no markup."


def test_strip_html_does_not_unescape_forever():
    """The unescape bound must actually bind.

    A posting that literally discusses `&nbsp;` arrives from a board that
    escapes its HTML once, so it reaches us as `&amp;amp;nbsp;`. Two passes
    stop at `&nbsp;` and the prose survives; a third pass -- or unbounded
    fixed-point unescaping -- silently turns it into whitespace and the text
    is gone. "Research &amp; Development" is the single-escape base case: it
    reaches its fixed point after one pass, so it pins nothing on its own.
    """
    assert strip_html("&amp;amp;nbsp;") == "&nbsp;"
    assert "&" in strip_html("Research &amp; Development")
