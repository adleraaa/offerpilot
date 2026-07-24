import json
from pathlib import Path
from offerpilot.collectors import base, greenhouse, lever

FIX = Path(__file__).parent / "fixtures"


def test_canonicalize_url_strips_tracking_and_slash():
    u = "https://Boards.Greenhouse.io/x/jobs/1/?gh_src=a&utm_source=b&x=1"
    assert base.canonicalize_url(u) == \
        "https://boards.greenhouse.io/x/jobs/1?x=1"


def test_canonicalize_preserves_query_value_slashes():
    u = "https://x.co/jobs?next=/careers/"
    assert base.canonicalize_url(u) == "https://x.co/jobs?next=%2Fcareers%2F"


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
