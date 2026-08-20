"""Demo mode: synthetic data, pre-recorded outputs, no key, no network.

The tests that matter most here are the ones about `MockLLM`. A mock that
accepts `**kwargs` and quietly drops `validate` would disarm the grounding
gate for every job that goes through the demo path, so the demo would advertise
a citation check the demo itself does not run. Two tests pin that down: the
validator must be called, and an unrecorded job must raise rather than be
invented.
"""

import inspect
import json

import pytest

from offerpilot import cli
from offerpilot.brief import ApplicationBrief, make_brief_validator
from offerpilot.demo import DEMO_DIR, MockLLM, run_demo, seed_demo_db
from offerpilot.graph import make_evidence_validator, run_match_for_version
from offerpilot.llm import LLMClient, PermanentLLMError
from offerpilot.models import MatchResult
from offerpilot.profile import load_profile
from offerpilot.store import db
from tests.conftest import REPO_ROOT, _ready_row


# --- fixture shape ---------------------------------------------------------

def test_demo_dir_points_at_the_repo_demo_directory():
    """`parents[2]` is only correct for a src-layout install; check it."""
    assert DEMO_DIR == REPO_ROOT / "demo"
    assert (DEMO_DIR / "demo_jobs.json").exists()
    assert (DEMO_DIR / "demo_profile.yaml").exists()
    assert (DEMO_DIR / "recorded_outputs.json").exists()


def test_demo_profile_is_obviously_synthetic():
    p = load_profile("demo/demo_profile.yaml")
    assert "Example" in p.identity.education or "Doe" in p.identity.name


def test_demo_jobs_are_marked_synthetic():
    with open("demo/demo_jobs.json", encoding="utf-8") as f:
        jobs = json.load(f)
    assert 3 <= len(jobs) <= 5
    for job in jobs:
        assert "SYNTHETIC" in job["description_text"].upper()


def test_recorded_outputs_only_cite_ids_the_demo_profile_defines():
    """The fixture must not be the thing that breaks the grounding check."""
    valid = load_profile("demo/demo_profile.yaml").experience_ids()
    with open("demo/recorded_outputs.json", encoding="utf-8") as f:
        recorded = json.load(f)
    cited = set()
    for payload in recorded.values():
        cited |= {e["source_id"] for e in payload.get("evidence", [])}
        cited |= {e["source_id"] for e in payload.get("cited_evidence", [])}
        cited |= {tp["evidence_source_id"]
                  for tp in payload.get("talking_points", [])}
    assert cited
    assert cited <= valid


# --- MockLLM ---------------------------------------------------------------

def test_mock_llm_refuses_to_invent_output_for_unknown_jobs():
    llm = MockLLM({})
    with pytest.raises(KeyError):
        llm.structured(node="match", run_id=1, system="s", user="u",
                       schema=None, external_id="nope")


def test_mock_llm_accepts_every_keyword_the_real_client_takes():
    real = inspect.signature(LLMClient.structured).parameters
    mock = inspect.signature(MockLLM.structured).parameters
    for name, param in real.items():
        if name == "self":
            continue
        assert name in mock, f"MockLLM.structured drops {name!r}"
        assert mock[name].kind is param.kind
    assert "validate" in mock


def test_mock_llm_calls_the_validator_and_a_forged_citation_is_rejected(profile):
    """`validate=` is the only enforcement path the client offers."""
    recorded = {"match:x": {
        "eligibility": "pass", "eligibility_reasons": [],
        "skills_score": 30, "project_score": 20, "domain_score": 15,
        "seniority_score": 15, "preference_score": 20,
        "evidence": [{"source_id": "no_such_experience", "section": "projects",
                      "supporting_text": "invented"}],
        "gaps": [], "uncertainties": [], "confidence": 0.9}}
    llm = MockLLM(recorded)
    # PermanentLLMError, not ValueError: a recording cannot be repaired, so the
    # first rejection is terminal and the mock raises what the real client
    # raises once its repair budget is spent.
    with pytest.raises(PermanentLLMError, match="no_such_experience") as exc:
        llm.structured(node="match", run_id=1, system="s", user="u",
                       schema=MatchResult,
                       validate=make_evidence_validator(profile, 60),
                       external_id="x")
    # The cause proves the rejection came from `validate` and not from parsing.
    assert isinstance(exc.value.__cause__, ValueError)


def test_mock_llm_calls_the_brief_validator_too(profile):
    recorded = {"brief:x": {
        "why_it_fits": "fits", "cited_evidence": [],
        "main_gaps": [], "resume_bullets_to_emphasize": [],
        "talking_points": [{"theme": "why_this_role", "point": "p",
                            "evidence_source_id": "pathpilot",
                            "generic": False}],
        "outreach_paragraph": None}}
    llm = MockLLM(recorded)
    with pytest.raises(PermanentLLMError, match="generic"):
        llm.structured(node="brief", run_id=1, system="s", user="u",
                       schema=ApplicationBrief,
                       validate=make_brief_validator(profile),
                       external_id="x")


def test_a_forged_citation_never_reaches_pending_review(conn, profile):
    """End to end: a bad recorded output must not be reviewable."""
    row = _ready_row(conn)
    llm = MockLLM({"match:1": {
        "eligibility": "pass", "eligibility_reasons": [],
        "skills_score": 30, "project_score": 20, "domain_score": 15,
        "seniority_score": 15, "preference_score": 20,
        "evidence": [{"source_id": "no_such_experience", "section": "projects",
                      "supporting_text": "invented"}],
        "gaps": [], "uncertainties": [], "confidence": 0.9}})
    llm.external_id = "1"
    final = run_match_for_version(conn, llm, profile, row, threshold=60,
                                  max_auto_retries=3, brief_enabled=True)
    assert final == "permanent_error"
    assert db.get_review_queue(conn) == []


# --- seeding ---------------------------------------------------------------

def test_demo_seeds_without_any_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    path = str(tmp_path / "demo.db")
    _, profile = seed_demo_db(path)
    conn = db.connect(path)
    counts = {r["status"]: r["c"] for r in conn.execute(
        "SELECT status, COUNT(*) c FROM job_versions GROUP BY status")}
    assert sum(counts.values()) >= 3
    assert counts.get("pending_review", 0) >= 1
    assert profile.identity.name


def test_demo_review_items_have_briefs(tmp_path):
    path = str(tmp_path / "demo.db")
    seed_demo_db(path)
    conn = db.connect(path)
    rows = db.get_review_queue(conn)
    assert rows
    assert all(r["brief_json"] for r in rows)


def test_demo_covers_the_interesting_outcomes(tmp_path):
    """A demo that only shows happy paths teaches nothing."""
    path = str(tmp_path / "demo.db")
    seed_demo_db(path)
    conn = db.connect(path)
    statuses = {r["status"] for r in conn.execute(
        "SELECT DISTINCT status FROM job_versions")}
    assert "pending_review" in statuses
    assert statuses & {"filtered_out", "scored_low", "eligibility_failed"}


def test_demo_db_has_an_unresolved_eligibility_case(tmp_path):
    """The 'Eligibility unresolved' banner needs something to show."""
    path = str(tmp_path / "demo.db")
    seed_demo_db(path)
    conn = db.connect(path)
    matches = [json.loads(r["match_json"]) for r in db.get_review_queue(conn)]
    assert any(m["eligibility"] == "unknown" for m in matches)


def test_prefilter_drops_two_jobs_before_any_model_call(tmp_path):
    """MockLLM has no entry for them, so reaching it would have raised."""
    path = str(tmp_path / "demo.db")
    seed_demo_db(path)
    conn = db.connect(path)
    rows = conn.execute(
        "SELECT j.external_id e, jv.id, jv.status FROM job_versions jv "
        "JOIN jobs j ON j.id = jv.job_id").fetchall()
    by_ext = {r["e"]: r for r in rows}
    assert by_ext["demo-2"]["status"] == "filtered_out"
    assert by_ext["demo-5"]["status"] == "filtered_out"
    for ext in ("demo-2", "demo-5"):
        assert conn.execute("SELECT COUNT(*) c FROM runs WHERE job_version_id=?",
                            (by_ext[ext]["id"],)).fetchone()["c"] == 0
    assert conn.execute(
        "SELECT COUNT(*) c FROM filter_results WHERE job_version_id=? "
        "AND outcome='fail' AND rule='pay_floor'",
        (by_ext["demo-2"]["id"],)).fetchone()["c"] == 1
    assert conn.execute(
        "SELECT COUNT(*) c FROM filter_results WHERE job_version_id=? "
        "AND outcome='fail' AND rule='excluded_company'",
        (by_ext["demo-5"]["id"],)).fetchone()["c"] == 1


def test_demo_companies_are_named_so_the_panel_has_something_to_show(tmp_path):
    path = str(tmp_path / "demo.db")
    seed_demo_db(path)
    conn = db.connect(path)
    names = {r["id"]: r["name"] for r in conn.execute("SELECT * FROM companies")}
    assert names.get("examplecorp")
    assert names.get("excludedcorp")


# --- entry points ----------------------------------------------------------

def test_run_demo_without_the_panel_returns_a_seeded_db(tmp_path, monkeypatch,
                                                        capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)          # no config.yaml, no profile.yaml
    monkeypatch.setattr("tempfile.mkdtemp",
                        lambda *a, **k: str(tmp_path / "demo-run"))
    (tmp_path / "demo-run").mkdir()
    path = run_demo(serve_panel=False)
    conn = db.connect(path)
    assert db.get_review_queue(conn)
    assert "no API key" in capsys.readouterr().out


def test_demo_cli_needs_no_config_profile_or_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("offerpilot.demo.run_demo",
                        lambda **kw: calls.append(kw) or "x.db")
    cli.main(["demo"])
    assert calls == [{"host": "127.0.0.1", "port": 8000}]
    # Intercepted before setup: no config read, no database created.
    assert not (tmp_path / "data").exists()


def test_the_panel_serves_the_demo_database(tmp_path):
    """Step 9 by hand, in-process: the demo is only a demo if it renders."""
    from fastapi.testclient import TestClient
    from offerpilot.panel.app import create_app

    path = str(tmp_path / "demo.db")
    _, prof = seed_demo_db(path)
    client = TestClient(create_app(path, prof))
    queue = client.get("/api/queue").json()["items"]
    assert [q["title"] for q in queue] == ["AI Engineering Intern",
                                           "Software Engineer, Internal Tools"]
    item = client.get(f"/api/item/{queue[1]['job_version_id']}").json()
    assert item["eligibility_unresolved"] is True     # drives the banner
    assert item["brief"]["why_it_fits"]
    assert client.get("/api/blind/next").status_code == 200
