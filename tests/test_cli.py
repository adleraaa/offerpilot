import pytest
from offerpilot.models import NormalizedJob, MatchResult, EvidenceRef
from offerpilot.store import db
from offerpilot.profile import load_profile
from offerpilot import cli


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_schema(c)
    return c


def seed(conn, desc="Build agents in New York, NY.", ext="1"):
    job = NormalizedJob(source="lever", external_id=ext, company_id="c",
                        title="AI Intern", location="New York, NY",
                        url=f"https://x.co/{ext}",
                        canonical_url=f"https://x.co/{ext}",
                        description_text=desc)
    return db.upsert_job(conn, job)


def test_collect_applies_prefilter(conn, monkeypatch):
    calls = {}

    def fake_collect_company(company, cfg):
        return [NormalizedJob(
            source="greenhouse", external_id="9", company_id=company["id"],
            title="Staff Engineer", location="San Francisco, CA",
            url="https://x.co/9", canonical_url="https://x.co/9",
            description_text="Requires 8+ years. Onsite in San Francisco.")]

    monkeypatch.setattr(cli, "_collect_company", fake_collect_company)
    cfg = {"companies": [{"id": "c", "ats": "greenhouse", "ats_slug": "c"}]}
    profile = load_profile("profile.example.yaml")
    out = cli.cmd_collect(conn, cfg, profile)
    assert out["inserted"] == 1
    assert cli.cmd_status(conn).get("filtered_out") == 1


def test_match_command_processes_ready(conn):
    _, v = seed(conn)
    db.set_status(conn, v, "ready_for_match")

    class StubLLM:
        def structured(self, **kwargs):
            return MatchResult(
                eligibility="pass", skills_score=25, project_score=18,
                domain_score=12, seniority_score=12, preference_score=18,
                evidence=[EvidenceRef(source_id="sample_project",
                                      supporting_text="x")],
                confidence=0.8)

    cfg = {"match": {"score_threshold": 60, "max_auto_retries": 3}}
    profile = load_profile("profile.example.yaml")
    out = cli.cmd_match(conn, cfg, profile, StubLLM())
    assert out == {"pending_review": 1}


def test_status_counts(conn):
    seed(conn, ext="1")
    seed(conn, ext="2")
    assert cli.cmd_status(conn) == {"new": 2}


def test_retry_resets_permanent(conn):
    _, v = seed(conn)
    db.set_status(conn, v, "ready_for_match")
    db.set_status(conn, v, "matching")
    db.set_status(conn, v, "permanent_error")
    assert cli.cmd_retry(conn) == 1
    assert cli.cmd_status(conn) == {"ready_for_match": 1}
