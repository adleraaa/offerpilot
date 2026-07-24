import json
import pytest
from offerpilot.models import NormalizedJob
from offerpilot.store import db
from offerpilot.profile import load_profile
from offerpilot import graph
from offerpilot.llm import RetryableLLMError, PermanentLLMError


@pytest.fixture()
def env(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    db.init_schema(conn)
    job = NormalizedJob(source="lever", external_id="1", company_id="c",
                        title="AI Intern", location="New York, NY",
                        url="https://x.co/1", canonical_url="https://x.co/1",
                        description_text="Build agents.")
    _, v = db.upsert_job(conn, job)
    db.set_status(conn, v, "ready_for_match")
    row = conn.execute("SELECT * FROM job_versions WHERE id=?", (v,)).fetchone()
    return conn, row, load_profile("profile.example.yaml")


class StubLLM:
    def __init__(self, result=None, exc=None):
        self.result, self.exc = result, exc

    def structured(self, **kwargs):
        if self.exc:
            raise self.exc
        return self.result


def good_match(score_each):
    from offerpilot.models import MatchResult, EvidenceRef
    return MatchResult(
        eligibility="pass", eligibility_reasons=[],
        skills_score=score_each["skills"], project_score=score_each["proj"],
        domain_score=score_each["dom"], seniority_score=score_each["sen"],
        preference_score=score_each["pref"],
        evidence=[EvidenceRef(source_id="sample_project",
                              supporting_text="built LLM app")],
        gaps=[], uncertainties=[], confidence=0.7)


def test_high_score_goes_to_pending_review(env):
    conn, row, profile = env
    m = good_match(dict(skills=25, proj=18, dom=12, sen=12, pref=18))
    status = graph.run_match_for_version(conn, StubLLM(result=m), profile,
                                         row, threshold=60, max_auto_retries=3)
    assert status == "pending_review"
    item = conn.execute("SELECT * FROM review_items").fetchone()
    assert item["total_score"] == 85


def test_low_score_goes_scored_low(env):
    conn, row, profile = env
    m = good_match(dict(skills=5, proj=5, dom=5, sen=5, pref=5))
    status = graph.run_match_for_version(conn, StubLLM(result=m), profile,
                                         row, threshold=60, max_auto_retries=3)
    assert status == "scored_low"


def test_invalid_evidence_id_is_permanent(env):
    conn, row, profile = env
    m = good_match(dict(skills=25, proj=18, dom=12, sen=12, pref=18))
    m.evidence[0].source_id = "made_up_project"
    status = graph.run_match_for_version(conn, StubLLM(result=m), profile,
                                         row, threshold=60, max_auto_retries=3)
    assert status == "permanent_error"


def test_retryable_error_resets_until_limit(env):
    conn, row, profile = env
    stub = StubLLM(exc=RetryableLLMError("429"))
    for expected in ["ready_for_match", "ready_for_match", "permanent_error"]:
        status = graph.run_match_for_version(conn, stub, profile, row,
                                             threshold=60, max_auto_retries=3)
        row = conn.execute("SELECT * FROM job_versions WHERE id=?",
                           (row["id"],)).fetchone()
        assert status == expected


def test_prompt_wraps_untrusted_block(env):
    _, row, profile = env
    system, user = graph.build_prompts(row, profile)
    assert "<untrusted_job_posting>" in user
    assert "Build agents." in user
    assert "not instructions" in system.lower() or \
           "are data" in system.lower()
