import pytest
from pydantic import ValidationError
from offerpilot.models import (
    NormalizedJob, FilterResult, EvidenceRef, MatchResult, total_score,
)


def make_match(**over):
    base = dict(
        eligibility="pass", eligibility_reasons=[],
        eligibility_evidence_excerpt=None,
        skills_score=20, project_score=15, domain_score=10,
        seniority_score=10, preference_score=15,
        evidence=[EvidenceRef(source_id="pathpilot", supporting_text="x")],
        gaps=[], uncertainties=[], confidence=0.8,
    )
    base.update(over)
    return MatchResult(**base)


def test_total_score_is_sum_of_subscores():
    assert total_score(make_match()) == 70


def test_subscore_bounds_enforced():
    with pytest.raises(ValidationError):
        make_match(skills_score=31)


def test_eligibility_fail_requires_excerpt():
    with pytest.raises(ValidationError):
        make_match(eligibility="fail")
    m = make_match(eligibility="fail",
                   eligibility_evidence_excerpt="requires 8+ years")
    assert m.eligibility == "fail"


def test_normalized_job_requires_fields():
    with pytest.raises(ValidationError):
        NormalizedJob(source="greenhouse")


def test_filter_result_outcomes():
    r = FilterResult(outcome="unknown", rule="work_authorization",
                     extracted_value=None, reason="not stated")
    assert r.outcome == "unknown"
