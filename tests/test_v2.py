"""Pipeline v2 (PRV 201 M2 design): graduation year and location stop being
hard filters.

v1 drops a job the moment the prefilter finds a graduation window or an onsite
location that excludes the candidate. The August eval traced 6 of its 7 missed
good jobs to the graduation year, so v2 passes those two findings to the match
model as warnings instead, and the gate removes a job only for a quoted
location or work-authorization conflict. v1 stays the default so the two can
be run side by side on the same labeled jobs.
"""

import pytest

from offerpilot import graph, prefilter
from offerpilot.models import EvidenceRef, FilterResult, MatchResult, NormalizedJob
from offerpilot.profile import load_profile
from offerpilot.store import db


def _fr(rule, outcome, value=None):
    return FilterResult(outcome=outcome, rule=rule, extracted_value=value,
                        reason="r")


# --- prefilter.decide ------------------------------------------------------

def test_v1_still_drops_a_graduation_window_fail():
    results = [_fr("graduation_window", "fail"), _fr("pay_floor", "pass")]
    assert prefilter.decide(results) == "filtered_out"


def test_v2_lets_a_graduation_window_fail_through_as_a_warning():
    results = [_fr("graduation_window", "fail"), _fr("pay_floor", "pass")]
    assert prefilter.decide(results, version="v2") == "ready_for_match"


def test_v2_lets_a_location_fail_through_as_a_warning():
    assert prefilter.decide([_fr("location", "fail")],
                            version="v2") == "ready_for_match"


@pytest.mark.parametrize("rule", ["years_of_experience", "work_authorization",
                                  "excluded_company", "pay_floor"])
def test_v2_keeps_every_other_rule_a_hard_filter(rule):
    results = [_fr("graduation_window", "fail"), _fr(rule, "fail")]
    assert prefilter.decide(results, version="v2") == "filtered_out"


def test_unknown_pipeline_version_is_refused():
    with pytest.raises(ValueError):
        prefilter.decide([], version="v3")


def test_prefilter_warnings_lists_only_soft_fails():
    results = [_fr("graduation_window", "fail", "Class of 2026"),
               _fr("location", "pass"), _fr("pay_floor", "unknown")]
    assert prefilter.warnings(results) == [results[0]]


# --- MatchResult -----------------------------------------------------------

def _match(eligibility="pass", conflict=None, score=(25, 18, 12, 12, 18)):
    s, p, d, se, pr = score
    return MatchResult(
        eligibility=eligibility,
        eligibility_evidence_excerpt="quoted" if eligibility == "fail" else None,
        eligibility_conflict_type=conflict,
        skills_score=s, project_score=p, domain_score=d, seniority_score=se,
        preference_score=pr,
        evidence=[EvidenceRef(source_id="sample_project", supporting_text="x")],
        confidence=0.7)


def test_conflict_type_defaults_to_none():
    assert _match().eligibility_conflict_type is None


def test_conflict_type_rejects_an_unknown_value():
    with pytest.raises(ValueError):
        _match(eligibility="fail", conflict="vibes")


# --- graph gate --------------------------------------------------------------

class _Stub:
    def __init__(self, result):
        self.result, self.prompts = result, []

    def structured(self, *, node, run_id, system, user, schema, validate=None):
        self.prompts.append((system, user))
        if validate is not None:
            validate(self.result)
        return self.result


@pytest.fixture()
def env(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    db.init_schema(conn)
    job = NormalizedJob(source="greenhouse", external_id="1", company_id="c",
                        title="SWE Intern", location="Austin, TX",
                        url="https://x.co/1", canonical_url="https://x.co/1",
                        description_text="Open to the Class of 2026. Onsite.")
    _, vid = db.upsert_job(conn, job)
    return conn, vid, load_profile("profile.example.yaml")


def _ready(conn, vid, results=()):
    db.record_filter_results(conn, vid, list(results))
    db.set_status(conn, vid, "ready_for_match")
    return conn.execute("SELECT * FROM job_versions WHERE id=?",
                        (vid,)).fetchone()


def _run(conn, row, profile, match, version):
    stub = _Stub(match)
    status = graph.run_match_for_version(
        conn, stub, profile, row, threshold=60, max_auto_retries=3,
        pipeline_version=version)
    return status, stub


def test_v1_gate_drops_any_eligibility_fail(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile,
                     _match("fail", "graduation_year"), "v1")
    assert status == "eligibility_failed"


def test_v2_gate_sends_a_graduation_conflict_to_review(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile,
                     _match("fail", "graduation_year"), "v2")
    assert status == "pending_review"


def test_v2_gate_treats_an_untyped_fail_as_a_warning(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile, _match("fail", None), "v2")
    assert status == "pending_review"


@pytest.mark.parametrize("conflict", ["location", "work_authorization"])
def test_v2_gate_still_drops_a_quoted_hard_conflict(env, conflict):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile, _match("fail", conflict), "v2")
    assert status == "eligibility_failed"


def test_v2_soft_fail_with_a_low_score_is_scored_low(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile,
                     _match("fail", "graduation_year", (5, 5, 5, 5, 5)), "v2")
    assert status == "scored_low"


# --- prompts -----------------------------------------------------------------

def test_v2_prompt_carries_the_prefilter_warnings(env):
    conn, vid, profile = env
    row = _ready(conn, vid, [_fr("graduation_window", "fail",
                                 "Open to the Class of 2026")])
    _, stub = _run(conn, row, profile, _match(), "v2")
    system, user = stub.prompts[0]
    assert "PREFILTER WARNINGS" in user
    assert "Open to the Class of 2026" in user
    assert "graduation" in system.lower()


def test_v1_prompt_has_no_warning_block(env):
    conn, vid, profile = env
    row = _ready(conn, vid, [_fr("graduation_window", "fail",
                                 "Open to the Class of 2026")])
    _, stub = _run(conn, row, profile, _match(), "v1")
    _, user = stub.prompts[0]
    assert "PREFILTER WARNINGS" not in user


# --- cli ---------------------------------------------------------------------

def test_collect_uses_the_configured_pipeline_version(conn, monkeypatch):
    from offerpilot import cli
    job = NormalizedJob(
        source="greenhouse", external_id="9", company_id="acme",
        title="SWE Intern", location="Remote",
        url="https://x.co/9", canonical_url="https://x.co/9",
        description_text="Open to students graduating in 2026.")
    monkeypatch.setattr(cli, "_collect_company", lambda company: [job])
    profile = load_profile("profile.example.yaml")
    cfg = {"companies": [{"id": "acme", "ats": "greenhouse",
                          "ats_slug": "acme"}],
           "pipeline": {"version": "v2"}}
    cli.cmd_collect(conn, cfg, profile)
    statuses = [r["status"] for r in conn.execute(
        "SELECT status FROM job_versions")]
    rules = {r["rule"]: r["outcome"] for r in conn.execute(
        "SELECT rule, outcome FROM filter_results")}
    assert rules["graduation_window"] == "fail"
    assert statuses == ["ready_for_match"]
