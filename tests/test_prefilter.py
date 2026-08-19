import json

from conftest import REPO_ROOT

from offerpilot.models import NormalizedJob
from offerpilot.profile import load_profile
from offerpilot import prefilter
from offerpilot.prefilter import _rule_authorization, _rule_years

REAL_SNIPPETS = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "real_posting_snippets.json")
    .read_text(encoding="utf-8"))


def make_job(description_text, location="New York, NY"):
    return NormalizedJob(source="lever", external_id="1", company_id="c",
                         title="Engineer", location=location,
                         url="https://x.co/1", canonical_url="https://x.co/1",
                         description_text=description_text)


def make_profile(work_authorization=None):
    p = load_profile("profile.example.yaml")
    if work_authorization is not None:
        p.constraints.work_authorization = work_authorization
    return p


def outcome_of(results, rule):
    return next(r.outcome for r in results if r.rule == rule)


def test_explicit_years_requirement_fails():
    r = prefilter.run_prefilter(
        make_job("Requires 8+ years of professional experience."),
        make_profile())
    assert outcome_of(r, "years_of_experience") == "fail"
    assert prefilter.decide(r) == "filtered_out"


def test_no_mention_is_unknown():
    r = prefilter.run_prefilter(make_job("We build agents."), make_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"
    assert outcome_of(r, "work_authorization") == "unknown"


def test_clearance_requirement_fails_authorization():
    r = prefilter.run_prefilter(
        make_job("Active TS/SCI security clearance required."), make_profile())
    assert outcome_of(r, "work_authorization") == "fail"


def test_location_conflict_fails_when_onsite_elsewhere():
    r = prefilter.run_prefilter(
        make_job("This role is onsite in San Francisco, CA.",
                 location="San Francisco, CA"), make_profile())
    assert outcome_of(r, "location") == "fail"


def test_remote_location_passes():
    r = prefilter.run_prefilter(
        make_job("Fully remote role.", location="Remote"), make_profile())
    assert outcome_of(r, "location") == "pass"


def test_excluded_company():
    p = make_profile()
    p.constraints.excluded_companies = ["c"]
    r = prefilter.run_prefilter(make_job("Anything"), p)
    assert outcome_of(r, "excluded_company") == "fail"


def test_no_clearance_required_is_unknown():
    r = prefilter.run_prefilter(
        make_job("No security clearance required for this role."),
        make_profile())
    assert outcome_of(r, "work_authorization") == "unknown"


def test_clearance_not_required_is_unknown():
    r = prefilter.run_prefilter(
        make_job("Security clearance is not required."), make_profile())
    assert outcome_of(r, "work_authorization") == "unknown"


def test_vesting_years_is_unknown():
    r = prefilter.run_prefilter(
        make_job("Benefits: minimum of 3 years of company tenure required "
                 "for full 401k vesting."), make_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"


def test_not_remote_onsite_elsewhere_fails():
    r = prefilter.run_prefilter(
        make_job("This role is NOT remote; onsite in Chicago is required, "
                 "in-person only.", location="Chicago, IL"), make_profile())
    assert outcome_of(r, "location") == "fail"


def test_waived_years_requirement_is_unknown():
    r = prefilter.run_prefilter(
        make_job("This position does NOT require 3 years of experience - "
                 "even entry-level candidates are welcome."), make_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"


def test_no_minimum_years_is_unknown():
    r = prefilter.run_prefilter(
        make_job("No minimum of 5 years experience is needed to apply."),
        make_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"


def test_hybrid_remote_onsite_is_unknown():
    r = prefilter.run_prefilter(
        make_job("Remote position, though onsite presence in our Austin "
                 "office is required 3 days/week.", location="Austin, TX"),
        make_profile())
    assert outcome_of(r, "location") == "unknown"


def test_years_requirement_after_the_number_is_caught():
    """The phrasing real postings actually use."""
    job = make_job(description_text="8+ years of experience required.")
    r = _rule_years(job, make_profile())
    assert r.outcome == "fail"
    assert "8" in r.extracted_value


def test_years_requirement_with_minimum_after_the_number():
    job = make_job(description_text="5 years of professional experience "
                                    "is a minimum for this role.")
    assert _rule_years(job, make_profile()).outcome == "fail"


def test_years_requirement_before_the_number_still_caught():
    job = make_job(description_text="Requires 8+ years of experience.")
    assert _rule_years(job, make_profile()).outcome == "fail"


def test_preferred_years_are_not_a_hard_requirement():
    """'Preferred' is not 'required' - the conservative principle applies."""
    job = make_job(description_text="5+ years of experience preferred.")
    assert _rule_years(job, make_profile()).outcome != "fail"


def test_negated_years_requirement_is_not_a_fail():
    job = make_job(description_text="This role does not require 3 years "
                                    "of experience.")
    assert _rule_years(job, make_profile()).outcome != "fail"


def test_sponsorship_refusal_fails_for_a_candidate_needing_it():
    p = make_profile(work_authorization="needs_sponsorship")
    job = make_job(description_text="We are unable to sponsor visas for "
                                    "this position.")
    r = _rule_authorization(job, p)
    assert r.outcome == "fail"


def test_sponsorship_refusal_does_not_fail_a_permanent_resident():
    """The rule must read the profile, not just the posting."""
    p = make_profile(work_authorization="permanent_resident")
    job = make_job(description_text="We are unable to sponsor visas for "
                                    "this position.")
    assert _rule_authorization(job, p).outcome != "fail"


def test_sponsorship_offered_is_not_a_fail():
    p = make_profile(work_authorization="needs_sponsorship")
    job = make_job(description_text="We are happy to sponsor visas.")
    assert _rule_authorization(job, p).outcome != "fail"


def test_clearance_still_fails_without_clearance():
    p = make_profile(work_authorization="permanent_resident")
    job = make_job(description_text="An active TS/SCI security clearance "
                                    "is required.")
    assert _rule_authorization(job, p).outcome == "fail"


def test_rules_catch_the_recorded_real_postings():
    """Regression net built from the 209 postings already collected."""
    for case in REAL_SNIPPETS:
        p = make_profile(work_authorization=case.get("work_authorization"))
        job = make_job(description_text=case["text"])
        rule = {"years_of_experience": _rule_years,
                "work_authorization": _rule_authorization}[case["rule"]]
        assert rule(job, p).outcome == case["expected"], case["text"][:80]
