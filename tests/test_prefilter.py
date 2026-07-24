from offerpilot.models import NormalizedJob
from offerpilot.profile import load_profile
from offerpilot import prefilter


def make_job(desc, location="New York, NY"):
    return NormalizedJob(source="lever", external_id="1", company_id="c",
                         title="Engineer", location=location,
                         url="https://x.co/1", canonical_url="https://x.co/1",
                         description_text=desc)


def get_profile():
    return load_profile("profile.example.yaml")


def outcome_of(results, rule):
    return next(r.outcome for r in results if r.rule == rule)


def test_explicit_years_requirement_fails():
    r = prefilter.run_prefilter(
        make_job("Requires 8+ years of professional experience."),
        get_profile())
    assert outcome_of(r, "years_of_experience") == "fail"
    assert prefilter.decide(r) == "filtered_out"


def test_preferred_years_is_unknown_not_fail():
    r = prefilter.run_prefilter(
        make_job("5+ years preferred but not required."), get_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"
    assert prefilter.decide(r) == "ready_for_match"


def test_no_mention_is_unknown():
    r = prefilter.run_prefilter(make_job("We build agents."), get_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"
    assert outcome_of(r, "work_authorization") == "unknown"


def test_clearance_requirement_fails_authorization():
    r = prefilter.run_prefilter(
        make_job("Active TS/SCI security clearance required."), get_profile())
    assert outcome_of(r, "work_authorization") == "fail"


def test_location_conflict_fails_when_onsite_elsewhere():
    r = prefilter.run_prefilter(
        make_job("This role is onsite in San Francisco, CA.",
                 location="San Francisco, CA"), get_profile())
    assert outcome_of(r, "location") == "fail"


def test_remote_location_passes():
    r = prefilter.run_prefilter(
        make_job("Fully remote role.", location="Remote"), get_profile())
    assert outcome_of(r, "location") == "pass"


def test_excluded_company():
    p = get_profile()
    p.constraints.excluded_companies = ["c"]
    r = prefilter.run_prefilter(make_job("Anything"), p)
    assert outcome_of(r, "excluded_company") == "fail"


def test_no_clearance_required_is_unknown():
    r = prefilter.run_prefilter(
        make_job("No security clearance required for this role."),
        get_profile())
    assert outcome_of(r, "work_authorization") == "unknown"


def test_clearance_not_required_is_unknown():
    r = prefilter.run_prefilter(
        make_job("Security clearance is not required."), get_profile())
    assert outcome_of(r, "work_authorization") == "unknown"


def test_vesting_years_is_unknown():
    r = prefilter.run_prefilter(
        make_job("Benefits: minimum of 3 years of company tenure required "
                 "for full 401k vesting."), get_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"


def test_not_remote_onsite_elsewhere_fails():
    r = prefilter.run_prefilter(
        make_job("This role is NOT remote; onsite in Chicago is required, "
                 "in-person only.", location="Chicago, IL"), get_profile())
    assert outcome_of(r, "location") == "fail"


def test_waived_years_requirement_is_unknown():
    r = prefilter.run_prefilter(
        make_job("This position does NOT require 3 years of experience - "
                 "even entry-level candidates are welcome."), get_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"


def test_no_minimum_years_is_unknown():
    r = prefilter.run_prefilter(
        make_job("No minimum of 5 years experience is needed to apply."),
        get_profile())
    assert outcome_of(r, "years_of_experience") == "unknown"


def test_hybrid_remote_onsite_is_unknown():
    r = prefilter.run_prefilter(
        make_job("Remote position, though onsite presence in our Austin "
                 "office is required 3 days/week.", location="Austin, TX"),
        get_profile())
    assert outcome_of(r, "location") == "unknown"
