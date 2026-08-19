import json

from conftest import REPO_ROOT

from offerpilot.models import NormalizedJob
from offerpilot.profile import load_profile
from offerpilot import prefilter
from offerpilot.prefilter import (_rule_authorization,
                                  _rule_graduation_window,
                                  _rule_pay_floor, _rule_years)

REAL_SNIPPETS = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "real_posting_snippets.json")
    .read_text(encoding="utf-8"))


def make_job(description_text, location="New York, NY"):
    return NormalizedJob(source="lever", external_id="1", company_id="c",
                         title="Engineer", location=location,
                         url="https://x.co/1", canonical_url="https://x.co/1",
                         description_text=description_text)


def make_profile(work_authorization=None, graduation=None,
                 pay_floor_hourly_usd=None):
    p = load_profile("profile.example.yaml")
    if work_authorization is not None:
        p.constraints.work_authorization = work_authorization
    if graduation is not None:
        p.identity.graduation = graduation
    if pay_floor_hourly_usd is not None:
        p.constraints.pay_floor_hourly_usd = pay_floor_hourly_usd
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


# --- Review findings: the years rule must not reject on years that are not
# --- evidence about the candidate's own experience.

def test_age_minimum_is_not_an_experience_requirement():
    for text in ("Requirements: Must be at least 18 years of age and legally "
                 "able to work.",
                 "You must be at least 21 years of age to apply."):
        assert _rule_years(make_job(text), make_profile()).outcome != "fail", text


def test_licence_and_record_tenure_are_not_experience():
    for text in ("Requirements: valid driver's license held for at least "
                 "3 years.",
                 "Minimum Qualifications: must have a clean driving record "
                 "for 5 years."):
        assert _rule_years(make_job(text), make_profile()).outcome != "fail", text


def test_degree_recency_window_is_not_an_experience_requirement():
    """Actively inverted before the fix: this posting is recent-grad friendly."""
    job = make_job("This is an entry-level role. Requirements: BS degree "
                   "within the last 3 years.")
    assert _rule_years(job, make_profile()).outcome != "fail"


def test_years_offered_as_one_alternative_is_not_a_hard_requirement():
    job = make_job("At least one of the following: 3 years in Python OR a "
                   "relevant degree.")
    assert _rule_years(job, make_profile()).outcome != "fail"


def test_open_ended_years_in_a_field_under_a_requirement_header_still_fails():
    """The corpus shape that states an experience bar without the noun."""
    for text in ("Qualifications Required 8+ years in cybersecurity, with a "
                 "record of leading engineers.",
                 "Must-Have Qualifications 6+ years in product management."):
        assert _rule_years(make_job(text), make_profile()).outcome == "fail", text


def test_preference_header_governs_across_a_sentence_break():
    """The real posting puts an unrelated sentence between header and bullet."""
    job = make_job("Ideally you'd have: JD and a member of the California Bar "
                   "in good standing. At least 9+ years of combined law firm "
                   "and in-house experience.")
    assert _rule_years(job, make_profile()).outcome != "fail"


def test_a_later_requirement_header_overrides_an_earlier_preference_header():
    job = make_job("Nice to have: Docker. Requirements: 5+ years of "
                   "professional experience.")
    assert _rule_years(job, make_profile()).outcome == "fail"


# --- Review findings: a requirement word about something else must not bind
# --- to a clearance term further along the same sentence.

def test_unrelated_requirement_does_not_bind_the_clearance():
    for text in ("A degree is required, and a security clearance is a plus.",
                 "Experience in classified environments required; an active "
                 "security clearance is preferred.",
                 "US work authorization is required, though a security "
                 "clearance is not necessary."):
        r = _rule_authorization(make_job(text), make_profile())
        assert r.outcome != "fail", text


def test_clearance_preferred_but_not_required_is_not_a_fail():
    job = make_job("Security clearance preferred but not required.")
    assert _rule_authorization(job, make_profile()).outcome != "fail"


def test_clearance_required_before_the_term_still_fails():
    for text in ("Must have: At least an active TS/SCI clearance and the "
                 "ability to up level to CI Poly.",
                 "This role will require at minimum an active Secret "
                 "clearance and willingness to obtain a TS/SCI clearance.",
                 "This role will require an active security clearance"):
        r = _rule_authorization(make_job(text), make_profile())
        assert r.outcome == "fail", text


def test_clearance_cancelled_by_a_synonym_of_not_required_is_not_a_fail():
    """A requirement word can legitimately bind inside the clause and still be
    cancelled: the veto must not key on the single word "required"."""
    for text in ("We require US citizenship but a security clearance is not "
                 "needed.",
                 "You must have strong Python skills and a security clearance "
                 "is not necessary."):
        r = _rule_authorization(make_job(text), make_profile())
        assert r.outcome != "fail", text


# --- Spec rules 5 and 6: graduation window and pay floor. The conservative
# --- principle binds both: only a stated requirement that definitely excludes
# --- the candidate may `fail`; ambiguity and negation stay `unknown`.

def test_graduation_window_excluding_candidate_year_fails():
    p = make_profile(graduation="2029-05")
    job = make_job("Open to the class of 2025 and 2026 only.")
    r = _rule_graduation_window(job, p)
    assert r.outcome == "fail"
    assert "2025" in r.extracted_value


def test_graduation_window_including_candidate_year_passes():
    p = make_profile(graduation="2029-05")
    job = make_job("For students graduating in 2029.")
    assert _rule_graduation_window(job, p).outcome == "pass"


def test_graduation_range_spanning_candidate_year_passes():
    p = make_profile(graduation="2029-05")
    job = make_job("Graduating between 2027 and 2030.")
    assert _rule_graduation_window(job, p).outcome == "pass"


def test_no_graduation_statement_is_unknown():
    p = make_profile(graduation="2029-05")
    job = make_job("We build distributed systems.")
    assert _rule_graduation_window(job, p).outcome == "unknown"


def test_graduation_negation_is_unknown_not_fail():
    p = make_profile(graduation="2029-05")
    job = make_job("No graduation year restriction; we hired someone from "
                   "the class of 2024.")
    assert _rule_graduation_window(job, p).outcome == "unknown"


def test_an_including_window_anywhere_prevents_the_fail():
    """Contradictory windows are ambiguity, and ambiguity may not reject."""
    p = make_profile(graduation="2029-05")
    job = make_job("Class of 2026 preferred. We also welcome students "
                   "graduating in 2029.")
    assert _rule_graduation_window(job, p).outcome != "fail"


def test_a_year_in_a_neighbouring_sentence_is_not_a_graduation_window():
    """Marketing copy puts unrelated years next to the word 'graduating'."""
    p = make_profile(graduation="2029-05")
    job = make_job("Founded in 2015. We have hired graduating students from "
                   "every background.")
    assert _rule_graduation_window(job, p).outcome == "unknown"


def test_graduation_rule_is_unknown_when_the_profile_year_is_unparseable():
    p = make_profile(graduation="sometime soon")
    job = make_job("Open to the class of 2025 and 2026 only.")
    assert _rule_graduation_window(job, p).outcome == "unknown"


def test_hourly_rate_below_floor_fails():
    p = make_profile(pay_floor_hourly_usd=20)
    job = make_job("Compensation: $14 - $16 per hour.")
    r = _rule_pay_floor(job, p)
    assert r.outcome == "fail"
    assert r.extracted_value is not None


def test_hourly_range_top_above_floor_passes():
    p = make_profile(pay_floor_hourly_usd=20)
    job = make_job("Pay: $18 - $28/hr depending on experience.")
    assert _rule_pay_floor(job, p).outcome == "pass"


def test_annual_salary_is_converted_at_2080_hours():
    p = make_profile(pay_floor_hourly_usd=20)
    low = make_job("Salary: $30,000 - $35,000 per year.")
    high = make_job("Salary: $90,000 - $120,000 annually.")
    assert _rule_pay_floor(low, p).outcome == "fail"
    assert _rule_pay_floor(high, p).outcome == "pass"


def test_equity_or_unparseable_pay_is_unknown():
    p = make_profile(pay_floor_hourly_usd=20)
    job = make_job("Competitive salary and equity.")
    assert _rule_pay_floor(job, p).outcome == "unknown"


def test_bare_dollar_amount_without_unit_is_unknown():
    p = make_profile(pay_floor_hourly_usd=20)
    job = make_job("We raised $12 million in Series A funding.")
    assert _rule_pay_floor(job, p).outcome == "unknown"


def test_the_highest_stated_rate_governs_the_pay_floor():
    """A low figure elsewhere in the posting may not sink a qualifying range."""
    p = make_profile(pay_floor_hourly_usd=20)
    job = make_job("Interns start at $15/hr; this role pays $32 per hour.")
    assert _rule_pay_floor(job, p).outcome == "pass"


def test_run_prefilter_returns_all_six_rules():
    p = make_profile()
    names = {r.rule for r in prefilter.run_prefilter(make_job("Anything"), p)}
    assert names == {"years_of_experience", "work_authorization", "location",
                     "excluded_company", "graduation_window", "pay_floor"}


def test_an_hour_phrasing_is_read_as_a_rate():
    """'$22 an hour' is the same statement as '$22/hr'."""
    p = make_profile(pay_floor_hourly_usd=20)
    assert _rule_pay_floor(make_job("Pay is $22 an hour."), p).outcome == "pass"
    assert _rule_pay_floor(make_job("Pay is $12 an hour."), p).outcome == "fail"
