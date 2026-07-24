import re
from offerpilot.models import NormalizedJob, FilterResult
from offerpilot.profile import Profile

_YEARS_REQ = re.compile(
    r"(?:requires?|must have|minimum(?: of)?)\s+(\d+)\s*\+?\s*years?", re.I)
_YEARS_ANY = re.compile(r"(\d+)\s*\+?\s*years?", re.I)
_CLEARANCE = re.compile(
    r"(?:security clearance|TS/SCI|top secret)[^.]*?(?:required|must)", re.I)
_CLEARANCE_ANY = re.compile(
    r"(?:active|current)?\s*(?:TS/SCI|security clearance)\s*(?:required)", re.I)
_REMOTE = re.compile(r"\bremote\b", re.I)
_ONSITE = re.compile(r"\b(?:onsite|on-site|in[- ]office|in[- ]person)\b", re.I)


def _rule_years(job: NormalizedJob, profile: Profile) -> FilterResult:
    m = _YEARS_REQ.search(job.description_text)
    if m:
        years = int(m.group(1))
        if years >= 3:
            return FilterResult(outcome="fail", rule="years_of_experience",
                               extracted_value=m.group(0),
                               reason=f"explicitly requires {years}+ years")
        return FilterResult(outcome="pass", rule="years_of_experience",
                           extracted_value=m.group(0),
                           reason="requirement within reach")
    return FilterResult(outcome="unknown", rule="years_of_experience",
                       reason="no explicit requirement parsed")


def _rule_authorization(job: NormalizedJob, profile: Profile) -> FilterResult:
    text = job.description_text
    if _CLEARANCE.search(text) or _CLEARANCE_ANY.search(text):
        return FilterResult(outcome="fail", rule="work_authorization",
                           extracted_value="security clearance required",
                           reason="requires clearance candidate lacks")
    return FilterResult(outcome="unknown", rule="work_authorization",
                       reason="posting does not state a blocking requirement")


def _rule_location(job: NormalizedJob, profile: Profile) -> FilterResult:
    loc = job.location or ""
    text = job.description_text
    if profile.constraints.remote_ok and (_REMOTE.search(loc)
                                          or _REMOTE.search(text)):
        return FilterResult(outcome="pass", rule="location",
                           extracted_value=loc, reason="remote allowed")
    for ok in profile.constraints.locations:
        city = ok.split(",")[0].strip().lower()
        if city and city in loc.lower():
            return FilterResult(outcome="pass", rule="location",
                               extracted_value=loc, reason=f"matches {ok}")
    if loc and _ONSITE.search(text):
        return FilterResult(outcome="fail", rule="location",
                           extracted_value=loc,
                           reason="explicitly onsite outside allowed locations")
    return FilterResult(outcome="unknown", rule="location",
                       extracted_value=loc or None,
                       reason="location not conclusively incompatible")


def _rule_excluded(job: NormalizedJob, profile: Profile) -> FilterResult:
    if job.company_id in profile.constraints.excluded_companies:
        return FilterResult(outcome="fail", rule="excluded_company",
                           extracted_value=job.company_id,
                           reason="company on exclusion list")
    return FilterResult(outcome="pass", rule="excluded_company",
                       reason="not excluded")


RULES = [_rule_years, _rule_authorization, _rule_location, _rule_excluded]


def run_prefilter(job: NormalizedJob, profile: Profile) -> list[FilterResult]:
    return [rule(job, profile) for rule in RULES]


def decide(results: list[FilterResult]) -> str:
    if any(r.outcome == "fail" for r in results):
        return "filtered_out"
    return "ready_for_match"
