import re
from offerpilot.models import NormalizedJob, FilterResult
from offerpilot.profile import Profile

_YEARS_REQ = re.compile(
    r"(?:requires?|must have|minimum(?: of)?)\s+(\d+)\s*\+?\s*years?"
    r"[^.\n]{0,40}?(?:experience|\bexp\b)", re.I)
_CLEARANCE_TERM = re.compile(r"(?:security clearance|TS/SCI|top secret)", re.I)
_NEG_BEFORE = re.compile(r"(?:\bno|\bnot|\bwithout|n'?t)\s+(?:\w+\s+){0,3}$", re.I)
_NOT_REQ_AFTER = re.compile(r"^\s*(?:is\s+|are\s+)?not\s+required", re.I)
_REQ_AFTER = re.compile(r"^\s*(?:\w+\s+){0,3}?(?:required|must|mandatory)", re.I)
_REMOTE = re.compile(r"\bremote\b", re.I)
_NOT_REMOTE = re.compile(r"\b(?:not|isn'?t|no)\s+remote\b", re.I)
_ONSITE = re.compile(r"\b(?:onsite|on-site|in[- ]office|in[- ]person)\b", re.I)
_NEG_YEARS_BEFORE = re.compile(
    r"(?:\bdoes\s+not|\bdoesn'?t|\bdo\s+not|\bdon'?t|\bno\b|\bnot\b|\bwithout\b)"
    r"\s*(?:\w+\s+){0,2}$", re.I)


def _clearance_requirement(text: str) -> str | None:
    for m in _CLEARANCE_TERM.finditer(text):
        before = text[max(0, m.start() - 40):m.start()]
        after = text[m.end():m.end() + 40]
        if _NEG_BEFORE.search(before) or _NOT_REQ_AFTER.search(after):
            continue
        if _REQ_AFTER.search(after):
            return m.group(0)
    return None


def _rule_years(job: NormalizedJob, profile: Profile) -> FilterResult:
    text = job.description_text
    for m in _YEARS_REQ.finditer(text):
        before = text[max(0, m.start() - 30):m.start()]
        if _NEG_YEARS_BEFORE.search(before):
            continue
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
    term = _clearance_requirement(text)
    if term is not None:
        return FilterResult(outcome="fail", rule="work_authorization",
                           extracted_value=term,
                           reason="requires clearance candidate lacks")
    return FilterResult(outcome="unknown", rule="work_authorization",
                       reason="posting does not state a blocking requirement")


def _rule_location(job: NormalizedJob, profile: Profile) -> FilterResult:
    loc = job.location or ""
    text = job.description_text
    for ok in profile.constraints.locations:
        city = ok.split(",")[0].strip().lower()
        if city and city in loc.lower():
            return FilterResult(outcome="pass", rule="location",
                               extracted_value=loc, reason=f"matches {ok}")
    remote_mentioned = bool(_REMOTE.search(loc) or _REMOTE.search(text))
    if (profile.constraints.remote_ok and remote_mentioned
            and not _NOT_REMOTE.search(text)):
        if _ONSITE.search(text):
            return FilterResult(outcome="unknown", rule="location",
                               extracted_value=loc or None,
                               reason="mixed remote/onsite signals")
        return FilterResult(outcome="pass", rule="location",
                           extracted_value=loc, reason="remote allowed")
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
