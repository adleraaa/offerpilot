import re
from offerpilot.models import NormalizedJob, FilterResult
from offerpilot.profile import Profile

# Real postings state a years requirement on either side of the number
# ("Requires 8+ years of experience" / "8+ years of experience required"),
# and often anchor it on a section label ("Requirements:", "Qualifications")
# rather than on a verb. Both orders are tried.
_YEARS_ANCHOR_FIRST = re.compile(
    r"(?:required|requires?|must\s+have|minimum(?:\s+of)?|at\s+least|"
    r"requirements?|qualifications?)\b[^.\n]{0,40}?"
    r"(\d+)\s*(?:\+|\+?\s*(?:-|–|—|to)\s*\d+\s*\+?)?\s*years?", re.I)
_YEARS_NUMBER_FIRST = re.compile(
    r"(\d+)\s*\+?\s*years?[^.\n]{0,60}?"
    r"(?:experience|\bexp\b)[^.\n]{0,40}?"
    r"(?:required|is required|are required|is a minimum|minimum|must have|"
    r"mandatory)", re.I)
_YEARS_PREFERRED = re.compile(
    r"(?:preferred|prefer\b|nice to have|a plus|bonus|ideally|ideal\b|"
    r"desirable|desired)", re.I)
# The anchor-first pattern deliberately does not demand the word "experience"
# ("Required 8+ years in cybersecurity"), so years counted for something other
# than the candidate's own experience are excluded explicitly.
_YEARS_NON_EXPERIENCE = re.compile(
    r"\b(?:tenure|vesting|vested|401\s*\(?k\)?|seniority|ago)\b", re.I)
_PREFERENCE_LOOKBACK = 200

_CLEARANCE_TERM = re.compile(
    r"(?:security clearance|secret clearance|TS/SCI|top secret)", re.I)
_NEG_BEFORE = re.compile(r"(?:\bno|\bnot|\bwithout|n'?t)\s+(?:\w+\s+){0,3}$", re.I)
_NOT_REQ_AFTER = re.compile(r"^\s*(?:is\s+|are\s+)?not\s+required", re.I)
_REQ_AFTER = re.compile(r"^\s*(?:\w+\s+){0,3}?(?:required|must|mandatory)", re.I)
# "This role will require at minimum an active Secret clearance" states the
# requirement before the term rather than after it.
_REQ_BEFORE = re.compile(
    r"(?:requires?|required|must\s+(?:have|hold|possess)|need)\b[^.\n]{0,40}$",
    re.I)

_NO_SPONSORSHIP = re.compile(
    r"(?:not|unable to|cannot|will not|do(?:es)? not)\s+(?:\w+\s+){0,3}"
    r"(?:sponsor|provide sponsorship|offer sponsorship)"
    r"|no\s+(?:visa\s+)?sponsorship"
    r"|sponsorship\s+is\s+not\s+(?:available|offered|provided)", re.I)
_NEEDS_SPONSORSHIP = frozenset({"needs_sponsorship", "f1_opt", "f1", "h1b",
                                "requires_sponsorship"})

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
        if _REQ_AFTER.search(after) or _REQ_BEFORE.search(before):
            return m.group(0)
    return None


def _is_preference(text: str, start: int, end: int) -> bool:
    """True when the phrasing around a years match marks it as a preference.

    Backwards, the scan runs to the start of the sentence (bounded), because a
    header like "Ideally you'd have:" governs every item that follows it.
    Forwards, it stops at the end of the current clause, because a later clause
    ("..., ideally in customer-facing contexts") qualifies something else.
    """
    lo = max(0, start - _PREFERENCE_LOOKBACK)
    for boundary in (".", "\n"):
        idx = text.rfind(boundary, lo, start)
        if idx != -1:
            lo = idx + 1
    if _YEARS_PREFERRED.search(text, lo, start):
        return True
    hi = min(len(text), end + 60)
    for i in range(end, hi):
        if text[i] in ",;.\n":
            hi = i
            break
    return bool(_YEARS_PREFERRED.search(text, end, hi))


def _rule_years(job: NormalizedJob, profile: Profile) -> FilterResult:
    text = job.description_text
    for pattern in (_YEARS_ANCHOR_FIRST, _YEARS_NUMBER_FIRST):
        for m in pattern.finditer(text):
            before = text[max(0, m.start() - 30):m.start()]
            if _NEG_YEARS_BEFORE.search(before):
                continue
            if _YEARS_NON_EXPERIENCE.search(text[m.end():m.end() + 60]):
                continue
            if _is_preference(text, m.start(), m.end()):
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
    auth = (profile.constraints.work_authorization or "").lower()

    term = _clearance_requirement(text)
    if term is not None:
        return FilterResult(outcome="fail", rule="work_authorization",
                           extracted_value=term,
                           reason="requires clearance candidate lacks")

    # Asymmetric on purpose: a posting refusing to sponsor only blocks a
    # candidate who needs sponsorship. The same sentence is irrelevant to a
    # citizen or a permanent resident.
    if auth in _NEEDS_SPONSORSHIP:
        m = _NO_SPONSORSHIP.search(text)
        if m is not None:
            return FilterResult(outcome="fail", rule="work_authorization",
                               extracted_value=m.group(0),
                               reason="employer will not sponsor and the "
                                      "candidate requires sponsorship")

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
