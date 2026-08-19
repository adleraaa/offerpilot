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
    r"(?P<n>\d+)\s*(?:\+|\+?\s*(?:-|–|—|to)\s*\d+\s*\+?)?\s*years?", re.I)
_YEARS_NUMBER_FIRST = re.compile(
    r"(?P<n>\d+)\s*\+?\s*years?[^.\n]{0,60}?"
    r"(?:experience|\bexp\b)[^.\n]{0,40}?"
    r"(?:required|is required|are required|is a minimum|minimum|must have|"
    r"mandatory)", re.I)
_YEARS_PREFERRED = re.compile(
    r"(?:preferred|prefer\b|nice to have|a plus|bonus|ideally|ideal\b|"
    r"desirable|desired)", re.I)
# A section label frames every bullet under it, so whichever framing sits
# *nearest* the match governs: a later "Requirements:" beats an earlier "Nice
# to have:". Only labels qualify. "Minimum", "Required" and "at least" are
# quantifiers bound to the number they modify, not new sections — under an
# "Ideally you'd have:" header, "Minimum 8+ years experience" is still a
# preference, and "At least one of the following" is not a section at all.
_YEARS_SECTION_LABEL = re.compile(r"\b(?:requirements?|qualifications?)\b", re.I)
# Postings count years of many things: age minimums, licence tenure, degree
# recency, vesting, company history. Only a phrase the posting itself ties to
# the candidate's own experience may hard-reject; the rest stay `unknown`.
_EXPERIENCE_TOKEN = re.compile(r"\bexperience\b|\bexp\b", re.I)
# Kept to things a posting counts years *of*. "citizenship" and "residency" are
# deliberately absent: they co-occur with real experience bars in public-sector
# postings ("5+ years of experience and US citizenship required") and would
# suppress them.
_YEARS_NON_EXPERIENCE = re.compile(
    r"\b(?:tenure|vesting|vested|401\s*\(?k\)?|seniority|ago|age|old|"
    r"licen[cs]e|driving\s+record)\b", re.I)
# "Qualifications Required 8+ years in cybersecurity" is the one corpus shape
# that states an experience bar with no "experience" noun anywhere near it: an
# open-ended minimum bound to a field, directly under a requirement header.
_YEARS_IN_FIELD = re.compile(r"\s*(?:in|of|as)\s+\w", re.I)
_REQUIREMENT_HEADER = re.compile(
    r"(?:required|requires?|must\s+have|minimum|requirements?|qualifications?)"
    r"\b", re.I)
_EXPERIENCE_WINDOW = 60
_PREFERENCE_LOOKBACK = 200

_CLEARANCE_TERM = re.compile(
    r"(?:security clearance|secret clearance|TS/SCI|top secret)", re.I)
_NEG_BEFORE = re.compile(r"(?:\bno|\bnot|\bwithout|n'?t)\s+(?:\w+\s+){0,3}$", re.I)
_NOT_REQ_AFTER = re.compile(
    r"^\s*(?:is\s+|are\s+)?not\s+(?:required|necessary|needed|mandatory)", re.I)
# The words between the term and the requirement word are captured so a
# cancelling one can veto the match ("clearance preferred but not required").
_REQ_AFTER = re.compile(
    r"^\s*(?P<gap>(?:\w+\s+){0,3}?)(?:required|must|mandatory)", re.I)
# "This role will require at minimum an active Secret clearance" states the
# requirement before the term rather than after it. The gap may not cross
# clause punctuation: a comma or semicolon means the requirement word belongs
# to something else ("A degree is required, and a clearance is a plus").
_REQ_BEFORE = re.compile(
    r"(?:requires?|required|must\s+(?:have|hold|possess)|need)\b[^.,;\n]{0,40}$",
    re.I)
_PREFERENCE_OR_NEGATION = re.compile(
    r"\b(?:preferred|prefer|preference|nice\s+to\s+have|a\s+plus|bonus|"
    r"ideally|desirable|desired|not|no|without)\b", re.I)

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
        req_after = _REQ_AFTER.search(after)
        if (req_after is not None
                and not _PREFERENCE_OR_NEGATION.search(req_after.group("gap"))):
            return m.group(0)
        if _REQ_BEFORE.search(before):
            return m.group(0)
    return None


def _last_match_end(pattern: re.Pattern[str], text: str, lo: int,
                    hi: int) -> int | None:
    end = None
    for m in pattern.finditer(text, lo, hi):
        end = m.end()
    return end


def _is_preference(text: str, start: int, end: int) -> bool:
    """True when the phrasing around a years match marks it as a preference.

    Backwards the scan crosses sentence boundaries, because a header like
    "Ideally you'd have:" governs every bullet under it and stripping HTML
    turns each bullet into its own sentence. To keep that from over-reaching,
    the nearest framing wins: a "Requirements:" section label between the
    preference word and the match hands the match back to the requirement.
    The label scan includes the match itself, because the match usually opens
    with that label ("Requirements: 5+ years").

    Forwards it stops at the end of the current clause, because a later clause
    ("..., ideally in customer-facing contexts") qualifies something else.
    """
    lo = max(0, start - _PREFERENCE_LOOKBACK)
    pref = _last_match_end(_YEARS_PREFERRED, text, lo, start)
    if pref is not None:
        label = _last_match_end(_YEARS_SECTION_LABEL, text, lo, end)
        if label is None or pref > label:
            return True
    hi = min(len(text), end + 60)
    for i in range(end, hi):
        if text[i] in ",;.\n":
            hi = i
            break
    return bool(_YEARS_PREFERRED.search(text, end, hi))


def _sentence_window(text: str, start: int, end: int, radius: int) -> str:
    """The text around a match, clipped to the sentence it sits in."""
    lo = max(0, start - radius)
    for boundary in (".", "\n"):
        idx = text.rfind(boundary, lo, start)
        if idx != -1:
            lo = max(lo, idx + 1)
    hi = min(len(text), end + radius)
    for boundary in (".", "\n"):
        idx = text.find(boundary, end, hi)
        if idx != -1:
            hi = min(hi, idx)
    return text[lo:hi]


def _is_about_experience(text: str, m: re.Match[str]) -> bool:
    """True when the posting ties this years phrase to the candidate's experience.

    Required because the anchor alternation includes bare section labels, so
    "Requirements: Must be at least 18 years of age" and "valid driver's
    license held for at least 3 years" both match the pattern while being no
    evidence at all of a years-of-experience bar.
    """
    window = _sentence_window(text, m.start(), m.end(), _EXPERIENCE_WINDOW)
    if _YEARS_NON_EXPERIENCE.search(window):
        return False
    if _EXPERIENCE_TOKEN.search(window):
        return True
    return ("+" in m.group(0)
            and _REQUIREMENT_HEADER.match(m.group(0)) is not None
            and _YEARS_IN_FIELD.match(text, m.end()) is not None)


def _rule_years(job: NormalizedJob, profile: Profile) -> FilterResult:
    text = job.description_text
    for pattern in (_YEARS_ANCHOR_FIRST, _YEARS_NUMBER_FIRST):
        for m in pattern.finditer(text):
            before = text[max(0, m.start() - 30):m.start()]
            if _NEG_YEARS_BEFORE.search(before):
                continue
            if not _is_about_experience(text, m):
                continue
            if _is_preference(text, m.start(), m.end()):
                continue
            years = int(m.group("n"))
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
