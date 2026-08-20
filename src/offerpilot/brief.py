"""The application-brief node: one structured call, grounded the same way.

The brief is the second (and last) LLM output in the pipeline, and it is the
one a human actually reads before applying, so it is held to the match node's
rules rather than looser ones: every id it cites has to exist in
`profile.yaml`, and the checks live in code, not only in the prompt.
"""

from typing import Literal, Optional

from pydantic import BaseModel

from offerpilot.models import EvidenceRef, MatchResult, total_score
from offerpilot.profile import Profile
from offerpilot.prompts import BRIEF_SYSTEM, BRIEF_USER

Theme = Literal["why_this_role", "relevant_project", "main_strength",
                "gap_to_address"]


class TalkingPoint(BaseModel):
    theme: Theme
    point: str
    evidence_source_id: str
    # True for everything this milestone produces: OfferPilot never collects
    # an employer's actual application questions (spec hard boundary 1), so a
    # point cannot be tailored to one. The field exists so the reader is told
    # that, and `make_brief_validator` enforces it.
    generic: bool = True


class ApplicationBrief(BaseModel):
    why_it_fits: str
    cited_evidence: list[EvidenceRef] = []
    main_gaps: list[str] = []
    resume_bullets_to_emphasize: list[str] = []
    talking_points: list[TalkingPoint] = []
    outreach_paragraph: Optional[str] = None


def build_brief_prompts(job_row, profile: Profile, match: MatchResult):
    """Trusted profile and prior match first, untrusted posting last."""
    from offerpilot.graph import _sanitize  # single delimiter-stripping impl

    user = BRIEF_USER.format(
        profile_json=profile.model_dump_json(indent=2),
        total=total_score(match),
        skills=match.skills_score, projects=match.project_score,
        domain=match.domain_score, seniority=match.seniority_score,
        preferences=match.preference_score,
        eligibility=match.eligibility,
        gaps="; ".join(match.gaps) or "none recorded",
        uncertainties="; ".join(match.uncertainties) or "none recorded",
        title=_sanitize(job_row["title"]),
        location=_sanitize(job_row["location"] or ""),
        description=_sanitize(job_row["description_text"]))
    return BRIEF_SYSTEM, user


def make_brief_validator(profile: Profile):
    """Reject a semantically wrong ApplicationBrief by raising ValueError.

    Handed to `LLMClient.structured(validate=...)`, so a rejection buys the
    model a repair turn inside the client's 3-attempt budget. Messages are
    written to be read by the model: name the mistake, name the legal
    alternative.
    """
    valid = profile.experience_ids()

    def _validate(brief: ApplicationBrief) -> None:
        bad = sorted(
            {e.source_id for e in brief.cited_evidence if e.source_id not in valid}
            | {tp.evidence_source_id for tp in brief.talking_points
               if tp.evidence_source_id not in valid})
        if bad:
            raise ValueError(
                f"evidence source_id {bad} do not exist. Valid ids are: "
                f"{sorted(valid)}. Cite only these.")
        # generic=false asserts the point was written against the employer's
        # own application questions. Nothing in this system collects those, so
        # the assertion is always false, and a brief that carries it would
        # tell its reader something untrue. The prompt asks for true; asking
        # is not enforcement.
        ungrounded = sorted({tp.theme for tp in brief.talking_points
                             if not tp.generic})
        if ungrounded:
            raise ValueError(
                f'talking points {ungrounded} set "generic": false. No '
                f"application questions were collected for this job, so every "
                f'talking point must set "generic": true.')

    return _validate


def generate_brief(llm, run_id, job_row, profile: Profile,
                   match: MatchResult) -> ApplicationBrief:
    system, user = build_brief_prompts(job_row, profile, match)
    return llm.structured(node="brief", run_id=run_id, system=system,
                          user=user, schema=ApplicationBrief,
                          validate=make_brief_validator(profile))
