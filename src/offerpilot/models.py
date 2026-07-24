from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class NormalizedJob(BaseModel):
    source: Literal["greenhouse", "lever"]
    external_id: str
    company_id: str
    title: str
    location: str
    url: str
    canonical_url: str
    description_text: str
    posted_at: Optional[str] = None


class FilterResult(BaseModel):
    outcome: Literal["pass", "fail", "unknown"]
    rule: str
    extracted_value: Optional[str] = None
    reason: str


class EvidenceRef(BaseModel):
    source_id: str
    section: str = ""
    supporting_text: str


class MatchResult(BaseModel):
    eligibility: Literal["pass", "fail", "unknown"]
    eligibility_reasons: list[str] = []
    eligibility_evidence_excerpt: Optional[str] = None
    skills_score: int = Field(ge=0, le=30)
    project_score: int = Field(ge=0, le=20)
    domain_score: int = Field(ge=0, le=15)
    seniority_score: int = Field(ge=0, le=15)
    preference_score: int = Field(ge=0, le=20)
    evidence: list[EvidenceRef] = []
    gaps: list[str] = []
    uncertainties: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def fail_needs_excerpt(self):
        if self.eligibility == "fail" and not self.eligibility_evidence_excerpt:
            raise ValueError(
                "eligibility=fail requires eligibility_evidence_excerpt")
        return self


def total_score(m: MatchResult) -> int:
    return (m.skills_score + m.project_score + m.domain_score
            + m.seniority_score + m.preference_score)
