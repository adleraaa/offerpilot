import yaml
from pydantic import BaseModel


class Identity(BaseModel):
    name: str
    education: str
    graduation: str


class Constraints(BaseModel):
    locations: list[str]
    remote_ok: bool
    pay_floor_hourly_usd: float
    work_authorization: str
    employment_types: list[str]
    excluded_companies: list[str] = []


class Skills(BaseModel):
    languages: list[str] = []
    frameworks: list[str] = []
    ai_ml: list[str] = []


class Experience(BaseModel):
    id: str
    title: str
    summary: str
    skills: list[str] = []


class Profile(BaseModel):
    identity: Identity
    constraints: Constraints
    skills: Skills
    experiences: list[Experience]

    def experience_ids(self) -> set[str]:
        return {e.id for e in self.experiences}


def load_profile(path: str) -> Profile:
    with open(path, encoding="utf-8") as f:
        return Profile(**yaml.safe_load(f))
