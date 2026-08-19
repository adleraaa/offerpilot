import pathlib

import pytest

from offerpilot.models import EvidenceRef, MatchResult, NormalizedJob
from offerpilot.profile import Profile
from offerpilot.store import db

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Tests read profile.example.yaml etc. by relative path; anchor cwd."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_schema(c)
    return c


@pytest.fixture
def profile():
    return Profile(
        identity={"name": "Alex Doe", "education": "B.S. CS",
                  "graduation": "2029-05"},
        constraints={"locations": ["New York, NY"], "remote_ok": True,
                     "pay_floor_hourly_usd": 20,
                     "work_authorization": "permanent_resident",
                     "employment_types": ["internship"]},
        skills={"languages": ["Python"]},
        experiences=[{"id": "pathpilot", "title": "PathPilot",
                      "summary": "LLM app", "skills": ["Python"]}])


def _make_job(ext="1"):
    return NormalizedJob(
        source="greenhouse", external_id=ext, company_id="acme",
        title="SWE Intern", location="Remote",
        url="https://boards.greenhouse.io/acme/jobs/1",
        canonical_url="https://boards.greenhouse.io/acme/jobs/1",
        description_text="Build agent tooling in Python.")


def _ready_row(conn, ext="1"):
    _, vid = db.upsert_job(conn, _make_job(ext))
    db.set_status(conn, vid, "ready_for_match")
    return conn.execute("SELECT * FROM job_versions WHERE id=?",
                        (vid,)).fetchone()


@pytest.fixture
def scoring_llm():
    from offerpilot.llm import PermanentLLMError

    class ScoringLLM:
        def __init__(self, total):
            self.total = total
            self.fail_node = None

        def structured(self, *, node, run_id, system, user, schema,
                       validate=None):
            if node == self.fail_node:
                raise PermanentLLMError(f"{node} exploded")
            if node == "match":
                per = self.total
                result = MatchResult(
                    eligibility="pass", skills_score=min(30, per),
                    project_score=min(20, max(0, per - 30)),
                    domain_score=min(15, max(0, per - 50)),
                    seniority_score=min(15, max(0, per - 65)),
                    preference_score=min(20, max(0, per - 80)),
                    evidence=[EvidenceRef(source_id="pathpilot",
                                          supporting_text="x")],
                    confidence=0.8)
            else:
                # Imported lazily: offerpilot.brief only exists from Task 4 on,
                # and the match branch above must work before then.
                from offerpilot.brief import ApplicationBrief
                result = ApplicationBrief(
                    why_it_fits="fits", cited_evidence=[],
                    main_gaps=[], resume_bullets_to_emphasize=[],
                    talking_points=[], outreach_paragraph=None)
            if validate is not None:
                validate(result)
            return result

    return ScoringLLM
