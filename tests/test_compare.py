"""Running v1 and v2 on the same blind-labeled jobs."""

import json
import os

from offerpilot import compare, evalset
from offerpilot.models import EvidenceRef, MatchResult, NormalizedJob
from offerpilot.profile import load_profile
from offerpilot.store import db


class _TitleLLM:
    """Scores 85 for an intern title, 20 otherwise; never fails eligibility."""

    def __init__(self):
        self.calls = 0

    def structured(self, *, node, run_id, system, user, schema, validate=None):
        self.calls += 1
        high = "Intern" in user
        r = MatchResult(
            eligibility="pass",
            skills_score=25 if high else 5, project_score=18 if high else 5,
            domain_score=12 if high else 5, seniority_score=12 if high else 5,
            preference_score=18 if high else 0,
            evidence=[EvidenceRef(source_id="sample_project",
                                  supporting_text="x")],
            confidence=0.8)
        if validate is not None:
            validate(r)
        return r


def _job(ext, title, text):
    return NormalizedJob(source="greenhouse", external_id=str(ext),
                         company_id="acme", title=title, location="Remote",
                         url=f"https://x.co/{ext}",
                         canonical_url=f"https://x.co/{ext}",
                         description_text=text)


def _eval_db(tmp_path):
    path = str(tmp_path / "eval.db")
    conn = db.connect(path)
    db.init_schema(conn)
    grad = db.upsert_job(conn, _job(1, "SWE Intern",
                                    "Open to students graduating in 2026."))[1]
    plain = db.upsert_job(conn, _job(2, "SWE Intern", "Build tools."))[1]
    senior = db.upsert_job(conn, _job(3, "Staff Engineer", "Lead teams."))[1]
    evalset.record_consensus(conn, {grad: "good_fit", plain: "good_fit",
                                    senior: "poor_fit"}, note="test")
    conn.close()
    return path


def _cfg():
    return {"match": {"score_threshold": 60, "max_auto_retries": 3},
            "brief": {"enabled": False}, "eval": {"precision_at": [5]}}


def test_comparison_runs_each_version_on_its_own_copy(tmp_path):
    src = _eval_db(tmp_path)
    profile = load_profile("profile.example.yaml")
    out = compare.run_comparison(src, _cfg(), profile,
                                 make_llm=lambda conn: _TitleLLM(),
                                 versions=("v1", "v2"),
                                 out_dir=str(tmp_path / "runs"))
    v1, v2 = out["v1"]["classification"], out["v2"]["classification"]
    # v1's prefilter drops the graduation-window job; v2 lets it through.
    assert (v1["tp"], v1["fn"]) == (1, 1)
    assert (v2["tp"], v2["fn"]) == (2, 0)
    assert out["v1"]["prefilter_false_negatives"] == 1
    assert out["v2"]["prefilter_false_negatives"] == 0
    assert out["v1"]["pipeline_version"] == "v1"
    for v in ("v1", "v2"):
        assert os.path.exists(out[v]["written_to"])
        assert json.load(open(out[v]["written_to"]))["pipeline_version"] == v

    # The labeled source database is never written to.
    conn = db.connect(src)
    assert {r["status"] for r in conn.execute(
        "SELECT status FROM job_versions")} == {"new"}


def test_reset_clears_a_previous_run_but_keeps_labels(tmp_path):
    src = _eval_db(tmp_path)
    conn = db.connect(src)
    profile = load_profile("profile.example.yaml")
    compare.reset_versions(conn, profile, "v1")
    compare.reset_versions(conn, profile, "v2")
    rules = conn.execute("SELECT COUNT(*) n FROM filter_results").fetchone()["n"]
    assert rules == 3 * 6          # one prefilter pass, not two
    assert len(db.get_labels(conn, label_source="blind_eval")) == 3
    assert {r["status"] for r in conn.execute(
        "SELECT status FROM job_versions")} == {"ready_for_match"}
