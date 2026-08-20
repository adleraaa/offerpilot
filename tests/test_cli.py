import pytest
from offerpilot.models import NormalizedJob, MatchResult, EvidenceRef
from offerpilot.store import db
from offerpilot.profile import load_profile
from offerpilot import cli


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_schema(c)
    return c


def seed(conn, desc="Build agents in New York, NY.", ext="1"):
    job = NormalizedJob(source="lever", external_id=ext, company_id="c",
                        title="AI Intern", location="New York, NY",
                        url=f"https://x.co/{ext}",
                        canonical_url=f"https://x.co/{ext}",
                        description_text=desc)
    return db.upsert_job(conn, job)


def test_collect_applies_prefilter(conn, monkeypatch):
    def fake_collect_company(company):
        return [NormalizedJob(
            source="greenhouse", external_id="9", company_id=company["id"],
            title="Staff Engineer", location="San Francisco, CA",
            url="https://x.co/9", canonical_url="https://x.co/9",
            description_text="Requires 8+ years. Onsite in San Francisco.")]

    monkeypatch.setattr(cli, "_collect_company", fake_collect_company)
    cfg = {"companies": [{"id": "c", "ats": "greenhouse", "ats_slug": "c"}]}
    profile = load_profile("profile.example.yaml")
    out = cli.cmd_collect(conn, cfg, profile)
    assert out["inserted"] == 1
    assert cli.cmd_status(conn).get("filtered_out") == 1


def test_match_command_processes_ready(conn):
    _, v = seed(conn)
    db.set_status(conn, v, "ready_for_match")

    class StubLLM:
        def structured(self, **kwargs):
            return MatchResult(
                eligibility="pass", skills_score=25, project_score=18,
                domain_score=12, seniority_score=12, preference_score=18,
                evidence=[EvidenceRef(source_id="sample_project",
                                      supporting_text="x")],
                confidence=0.8)

    cfg = {"match": {"score_threshold": 60, "max_auto_retries": 3}}
    profile = load_profile("profile.example.yaml")
    out = cli.cmd_match(conn, cfg, profile, StubLLM())
    assert out == {"pending_review": 1}


def test_status_counts(conn):
    seed(conn, ext="1")
    seed(conn, ext="2")
    assert cli.cmd_status(conn) == {"new": 2}


def test_retry_resets_permanent(conn, profile):
    _, v = seed(conn)
    db.set_status(conn, v, "ready_for_match")
    db.set_status(conn, v, "matching")
    db.set_status(conn, v, "permanent_error")
    assert cli.cmd_retry(conn, profile)["reset"] == 1
    assert cli.cmd_status(conn) == {"ready_for_match": 1}


def test_retry_recovers_a_row_parked_at_retryable_error(conn, profile,
                                                        monkeypatch):
    """`retryable_error` was a state with a legal exit nothing ever took.

    `run_match_for_version` writes `retryable_error` and then `ready_for_match`
    as two separately committed transitions. Kill the batch in between -- Ctrl-C
    on a long `offerpilot match`, an OOM, a dropped connection -- and the row
    stops at `retryable_error` for good. `sweep_stale_matching` only looks at
    `matching`, `sweep_stuck_new` only at `new`, and `cmd_retry` only reset
    `permanent_error`, so the job was parked in a state `ALLOWED_TRANSITIONS`
    lets it out of and no command ever asked it to leave: invisible to the
    queue, unreachable by `match`, and not counted as an error anyone would
    look at.

    The interruption is simulated rather than hand-set so the state is reached
    the way it is reached in production, not just written into the row.
    """
    from offerpilot import graph
    from offerpilot.llm import RetryableLLMError
    from tests.test_graph import StubLLM

    job = NormalizedJob(source="lever", external_id="1", company_id="c",
                        title="AI Intern", location="New York, NY",
                        url="https://x.co/1", canonical_url="https://x.co/1",
                        description_text="Build agents in New York, NY.")
    _, v = db.upsert_job(conn, job)
    db.set_status(conn, v, "ready_for_match")
    row = conn.execute("SELECT * FROM job_versions WHERE id=?", (v,)).fetchone()

    real_set_status = db.set_status

    def die_mid_reset(c, version_id, status):
        current = c.execute("SELECT status FROM job_versions WHERE id=?",
                            (version_id,)).fetchone()["status"]
        if current == "retryable_error" and status == "ready_for_match":
            raise KeyboardInterrupt("batch killed between the two commits")
        return real_set_status(c, version_id, status)

    monkeypatch.setattr(db, "set_status", die_mid_reset)
    with pytest.raises(KeyboardInterrupt):
        graph.run_match_for_version(conn, StubLLM(exc=RetryableLLMError("429")),
                                    profile, row, threshold=60,
                                    max_auto_retries=3)
    monkeypatch.setattr(db, "set_status", real_set_status)
    assert cli.cmd_status(conn) == {"retryable_error": 1}, (
        "the interrupted run should have left the row parked at "
        "retryable_error; if it did not, this test no longer tests anything")

    out = cli.cmd_retry(conn, profile)
    assert out["reset"] == 1
    assert cli.cmd_status(conn) == {"ready_for_match": 1}
    assert conn.execute("SELECT attempt_count FROM job_versions WHERE id=?",
                        (v,)).fetchone()["attempt_count"] == 0


def test_collect_isolates_per_job_failures(conn, monkeypatch):
    def fake_collect_company(company):
        return [
            NormalizedJob(source="greenhouse", external_id="ok1",
                          company_id=company["id"], title="A",
                          location="New York, NY", url="https://x.co/ok1",
                          canonical_url="https://x.co/ok1",
                          description_text="Build agents."),
            NormalizedJob(source="greenhouse", external_id="boom",
                          company_id=company["id"], title="B",
                          location="New York, NY", url="https://x.co/boom",
                          canonical_url="https://x.co/boom",
                          description_text="Explodes."),
            NormalizedJob(source="greenhouse", external_id="ok2",
                          company_id=company["id"], title="C",
                          location="New York, NY", url="https://x.co/ok2",
                          canonical_url="https://x.co/ok2",
                          description_text="Build more agents."),
        ]

    real_prefilter = cli.prefilter.run_prefilter

    def exploding_prefilter(job, profile):
        if job.external_id == "boom":
            raise RuntimeError("boom")
        return real_prefilter(job, profile)

    monkeypatch.setattr(cli, "_collect_company", fake_collect_company)
    monkeypatch.setattr(cli.prefilter, "run_prefilter", exploding_prefilter)
    cfg = {"companies": [{"id": "c", "ats": "greenhouse", "ats_slug": "c"}]}
    profile = load_profile("profile.example.yaml")
    out = cli.cmd_collect(conn, cfg, profile)
    assert out["errors"] == 1
    statuses = cli.cmd_status(conn)
    assert statuses.get("ready_for_match", 0) == 2
    assert statuses.get("new", 0) == 1


def test_match_limit_stops_after_n_jobs(conn, profile, scoring_llm):
    from offerpilot.cli import cmd_match
    from conftest import _ready_row
    for i in range(5):
        _ready_row(conn, str(i))
    cfg = {"match": {"score_threshold": 60, "max_auto_retries": 3}}
    counts = cmd_match(conn, cfg, profile, scoring_llm(90), limit=2)
    assert sum(counts.values()) == 2


def test_one_malformed_response_does_not_kill_the_batch(conn, profile):
    """A single bad job must not cost us the other 190."""
    from offerpilot.cli import cmd_match
    from conftest import _ready_row
    from offerpilot.models import MatchResult, EvidenceRef

    class FlakyLLM:
        def __init__(self):
            self.n = 0

        def structured(self, *, node, run_id, system, user, schema,
                       validate=None):
            self.n += 1
            if self.n == 1:
                raise AttributeError("'NoneType' object has no attribute "
                                     "'prompt_tokens'")
            return MatchResult(
                eligibility="pass", skills_score=30, project_score=20,
                domain_score=15, seniority_score=15, preference_score=20,
                evidence=[EvidenceRef(source_id="pathpilot",
                                      supporting_text="x")],
                confidence=0.9)

    for i in range(3):
        _ready_row(conn, str(i))
    cfg = {"match": {"score_threshold": 60, "max_auto_retries": 3}}
    counts = cmd_match(conn, cfg, profile, FlakyLLM())
    assert counts.get("pending_review") == 2


def test_config_fallback_warns(capsys, tmp_path, monkeypatch):
    from offerpilot.config import load_config
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.example.yaml").write_text("llm: {}\n", encoding="utf-8")
    load_config("config.yaml")
    assert "config.example.yaml" in capsys.readouterr().out


def test_config_strict_mode_refuses_to_fall_back(tmp_path, monkeypatch):
    from offerpilot.config import load_config
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.example.yaml").write_text("llm: {}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config("config.yaml", strict=True)


def test_retry_uses_the_state_machine(conn, profile, monkeypatch):
    """permanent_error -> ready_for_match must be a legal, recorded transition.

    Spying on ``db.set_status`` is the point of this test: a raw
    ``UPDATE job_versions SET status='ready_for_match' WHERE
    status='permanent_error'`` leaves an identical row behind, so only the
    call itself distinguishes the state machine from a bypass.
    """
    from offerpilot.cli import cmd_retry
    from offerpilot.store import db as dbmod
    from offerpilot.store.db import ALLOWED_TRANSITIONS
    assert "ready_for_match" in ALLOWED_TRANSITIONS["permanent_error"]
    from conftest import _ready_row
    row = _ready_row(conn)
    dbmod.set_status(conn, row["id"], "matching")
    dbmod.set_status(conn, row["id"], "permanent_error")
    conn.execute("UPDATE job_versions SET attempt_count=3 WHERE id=?",
                 (row["id"],))
    conn.commit()

    seen = []
    real = dbmod.set_status
    monkeypatch.setattr(dbmod, "set_status",
                        lambda c, v, st: (seen.append(st), real(c, v, st))[1])

    assert cmd_retry(conn, profile)["reset"] == 1
    assert seen == ["ready_for_match"]      # went through the state machine
    r = conn.execute("SELECT status, attempt_count FROM job_versions "
                     "WHERE id=?", (row["id"],)).fetchone()
    assert r["status"] == "ready_for_match" and r["attempt_count"] == 0


def test_run_records_git_commit_and_config_hash(conn, profile, scoring_llm):
    from offerpilot.graph import run_match_for_version
    from conftest import _ready_row
    row = _ready_row(conn)
    run_match_for_version(conn, scoring_llm(90), profile, row, threshold=60,
                          max_auto_retries=3,
                          run_meta={"git_commit": "abc123",
                                    "config_hash": "def456"})
    r = conn.execute("SELECT git_commit, config_hash FROM runs").fetchone()
    assert r["git_commit"] == "abc123" and r["config_hash"] == "def456"


def test_match_aborts_the_whole_batch_on_an_auth_error(conn, profile, capsys):
    """A rejected key must cost one call, not one call per job."""
    from offerpilot.cli import cmd_match
    from offerpilot.llm import AuthLLMError
    from conftest import _ready_row

    class BadKeyLLM:
        def __init__(self):
            self.calls = 0

        def structured(self, **kwargs):
            self.calls += 1
            raise AuthLLMError("401 invalid api key")

    for i in range(3):
        _ready_row(conn, str(i))
    llm = BadKeyLLM()
    cfg = {"match": {"score_threshold": 60, "max_auto_retries": 3}}
    counts = cmd_match(conn, cfg, profile, llm)
    assert llm.calls == 1
    assert counts.get("error", 0) == 0
    assert cli.cmd_status(conn) == {"ready_for_match": 3}
    assert "credential" in capsys.readouterr().out.lower()


def test_retry_prefilters_orphaned_new_versions(conn, profile):
    """A crash between upsert_job and prefilter leaves nothing to recover."""
    from offerpilot.cli import cmd_retry
    seed(conn, ext="orphan")
    out = cmd_retry(conn, profile)
    assert out["orphans_prefiltered"] == 1
    assert cli.cmd_status(conn).get("new", 0) == 0
