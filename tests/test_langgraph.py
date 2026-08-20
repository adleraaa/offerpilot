"""The pipeline as a compiled LangGraph, and its parity with Week 1.

`tests/test_graph.py` is the parity suite: every assertion in it was written
against the hand-rolled function pipeline and must keep passing unchanged.
This file covers what is new -- that the graph is a real compiled
`StateGraph`, that the gate routes to the brief node only when a job is
actually heading for review, and that losing the brief never costs the match.
"""

import pytest

from offerpilot.brief import ApplicationBrief
from offerpilot.graph import build_match_graph, run_match_for_version
from offerpilot.models import EvidenceRef, MatchResult
from offerpilot.store import db
from tests.conftest import _ready_row


def test_graph_is_a_compiled_langgraph_with_expected_nodes():
    g = build_match_graph()
    nodes = set(g.get_graph().nodes)
    assert {"match", "brief", "persist"} <= nodes
    assert type(g).__module__.startswith("langgraph")


def test_high_score_run_visits_brief_and_records_both_steps(conn, profile,
                                                            scoring_llm):
    row = _ready_row(conn)
    final = run_match_for_version(conn, scoring_llm(90), profile, row,
                                  threshold=60, max_auto_retries=3,
                                  brief_enabled=True)
    assert final == "pending_review"
    nodes = [r["node"] for r in conn.execute(
        "SELECT node FROM run_steps ORDER BY id")]
    assert nodes == ["match", "brief"]
    item = db.get_review_item(conn, row["id"])
    assert item["brief_json"] is not None
    assert ApplicationBrief.model_validate_json(item["brief_json"])


def test_low_score_run_skips_brief(conn, profile, scoring_llm):
    row = _ready_row(conn)
    final = run_match_for_version(conn, scoring_llm(10), profile, row,
                                  threshold=60, max_auto_retries=3,
                                  brief_enabled=True)
    assert final == "scored_low"
    nodes = [r["node"] for r in conn.execute("SELECT node FROM run_steps")]
    assert nodes == ["match"]


def test_brief_can_be_disabled_without_changing_status(conn, profile,
                                                       scoring_llm):
    row = _ready_row(conn)
    final = run_match_for_version(conn, scoring_llm(90), profile, row,
                                  threshold=60, max_auto_retries=3,
                                  brief_enabled=False)
    assert final == "pending_review"
    assert db.get_review_item(conn, row["id"])["brief_json"] is None
    assert [r["node"] for r in conn.execute(
        "SELECT node FROM run_steps")] == ["match"]


def test_brief_failure_still_leaves_the_job_reviewable(conn, profile,
                                                       scoring_llm):
    """A brief is a nice-to-have; losing it must not lose the match."""
    llm = scoring_llm(90)
    llm.fail_node = "brief"
    row = _ready_row(conn)
    final = run_match_for_version(conn, llm, profile, row, threshold=60,
                                  max_auto_retries=3, brief_enabled=True)
    assert final == "pending_review"
    item = db.get_review_item(conn, row["id"])
    assert item["brief_json"] is None
    steps = {r["node"]: r["status"] for r in conn.execute(
        "SELECT node, status FROM run_steps")}
    assert steps["brief"] == "brief_failed"
    assert conn.execute("SELECT status FROM runs").fetchone()["status"] == "ok"


def test_cmd_match_generates_briefs_by_default(conn, profile, scoring_llm):
    """`brief_enabled` is opt-in on the low-level entry point, so the default
    that actually ships is the CLI's: config `brief.enabled`, default true.
    A config with no `brief:` section must still produce a brief."""
    from offerpilot.cli import cmd_match

    row = _ready_row(conn)
    cfg = {"match": {"score_threshold": 60, "max_auto_retries": 3}}
    assert cmd_match(conn, cfg, profile, scoring_llm(90)) == {
        "pending_review": 1}
    assert db.get_review_item(conn, row["id"])["brief_json"] is not None


def test_cmd_match_honours_brief_disabled_in_config(conn, profile,
                                                    scoring_llm):
    from offerpilot.cli import cmd_match

    row = _ready_row(conn)
    cfg = {"match": {"score_threshold": 60, "max_auto_retries": 3},
           "brief": {"enabled": False}}
    cmd_match(conn, cfg, profile, scoring_llm(90))
    assert db.get_review_item(conn, row["id"])["brief_json"] is None


def test_ungrounded_high_score_never_reaches_the_brief_node(conn, profile):
    """The gate refuses to spend a second call on a result it will reject.

    A client that drops `validate` can hand back a high score citing an id
    that does not exist. The graph rejects it (see tests/test_graph.py), but
    the rejection happens in `persist`, downstream of the brief -- so the
    routing gate has to re-check grounding too, or every hallucinated match
    would buy the model a free brief written off invented evidence.
    """
    calls = []

    class IgnoresValidate:
        def structured(self, **kwargs):
            calls.append(kwargs["node"])
            return MatchResult(
                eligibility="pass", skills_score=30, project_score=20,
                domain_score=15, seniority_score=15, preference_score=20,
                evidence=[EvidenceRef(source_id="made_up_project",
                                      supporting_text="x")],
                confidence=0.9)

    row = _ready_row(conn)
    final = run_match_for_version(conn, IgnoresValidate(), profile, row,
                                  threshold=60, max_auto_retries=3,
                                  brief_enabled=True)
    assert final == "permanent_error"
    assert calls == ["match"]
    assert conn.execute("SELECT COUNT(*) c FROM review_items"
                        ).fetchone()["c"] == 0
    # The version's status and the run's status are set by two different
    # writers -- `db.set_status` in the node, `_finish_run` from the graph's
    # returned `run_status` -- so asserting only the first leaves the second
    # free to file a rejected run as "ok".
    assert conn.execute("SELECT status FROM runs").fetchone()["status"] == \
        "permanent_error"


def test_run_meta_still_reaches_the_runs_row(conn, profile, scoring_llm):
    """The graph rewrite must not drop provenance recorded by Task 3."""
    row = _ready_row(conn)
    run_match_for_version(conn, scoring_llm(90), profile, row, threshold=60,
                          max_auto_retries=3, brief_enabled=True,
                          run_meta={"git_commit": "abc123",
                                    "config_hash": "def456"})
    run = conn.execute("SELECT git_commit, config_hash FROM runs").fetchone()
    assert run["git_commit"] == "abc123"
    assert run["config_hash"] == "def456"


@pytest.mark.parametrize("exc_name", ["AuthLLMError", "PermanentLLMError"])
def test_auth_error_is_still_caught_before_permanent(conn, profile, exc_name):
    """AuthLLMError subclasses PermanentLLMError; ordering decides whether a
    bad key burns the job's attempt budget."""
    import offerpilot.llm as llm_mod

    exc = getattr(llm_mod, exc_name)

    class Boom:
        def structured(self, **kwargs):
            raise exc("boom")

    row = _ready_row(conn)
    if exc_name == "AuthLLMError":
        with pytest.raises(llm_mod.AuthLLMError):
            run_match_for_version(conn, Boom(), profile, row, threshold=60,
                                  max_auto_retries=3)
        expected = ("ready_for_match", 0, "auth_error")
    else:
        assert run_match_for_version(conn, Boom(), profile, row, threshold=60,
                                     max_auto_retries=3) == "permanent_error"
        expected = ("permanent_error", 1, "permanent_error")
    after = conn.execute("SELECT status, attempt_count FROM job_versions "
                         "WHERE id=?", (row["id"],)).fetchone()
    assert (after["status"], after["attempt_count"]) == expected[:2]
    assert conn.execute("SELECT status FROM runs").fetchone()["status"] == \
        expected[2]


def test_a_writer_that_steals_the_row_mid_call_makes_the_run_abandon(
        conn, profile, scoring_llm):
    """A run that loses a race abandons the write and is filed `stale_state`.

    No transaction is held across the LLM call -- that is a global constraint,
    not an accident -- so between `set_status(..., "matching")` and the write
    in `persist` another writer can move the version. The double below stands
    in for whoever does that: a `sweep_stale_matching` pass, a second process,
    a human resetting a row by hand.

    `persist` must then leave their status alone and hand back *their* status
    as `final_status`, which is why `run_status` is a separate key: the run
    did not do what it was asked, so filing it as "ok" would hide the race
    from the trace. Without the guard this is not merely mislabelled -- the
    node would insert a review_item and then raise ValueError out of
    `db.set_status`, because `scored_low -> pending_review` is not in
    `db.ALLOWED_TRANSITIONS`.

    `brief_enabled` stays off: the brief is a second call to the same double,
    and this test is about the third node, not the second.
    """
    inner = scoring_llm(90)
    row = _ready_row(conn)

    class StealsTheRowMidCall:
        def structured(self, **kwargs):
            result = inner.structured(**kwargs)
            conn.execute("UPDATE job_versions SET status='scored_low' "
                         "WHERE id=?", (row["id"],))
            conn.commit()
            return result

    final = run_match_for_version(conn, StealsTheRowMidCall(), profile, row,
                                  threshold=60, max_auto_retries=3)

    assert final == "scored_low"
    assert conn.execute("SELECT status FROM runs").fetchone()["status"] == \
        "stale_state"
    steps = {(r["node"], r["status"]) for r in conn.execute(
        "SELECT node, status FROM run_steps")}
    assert ("gate", "stale_state") in steps
    assert conn.execute("SELECT COUNT(*) c FROM review_items"
                        ).fetchone()["c"] == 0
    assert conn.execute("SELECT status FROM job_versions WHERE id=?",
                        (row["id"],)).fetchone()["status"] == "scored_low"
