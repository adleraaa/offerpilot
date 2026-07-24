import pytest
from offerpilot.models import NormalizedJob, FilterResult
from offerpilot.store import db


def make_job(**over):
    base = dict(source="greenhouse", external_id="123",
                company_id="examplecorp", title="AI Intern",
                location="New York, NY", url="https://x.co/j/123?utm_source=a",
                canonical_url="https://x.co/j/123",
                description_text="Do AI things.")
    base.update(over)
    return NormalizedJob(**base)


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_schema(c)
    return c


def test_insert_creates_job_and_version(conn):
    job_id, ver_id = db.upsert_job(conn, make_job())
    assert job_id is not None and ver_id is not None
    row = conn.execute("SELECT status FROM job_versions WHERE id=?",
                       (ver_id,)).fetchone()
    assert row["status"] == "new"


def test_same_content_no_new_version(conn):
    _, v1 = db.upsert_job(conn, make_job())
    _, v2 = db.upsert_job(conn, make_job())
    assert v1 is not None and v2 is None


def test_changed_content_new_version(conn):
    _, v1 = db.upsert_job(conn, make_job())
    _, v2 = db.upsert_job(conn, make_job(description_text="Now different."))
    assert v2 is not None and v2 != v1


def test_legal_and_illegal_transitions(conn):
    _, v = db.upsert_job(conn, make_job())
    db.set_status(conn, v, "ready_for_match")
    db.set_status(conn, v, "matching")
    db.set_status(conn, v, "retryable_error")
    db.set_status(conn, v, "ready_for_match")  # retry path
    with pytest.raises(ValueError):
        db.set_status(conn, v, "approved")     # not from ready_for_match


def test_stale_sweep_resets_matching(conn):
    _, v = db.upsert_job(conn, make_job())
    db.set_status(conn, v, "ready_for_match")
    db.set_status(conn, v, "matching")
    conn.execute(
        "UPDATE job_versions SET processing_started_at="
        "datetime('now','-30 minutes') WHERE id=?", (v,))
    conn.commit()
    assert db.sweep_stale_matching(conn) == 1
    row = conn.execute("SELECT status FROM job_versions WHERE id=?",
                       (v,)).fetchone()
    assert row["status"] == "ready_for_match"


def test_filter_results_persist(conn):
    _, v = db.upsert_job(conn, make_job())
    db.record_filter_results(conn, v, [FilterResult(
        outcome="unknown", rule="work_authorization", reason="not stated")])
    n = conn.execute("SELECT COUNT(*) c FROM filter_results "
                     "WHERE job_version_id=?", (v,)).fetchone()["c"]
    assert n == 1


def test_get_versions_by_status_filters_and_orders(conn):
    _, v1 = db.upsert_job(conn, make_job(external_id="a", url="https://x.co/j/a", canonical_url="https://x.co/j/a"))
    _, v2 = db.upsert_job(conn, make_job(external_id="b", url="https://x.co/j/b", canonical_url="https://x.co/j/b"))
    db.set_status(conn, v1, "ready_for_match")
    rows = db.get_versions_by_status(conn, "ready_for_match")
    assert [r["id"] for r in rows] == [v1]
    rows_new = db.get_versions_by_status(conn, "new")
    assert [r["id"] for r in rows_new] == [v2]


def test_sweep_leaves_fresh_matching_rows(conn):
    _, v = db.upsert_job(conn, make_job())
    db.set_status(conn, v, "ready_for_match")
    db.set_status(conn, v, "matching")
    assert db.sweep_stale_matching(conn) == 0
    row = conn.execute("SELECT status FROM job_versions WHERE id=?", (v,)).fetchone()
    assert row["status"] == "matching"
