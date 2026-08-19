import pytest
from pydantic import ValidationError

from conftest import _make_job
from offerpilot import cli
from offerpilot.labels import (
    ACTION_LABELS,
    FIT_LABELS,
    LABEL_SOURCES,
    REJECTION_REASONS,
    LabelInput,
)
from offerpilot.models import NormalizedJob
from offerpilot.store import db


def test_vocabularies_match_spec():
    assert FIT_LABELS == frozenset({"good_fit", "poor_fit", "uncertain"})
    assert ACTION_LABELS == frozenset({"apply", "skip", "save"})
    assert REJECTION_REASONS == frozenset({
        "skills", "seniority", "location", "compensation", "duplicate",
        "expired", "not_interested", "bad_draft", "other"})
    assert LABEL_SOURCES == frozenset({"review_feedback", "blind_eval"})


def test_label_input_rejects_unknown_vocabulary():
    with pytest.raises(ValidationError):
        LabelInput(fit_label="maybe")
    with pytest.raises(ValidationError):
        LabelInput(rejection_reason="vibes")


def test_record_label_requires_known_source(conn):
    _, vid = db.upsert_job(conn, _make_job())
    with pytest.raises(ValueError):
        db.record_label(conn, vid, label_source="hearsay", fit_label="good_fit")


def test_label_provenance_is_persisted_and_queryable(conn):
    _, vid = db.upsert_job(conn, _make_job())
    db.record_label(conn, vid, label_source="review_feedback",
                    fit_label="good_fit", action_label="apply")
    db.record_label(conn, vid, label_source="blind_eval", fit_label="poor_fit")
    blind = db.get_labels(conn, label_source="blind_eval")
    assert len(blind) == 1
    assert blind[0]["fit_label"] == "poor_fit"
    assert len(db.get_labels(conn, version_id=vid)) == 2


def test_upsert_companies_is_idempotent(conn):
    rows = [{"id": "acme", "name": "Acme"}, {"id": "globex", "name": "Globex"}]
    assert db.upsert_companies(conn, rows) == 2
    db.upsert_companies(conn, rows)
    assert conn.execute("SELECT COUNT(*) c FROM companies").fetchone()["c"] == 2


def test_migrate_is_idempotent_and_adds_edit_columns(conn):
    db.migrate(conn)
    db.migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(review_items)")}
    assert {"edited_brief_json", "edited_at"} <= cols
    assert "notes" in {r["name"] for r in conn.execute("PRAGMA table_info(labels)")}


def test_review_queue_only_returns_pending_review(conn):
    _, vid = db.upsert_job(conn, _make_job("1"))
    _, other = db.upsert_job(conn, _make_job("2"))
    for v in (vid, other):
        db.set_status(conn, v, "ready_for_match")
        db.set_status(conn, v, "matching")
    conn.execute("INSERT INTO review_items(job_version_id, match_json, "
                 "total_score) VALUES(?,?,?)", (vid, '{"eligibility":"pass"}', 72))
    conn.commit()
    db.set_status(conn, vid, "pending_review")
    db.set_status(conn, other, "scored_low")
    queue = db.get_review_queue(conn)
    assert [r["job_version_id"] for r in queue] == [vid]
    assert queue[0]["total_score"] == 72
    assert queue[0]["title"] == "SWE Intern"


def test_edited_brief_is_stored_separately_from_model_brief(conn):
    _, vid = db.upsert_job(conn, _make_job())
    conn.execute("INSERT INTO review_items(job_version_id, match_json, "
                 "total_score) VALUES(?,?,?)", (vid, "{}", 70))
    conn.commit()
    db.save_brief(conn, vid, '{"why_it_fits":"model"}')
    db.save_edited_brief(conn, vid, '{"why_it_fits":"human"}')
    row = db.get_review_item(conn, vid)
    assert row["brief_json"] == '{"why_it_fits":"model"}'
    assert row["edited_brief_json"] == '{"why_it_fits":"human"}'
    assert row["edited_at"] is not None


def test_blind_candidates_include_filtered_out_jobs(conn):
    _, kept = db.upsert_job(conn, _make_job("1"))
    _, dropped = db.upsert_job(conn, _make_job("2"))
    db.set_status(conn, kept, "ready_for_match")
    db.set_status(conn, dropped, "filtered_out")
    ids = {r["id"] for r in db.get_blind_candidates(conn)}
    assert {kept, dropped} <= ids


def test_blind_candidates_skip_already_blind_labeled(conn):
    _, vid = db.upsert_job(conn, _make_job())
    db.record_label(conn, vid, label_source="blind_eval", fit_label="good_fit")
    assert db.get_blind_candidates(conn, unlabeled_only=True) == []
    assert len(db.get_blind_candidates(conn, unlabeled_only=False)) == 1


def test_collect_populates_companies_table(conn, profile, monkeypatch):
    def fake_collect_company(company, cfg):
        return [NormalizedJob(
            source="greenhouse", external_id="9", company_id=company["id"],
            title="SWE Intern", location="Remote",
            url="https://x.co/9", canonical_url="https://x.co/9",
            description_text="Build agent tooling in Python.")]

    monkeypatch.setattr(cli, "_collect_company", fake_collect_company)
    cfg = {"companies": [{"id": "acme", "name": "Acme",
                          "ats": "greenhouse", "ats_slug": "acme"}]}
    cli.cmd_collect(conn, cfg, profile)
    row = conn.execute("SELECT name FROM companies WHERE id='acme'").fetchone()
    assert row is not None and row["name"] == "Acme"
