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


def test_upsert_companies_tolerates_malformed_entries(conn):
    """Config is user-authored YAML that nothing schema-validates.

    `name:` parses to None -- the key EXISTS, so a `.get(k, default)` default
    never fires -- and companies.name is NOT NULL.  An entry with no `id` at
    all must not abort the write for its well-formed siblings.
    """
    written = db.upsert_companies(conn, [
        {"id": "acme", "name": None},        # YAML `name:` -> None
        {"name": "No Id Here"},              # no id at all
        "just-a-string",                     # not even a mapping
        {"id": "globex", "name": "Globex"},
    ])
    assert written == 2                      # only the two usable entries
    stored = {r["id"]: r["name"]
              for r in conn.execute("SELECT id, name FROM companies")}
    assert stored == {"acme": "acme", "globex": "Globex"}


def test_migrate_is_idempotent_and_adds_edit_columns(conn):
    db.migrate(conn)
    db.migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(review_items)")}
    assert {"edited_brief_json", "edited_at"} <= cols
    assert "notes" in {r["name"] for r in conn.execute("PRAGMA table_info(labels)")}


def _scored(conn, ext, status, score):
    """A job version that owns a review_items row and ends in `status`."""
    _, vid = db.upsert_job(conn, _make_job(ext))
    db.set_status(conn, vid, "ready_for_match")
    db.set_status(conn, vid, "matching")
    conn.execute("INSERT INTO review_items(job_version_id, match_json, "
                 "total_score) VALUES(?,?,?)",
                 (vid, '{"eligibility":"pass"}', score))
    conn.commit()
    db.set_status(conn, vid, status)
    return vid


def test_review_queue_only_returns_pending_review(conn):
    """Every version here owns a review_items row.

    _REVIEW_SELECT is an INNER JOIN from review_items, so if only the pending
    rows had one the join alone would produce the expected answer and the
    status filter would be untested.  `decided` is the real-world case: the
    ALLOWED_TRANSITIONS pending_review -> approved keeps the review_items row,
    so a lost filter leaves already-decided items in the queue forever.
    """
    mid = _scored(conn, "1", "pending_review", 72)
    top = _scored(conn, "2", "pending_review", 85)
    tie = _scored(conn, "3", "pending_review", 72)
    low = _scored(conn, "4", "scored_low", 90)
    decided = _scored(conn, "5", "pending_review", 99)
    db.set_status(conn, decided, "approved")   # keeps its review_items row

    queue = db.get_review_queue(conn)

    # Ordering is score DESC, then job_version_id ASC as the tie-break; the two
    # excluded rows carry the highest scores, so losing the status filter both
    # lengthens the queue and puts the wrong row first.
    assert [r["job_version_id"] for r in queue] == [top, mid, tie]
    assert [r["total_score"] for r in queue] == [85, 72, 72]
    assert low not in {r["job_version_id"] for r in queue}
    assert decided not in {r["job_version_id"] for r in queue}
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
    def fake_collect_company(company):
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


def test_collect_survives_a_malformed_company_entry(conn, profile, monkeypatch):
    """One bad config entry must not cost the whole batch.

    Task B established per-company isolation around _collect_company; routing
    every entry through an unguarded pre-loop write to `companies` would
    reintroduce the same failure at whole-run granularity.
    """
    def fake_collect_company(company):
        return [NormalizedJob(
            source="greenhouse", external_id=f"job-{company['id']}",
            company_id=company["id"], title="SWE Intern", location="Remote",
            url="https://x.co/9", canonical_url="https://x.co/9",
            description_text="Build agent tooling in Python.")]

    monkeypatch.setattr(cli, "_collect_company", fake_collect_company)
    cfg = {"companies": [
        {"id": "acme", "name": None, "ats": "greenhouse", "ats_slug": "acme"},
        {"name": "No Id", "ats": "greenhouse", "ats_slug": "noid"},
        {"id": "globex", "name": "Globex", "ats": "greenhouse",
         "ats_slug": "globex"},
    ]}

    out = cli.cmd_collect(conn, cfg, profile)

    collected = {r["company_id"]
                 for r in conn.execute("SELECT company_id FROM jobs")}
    assert collected == {"acme", "globex"}   # healthy siblings still collected
    assert out["inserted"] == 2
    assert out["errors"] == 1                # the id-less entry is reported
