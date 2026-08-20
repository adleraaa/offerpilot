import json
import pathlib
import re

import pytest
from fastapi.testclient import TestClient

from offerpilot.panel.app import create_app
from offerpilot.store import db

STATIC = (pathlib.Path(__file__).resolve().parents[1] / "src" / "offerpilot"
          / "panel" / "static")


@pytest.fixture
def seeded(tmp_path):
    path = str(tmp_path / "p.db")
    conn = db.connect(path)
    db.init_schema(conn)
    from tests.conftest import _make_job
    _, vid = db.upsert_job(conn, _make_job("1"))
    db.set_status(conn, vid, "ready_for_match")
    db.set_status(conn, vid, "matching")
    match = {"eligibility": "unknown", "eligibility_reasons": ["unclear"],
             "eligibility_evidence_excerpt": None, "skills_score": 25,
             "project_score": 15, "domain_score": 10, "seniority_score": 10,
             "preference_score": 15,
             "evidence": [{"source_id": "pathpilot", "section": "",
                           "supporting_text": "built an LLM app"}],
             "gaps": ["no k8s"], "uncertainties": [], "confidence": 0.8}
    conn.execute("INSERT INTO review_items(job_version_id, match_json, "
                 "total_score, brief_json) VALUES(?,?,?,?)",
                 (vid, json.dumps(match), 75,
                  json.dumps({"why_it_fits": "fits", "cited_evidence": [],
                              "main_gaps": [], "resume_bullets_to_emphasize": [],
                              "talking_points": [], "outreach_paragraph": None})))
    conn.commit()
    db.set_status(conn, vid, "pending_review")
    conn.close()
    return path, vid


@pytest.fixture
def client(seeded, profile):
    path, _ = seeded
    return TestClient(create_app(path, profile))


def test_queue_lists_pending_items_with_scores(client, seeded):
    _, vid = seeded
    body = client.get("/api/queue").json()
    assert [i["job_version_id"] for i in body["items"]] == [vid]
    assert body["items"][0]["total_score"] == 75
    assert body["items"][0]["title"] == "SWE Intern"


def test_item_detail_exposes_evidence_and_unresolved_eligibility(client, seeded):
    _, vid = seeded
    body = client.get(f"/api/item/{vid}").json()
    assert body["eligibility_unresolved"] is True
    assert body["match"]["evidence"][0]["source_id"] == "pathpilot"
    assert body["brief"]["why_it_fits"] == "fits"
    assert body["job"]["description_text"]


def test_missing_item_is_404(client):
    assert client.get("/api/item/9999").status_code == 404


def test_approve_writes_review_feedback_label_and_moves_status(client, seeded):
    path, vid = seeded
    r = client.post(f"/api/item/{vid}/decision", json={
        "action": "approve", "fit_label": "good_fit", "action_label": "apply"})
    assert r.status_code == 200
    conn = db.connect(path)
    assert conn.execute("SELECT status FROM job_versions WHERE id=?",
                        (vid,)).fetchone()["status"] == "approved"
    label = db.get_labels(conn, version_id=vid)[0]
    assert label["label_source"] == "review_feedback"
    assert label["fit_label"] == "good_fit"
    conn.close()


def test_reject_requires_a_reason(client, seeded):
    _, vid = seeded
    bad = client.post(f"/api/item/{vid}/decision",
                      json={"action": "reject", "fit_label": "poor_fit"})
    assert bad.status_code == 422
    ok = client.post(f"/api/item/{vid}/decision",
                     json={"action": "reject", "fit_label": "poor_fit",
                           "rejection_reason": "seniority"})
    assert ok.status_code == 200


def test_a_refused_reject_does_not_move_the_status_or_write_a_label(client,
                                                                   seeded):
    """422 must be a no-op: the row stays reviewable and un-labelled."""
    path, vid = seeded
    client.post(f"/api/item/{vid}/decision",
                json={"action": "reject", "fit_label": "poor_fit"})
    conn = db.connect(path)
    assert conn.execute("SELECT status FROM job_versions WHERE id=?",
                        (vid,)).fetchone()["status"] == "pending_review"
    assert db.get_labels(conn, version_id=vid) == []
    conn.close()


def test_unknown_vocabulary_is_rejected(client, seeded):
    _, vid = seeded
    r = client.post(f"/api/item/{vid}/decision",
                    json={"action": "approve", "fit_label": "sort_of"})
    assert r.status_code == 422


def test_illegal_transition_returns_409_not_500(client, seeded):
    _, vid = seeded
    client.post(f"/api/item/{vid}/decision",
                json={"action": "approve", "fit_label": "good_fit"})
    again = client.post(f"/api/item/{vid}/decision",
                        json={"action": "approve", "fit_label": "good_fit"})
    assert again.status_code == 409


def test_a_refused_transition_does_not_write_a_second_label(client, seeded):
    """The 409 path must not leave a label behind for a decision that failed."""
    path, vid = seeded
    for _ in range(2):
        client.post(f"/api/item/{vid}/decision",
                    json={"action": "approve", "fit_label": "good_fit"})
    conn = db.connect(path)
    assert len(db.get_labels(conn, version_id=vid)) == 1
    conn.close()


def test_editing_the_brief_preserves_the_model_original(client, seeded):
    path, vid = seeded
    edited = {"why_it_fits": "my own words", "cited_evidence": [],
              "main_gaps": [], "resume_bullets_to_emphasize": [],
              "talking_points": [], "outreach_paragraph": None}
    assert client.put(f"/api/item/{vid}/brief",
                      json={"brief": edited}).status_code == 200
    conn = db.connect(path)
    row = db.get_review_item(conn, vid)
    assert json.loads(row["brief_json"])["why_it_fits"] == "fits"
    assert json.loads(row["edited_brief_json"])["why_it_fits"] == "my own words"
    conn.close()


def test_the_detail_view_prefers_the_edited_brief(client, seeded):
    _, vid = seeded
    edited = {"why_it_fits": "my own words", "cited_evidence": [],
              "main_gaps": [], "resume_bullets_to_emphasize": [],
              "talking_points": [], "outreach_paragraph": None}
    client.put(f"/api/item/{vid}/brief", json={"brief": edited})
    body = client.get(f"/api/item/{vid}").json()
    assert body["brief"]["why_it_fits"] == "my own words"
    assert body["brief_is_edited"] is True


def test_edited_brief_must_validate_against_the_schema(client, seeded):
    _, vid = seeded
    r = client.put(f"/api/item/{vid}/brief", json={"brief": {"nope": 1}})
    assert r.status_code == 422


def test_trace_returns_run_steps(client, seeded):
    path, vid = seeded
    conn = db.connect(path)
    run_id = conn.execute(
        "INSERT INTO runs(run_type, job_version_id, status) "
        "VALUES('match', ?, 'ok') RETURNING id", (vid,)).fetchone()["id"]
    conn.execute("INSERT INTO run_steps(run_id, node, attempt, status) "
                 "VALUES(?,'match',1,'ok')", (run_id,))
    conn.execute("INSERT INTO run_steps(run_id, node, attempt, status) "
                 "VALUES(?,'brief',1,'ok')", (run_id,))
    conn.commit()
    conn.close()
    body = client.get(f"/api/trace/{vid}").json()
    assert [s["node"] for s in body["steps"]] == ["match", "brief"]


def test_panel_javascript_never_uses_innerHTML():
    """Job text is untrusted; the panel must render it as text, not markup."""
    scripts = list(STATIC.glob("*.js"))
    assert scripts, "no panel JavaScript found to check"
    for js in scripts:
        source = js.read_text(encoding="utf-8")
        assert "innerHTML" not in source, f"{js.name} uses innerHTML"
        assert "outerHTML" not in source, f"{js.name} uses outerHTML"
        assert "insertAdjacentHTML" not in source, f"{js.name} injects markup"


def test_panel_javascript_only_links_out_to_http_urls():
    """The posting URL comes from the ATS payload, so it is untrusted too.

    `canonicalize_url` lowercases a scheme, it does not constrain one, so a
    `javascript:` href is the one injection `textContent` cannot stop. Every
    href assignment must therefore sit behind the single scheme allowlist.
    """
    source = (STATIC / "panel.js").read_text(encoding="utf-8")
    assigns = re.findall(r"\S+\.href\s*=[^\n]*", source)
    assert assigns == ["link.href = url;"], assigns
    assert re.search(r"/\^https\?:\\/\\//i", source), (
        "panel.js must allowlist an http(s) scheme before assigning an href")


def test_api_returns_job_text_verbatim_without_executing_it(client, tmp_path,
                                                            profile):
    """The API is a JSON boundary: no escaping games, no markup passthrough."""
    path = str(tmp_path / "x.db")
    conn = db.connect(path)
    db.init_schema(conn)
    from tests.conftest import _make_job
    job = _make_job("9")
    job.description_text = "<script>alert(1)</script>"
    _, vid = db.upsert_job(conn, job)
    db.set_status(conn, vid, "ready_for_match")
    db.set_status(conn, vid, "matching")
    conn.execute("INSERT INTO review_items(job_version_id, match_json, "
                 "total_score) VALUES(?,?,?)", (vid, '{"eligibility":"pass"}', 70))
    conn.commit()
    db.set_status(conn, vid, "pending_review")
    conn.close()
    c = TestClient(create_app(path, profile))
    body = c.get(f"/api/item/{vid}").json()
    assert body["job"]["description_text"] == "<script>alert(1)</script>"


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "OfferPilot" in r.text


def test_static_assets_are_mounted(client):
    """index.html is inert without them, and they ship as package data."""
    for asset in ("/static/panel.js", "/static/style.css"):
        assert client.get(asset).status_code == 200, asset


def test_blind_page_route_exists_without_crashing(client):
    """index.html links to /blind; the blind view itself lands in a later task.

    Until blind.html exists this must be an honest 404, not a FileResponse
    blowing up on a missing path.
    """
    assert client.get("/blind").status_code in (200, 404)


@pytest.fixture
def recorded_uvicorn(monkeypatch):
    """Capture what `serve` actually hands uvicorn, without binding a socket.

    Asserting on `serve`'s signature default only pins a value nothing has to
    use; hardcoding `0.0.0.0` inside the body passed that version of this test.
    """
    import uvicorn
    calls = []
    monkeypatch.setattr(uvicorn, "run",
                        lambda app, **kw: calls.append((app, kw)))
    return calls


def test_serve_binds_loopback_by_default(tmp_path, profile, recorded_uvicorn):
    """Single-user local tool: no auth, so it must never bind an interface."""
    from offerpilot.panel import app as panel_app
    panel_app.serve(str(tmp_path / "s.db"), profile)
    assert len(recorded_uvicorn) == 1
    assert recorded_uvicorn[0][1]["host"] == "127.0.0.1"


def test_serve_passes_an_explicit_loopback_host_through(tmp_path, profile,
                                                        recorded_uvicorn):
    """Loopback has more than one spelling; none of them is an interface."""
    from offerpilot.panel import app as panel_app
    for host in ("127.0.0.1", "localhost", "::1", "127.0.0.53"):
        recorded_uvicorn.clear()
        panel_app.serve(str(tmp_path / "s.db"), profile, host=host, port=9123)
        assert recorded_uvicorn[0][1] == {"host": host, "port": 9123}, host


@pytest.mark.parametrize("host", ["0.0.0.0", "", "::", "192.168.1.9",
                                  "example.com"])
def test_serve_refuses_a_non_loopback_host(tmp_path, profile, recorded_uvicorn,
                                           host):
    """The panel approves jobs with no auth: reaching it off-box is the bug.

    `cli` reads `panel.host` from config and passes it straight down, so the
    refusal has to live at the bind, not in the caller.
    """
    from offerpilot.panel import app as panel_app
    with pytest.raises(ValueError, match="loopback"):
        panel_app.serve(str(tmp_path / "s.db"), profile, host=host)
    assert recorded_uvicorn == []


def test_panel_cli_serves_the_configured_loopback_address(tmp_path, monkeypatch):
    from offerpilot import cli
    from offerpilot.panel import app as panel_app

    calls = {}

    def fake_serve(db_path, profile, host="127.0.0.1", port=8000):
        calls.update(db_path=db_path, host=host, port=port)

    monkeypatch.setattr(panel_app, "serve", fake_serve)
    cli.main(["panel", "--db", str(tmp_path / "cli.db"),
              "--config", "config.example.yaml",
              "--profile", "profile.example.yaml"])
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8000
    assert calls["db_path"] == str(tmp_path / "cli.db")


def test_panel_cli_refuses_a_non_loopback_configured_host(tmp_path, monkeypatch):
    """A config edit must not be able to publish the panel to the LAN."""
    from offerpilot import cli
    from offerpilot.panel import app as panel_app

    cfg = tmp_path / "bad.yaml"
    cfg.write_text("panel:\n  host: 0.0.0.0\n  port: 8000\n", encoding="utf-8")

    def boom(*a, **kw):
        raise AssertionError("serve must not be reached with a LAN host")

    monkeypatch.setattr(panel_app, "serve", boom)
    with pytest.raises(SystemExit) as e:
        cli.main(["panel", "--db", str(tmp_path / "cli.db"),
                  "--config", str(cfg),
                  "--profile", "profile.example.yaml"])
    assert e.value.code == 1


# --- Blind labeling view ---------------------------------------------------
#
# The blind view exists so the eval's ground truth is produced without seeing
# what the model said. That makes "the payload carries no model output" the
# whole feature, not a detail, so the leak test below greps the serialized
# response rather than naming fields.


def test_blind_next_hides_every_model_output(client, seeded):
    _, vid = seeded
    body = client.get("/api/blind/next").json()
    assert body["job"]["job_version_id"] == vid
    flat = json.dumps(body)
    for leaked in ("total_score", "skills_score", "eligibility", "evidence",
                   "match", "brief", "confidence", "status"):
        assert leaked not in flat, f"blind view leaked {leaked}"
    assert body["profile_summary"]["identity"]["name"]
    assert body["job"]["description_text"]


def test_blind_label_is_recorded_with_blind_eval_provenance(client, seeded):
    path, vid = seeded
    r = client.post(f"/api/blind/{vid}/label", json={"fit_label": "good_fit"})
    assert r.status_code == 200
    conn = db.connect(path)
    labels = db.get_labels(conn, version_id=vid)
    assert [l["label_source"] for l in labels] == ["blind_eval"]
    conn.close()


def test_blind_label_does_not_change_job_status(client, seeded):
    path, vid = seeded
    client.post(f"/api/blind/{vid}/label", json={"fit_label": "poor_fit"})
    conn = db.connect(path)
    assert conn.execute("SELECT status FROM job_versions WHERE id=?",
                        (vid,)).fetchone()["status"] == "pending_review"
    conn.close()


def test_blind_next_skips_already_labeled_and_reports_exhaustion(client,
                                                                seeded):
    _, vid = seeded
    client.post(f"/api/blind/{vid}/label", json={"fit_label": "uncertain"})
    body = client.get("/api/blind/next").json()
    assert body["job"] is None
    assert body["remaining"] == 0


def test_blind_label_requires_a_fit_label(client, seeded):
    _, vid = seeded
    assert client.post(f"/api/blind/{vid}/label", json={}).status_code == 422


def test_blind_label_on_an_unknown_version_is_404(client):
    r = client.post("/api/blind/9999/label", json={"fit_label": "good_fit"})
    assert r.status_code == 404


def test_blind_progress_counts_labeled_versus_total(client, seeded):
    _, vid = seeded
    before = client.get("/api/blind/progress").json()
    assert before["labeled"] == 0 and before["total"] >= 1
    client.post(f"/api/blind/{vid}/label", json={"fit_label": "good_fit"})
    assert client.get("/api/blind/progress").json()["labeled"] == 1


def test_blind_offers_versions_the_prefilter_dropped(tmp_path, profile):
    """`filtered_out` jobs are candidates, or the eval cannot see its own FNs.

    The prefilter's false negatives are only measurable if a human can label a
    job the prefilter threw away, so the blind queue must not be the review
    queue -- it is every job version, review item or not.
    """
    path = str(tmp_path / "f.db")
    conn = db.connect(path)
    db.init_schema(conn)
    from tests.conftest import _make_job
    _, dropped = db.upsert_job(conn, _make_job("77"))
    db.set_status(conn, dropped, "filtered_out")
    conn.close()
    c = TestClient(create_app(path, profile))
    body = c.get("/api/blind/next").json()
    assert body["job"]["job_version_id"] == dropped
    assert body["remaining"] == 1
    assert c.get("/api/blind/progress").json()["total"] == 1


def test_blind_page_and_its_script_are_served(client):
    page = client.get("/blind")
    assert page.status_code == 200
    assert "Blind labeling" in page.text
    assert client.get("/static/blind.js").status_code == 200


def test_blind_next_sends_exactly_these_keys_and_nothing_else(client, seeded):
    """An allowlist, because the denylist above only catches known names.

    `test_blind_next_hides_every_model_output` greps the serialized body for
    eight field names, which means a model output shipped under a *ninth*
    name passes it: adding `"priority": total_score` to the job dict leaks the
    score and trips none of those words. The guarantee this feature rests on
    is "nothing but the posting and the profile", so it has to be pinned as a
    closed set of keys -- then widening the payload fails here by default and
    a human has to decide the new field is not model output.
    """
    _, vid = seeded
    body = client.get("/api/blind/next").json()
    assert set(body) == {"job", "remaining", "profile_summary"}
    assert set(body["job"]) == {"job_version_id", "title", "company_id",
                                "location", "description_text", "url"}
    assert set(body["profile_summary"]) == {"identity", "constraints",
                                            "skills", "experiences"}
    for exp in body["profile_summary"]["experiences"]:
        assert set(exp) == {"id", "title", "summary"}
    # The exhausted branch is a second `return` and can drift on its own.
    client.post(f"/api/blind/{vid}/label", json={"fit_label": "good_fit"})
    empty = client.get("/api/blind/next").json()
    assert empty["job"] is None
    assert set(empty) == {"job", "remaining", "profile_summary"}


def test_blind_label_twice_is_a_conflict_not_a_second_ground_truth(client,
                                                                   seeded):
    """One job version, one blind label -- the eval reads every row it finds.

    The review route gets this for free: `db.set_status` refuses the second
    transition, so the label after it is never written. This route moves no
    status on purpose, so it has to refuse on its own, or a double-click
    leaves `run_eval` two contradictory ground truths for one job and no way
    to tell which one the human meant.
    """
    path, vid = seeded
    first = client.post(f"/api/blind/{vid}/label", json={"fit_label": "good_fit"})
    assert first.status_code == 200
    second = client.post(f"/api/blind/{vid}/label", json={"fit_label": "poor_fit"})
    assert second.status_code == 409
    conn = db.connect(path)
    rows = db.get_labels(conn, version_id=vid, label_source="blind_eval")
    assert [r["fit_label"] for r in rows] == ["good_fit"]
    conn.close()


def test_blind_javascript_disables_the_buttons_while_the_post_is_in_flight():
    """The 409 is the backstop; the page must not fire the second POST at all.

    `if (res.ok) next()` only re-renders after the await, so between the first
    click and the response the other two verdict buttons are still live. This
    is a source assertion because the suite has no JavaScript runtime -- what
    it pins is the ordering: disabled before the POST, re-enabled after it.
    """
    source = (STATIC / "blind.js").read_text(encoding="utf-8")
    post = source.index("/label")
    assert "disabled = true" in source, "blind.js never disables a button"
    assert source.index("disabled = true") < post, (
        "blind.js must disable the verdict buttons before it POSTs a label")
    assert "finally" in source, (
        "blind.js must re-enable the buttons even when the POST fails")
    assert source.index("disabled = false") > post
