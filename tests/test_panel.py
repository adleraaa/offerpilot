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

    `canonicalize_url` refuses a non-http(s) scheme at the collector boundary,
    but demo fixtures build `NormalizedJob` directly and skip it, so this
    render-site allowlist is load-bearing rather than belt-and-braces. An href
    is the one injection `textContent` cannot stop.

    This reads panel.js as source: there is no JS runtime in this suite. It
    therefore pins the guard's *shape* — a negated test that returns early —
    because asserting only that the regex appears somewhere stays green if
    someone inverts it.
    """
    source = (STATIC / "panel.js").read_text(encoding="utf-8")
    assigns = re.findall(r"\S+\.href\s*=[^\n]*", source)
    assert assigns == ["link.href = url;"], assigns
    guard = re.search(
        r"if \(typeof url !== \"string\" \|\| !/\^https\?:\\/\\//i\.test\(url\)\)"
        r"\s*\{\s*return ", source)
    assert guard, (
        "panel.js must reject a non-http(s) url and return BEFORE assigning "
        "an href; the negated early-return form is what this pins")


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


def test_a_missing_blind_page_is_an_honest_404_not_a_crash(client, tmp_path,
                                                          monkeypatch):
    """blind.html ships as package data; a partial install must still answer.

    This replaced `status_code in (200, 404)`, which was a tautology: with the
    whole `/blind` route deleted FastAPI returns 404 and the assertion still
    held, so the test passed whether or not the feature existed. The branch it
    described -- `page.exists()` guarding `FileResponse`, which raises at send
    on a missing path -- was never actually exercised, because blind.html is
    always there. So the file is taken away instead, and the 200 case stays
    pinned by `test_blind_page_and_its_script_are_served`.
    """
    from offerpilot.panel import app as panel_app
    monkeypatch.setattr(panel_app, "STATIC_DIR", tmp_path)
    r = client.get("/blind")
    assert r.status_code == 404
    assert "blind labeling page is missing" in r.json()["detail"]


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


# --- Readability, labels and hardening -------------------------------------
#
# The README's first instruction is to open this panel, so "can a reader see
# it at all" is a correctness property, not a preference. The stylesheet is
# read as text and the ratios are computed here rather than eyeballed: a
# palette that looks fine on the author's own machine is exactly the defect.

STYLE = STATIC / "style.css"

_TOKEN_RE = re.compile(r"(--[\w-]+)\s*:\s*([^;}]+)")
_HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def _css() -> str:
    """The stylesheet with comments removed.

    Everything below asserts on declarations. A comment is prose -- the one
    at the top of style.css quotes the #1a1a1a that caused this -- and prose
    that trips a colour audit would push the explanation out of the file.
    """
    return re.sub(r"/\*.*?\*/", "", STYLE.read_text(encoding="utf-8"),
                  flags=re.S)


def _srgb_to_linear(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(color: str) -> float:
    """WCAG 2.1 relative luminance of a `#rgb` or `#rrggbb` colour."""
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    assert len(h) == 6, f"{color} is not an opaque hex colour"
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.2126 * _srgb_to_linear(r) + 0.7152 * _srgb_to_linear(g)
            + 0.0722 * _srgb_to_linear(b))


def _contrast(fg: str, bg: str) -> float:
    a, b = _luminance(fg), _luminance(bg)
    lo, hi = min(a, b), max(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _block_bounds(css: str, start: int) -> tuple[int, int]:
    """Indices of the `{` at or after `start` and its matching `}`."""
    i = css.index("{", start)
    depth, j = 0, i
    while True:
        if css[j] == "{":
            depth += 1
        elif css[j] == "}":
            depth -= 1
            if depth == 0:
                return i, j
        j += 1


def _balanced_block(css: str, start: int) -> str:
    """The text inside the `{...}` that opens at or after `start`."""
    i, j = _block_bounds(css, start)
    return css[i + 1:j]


def _palettes() -> tuple[dict, dict]:
    """The light palette, and the token overrides the dark block applies.

    Returned separately so a token defined *only* inside the dark media
    query -- which would be undefined for every light-mode reader -- shows up
    as a key the light palette does not have.
    """
    css = _css()
    dark_at = css.find("prefers-color-scheme: dark")
    light, dark_overrides = {}, {}
    for m in re.finditer(r":root[^{,]*\{", css):
        block = _balanced_block(css, m.start())
        tokens = {k: v.strip() for k, v in _TOKEN_RE.findall(block)}
        if dark_at != -1 and m.start() > dark_at:
            dark_overrides.update(tokens)
        else:
            light.update(tokens)
    return light, dark_overrides


def test_the_stylesheet_defines_a_palette_for_both_themes():
    """`color-scheme: light dark` is a promise the tokens have to keep.

    Declaring it tells the browser to paint a dark canvas for a reader whose
    OS is dark. A stylesheet that then defines only light foregrounds -- and
    leaves `body` with no background at all -- inherits that dark canvas and
    puts near-black text on it. Either both palettes exist or the declaration
    is a lie, so this pins the shape: tokens on `:root`, a dark block that
    overrides tokens and nothing else, and an explicit `body` background.
    """
    css = _css()
    light, dark = _palettes()
    assert "prefers-color-scheme: dark" in css, (
        "style.css declares color-scheme: light dark but never defines a "
        "dark palette")
    for token in ("--bg", "--fg", "--muted", "--line", "--warn-bg",
                  "--warn-fg"):
        assert token in light, f"{token} is not defined for light mode"
        assert token in dark, f"{token} is not overridden for dark mode"
    assert not set(dark) - set(light), (
        f"dark-only tokens are undefined in light mode: "
        f"{sorted(set(dark) - set(light))}")

    dark_block = _balanced_block(css, css.index("@media (prefers-color-scheme"))
    stripped = re.sub(r":root[^{]*\{|\}", "", dark_block)
    for decl in (d.strip() for d in stripped.split(";")):
        assert not decl or decl.startswith("--"), (
            f"the dark block must redefine tokens only, not layout: {decl!r}")

    body = _balanced_block(css, css.index("\nbody"))
    assert re.search(r"background(-color)?\s*:\s*var\(--bg\)", body), (
        "body must paint its own background from a token; without one it "
        "inherits the browser's dark canvas while the text stays dark")


def test_every_colour_in_the_stylesheet_comes_from_a_token():
    """A hex literal outside `:root` is a colour only one theme can see.

    `pre` shipped `background: #0000000a`, which is a light-mode assumption
    hardcoded into a rule the dark block cannot reach. Keeping every literal
    inside the two `:root` blocks is what makes the contrast test below
    exhaustive rather than a spot check.
    """
    css = _css()
    root_spans = [_block_bounds(css, m.start())
                  for m in re.finditer(r":root[^{,]*\{", css)]
    outside = [m.group(0) for m in _HEX_RE.finditer(css)
               if not any(lo < m.start() < hi for lo, hi in root_spans)]
    assert not outside, (
        f"colour literals outside the palette blocks: {outside}")


def test_panel_text_clears_wcag_contrast_in_both_themes():
    """Measured, not eyeballed. Body text 4.5:1, the warning banner 3:1.

    Before this, a dark-OS reader got `--fg` #1a1a1a on the browser's dark
    canvas: 1.08:1, which is the queue buttons and the entire posting body
    rendered invisible on the one page the README tells people to open first.
    """
    light, overrides = _palettes()
    dark = {**light, **overrides}
    checks = [("--fg", "--bg", 4.5), ("--muted", "--bg", 4.5),
              ("--link", "--bg", 4.5), ("--fg", "--code-bg", 4.5),
              ("--muted", "--code-bg", 4.5), ("--warn-fg", "--warn-bg", 3.0)]
    failures = []
    for theme, palette in (("light", light), ("dark", dark)):
        for fg, bg, need in checks:
            assert fg in palette and bg in palette, (
                f"the {theme} palette is missing {fg} or {bg}")
            got = _contrast(palette[fg], palette[bg])
            if got < need:
                failures.append(f"{theme}: {fg} on {bg} is {got:.2f}:1, "
                                f"needs {need}:1")
    assert not failures, "; ".join(failures)


def test_the_fit_select_has_no_default_verdict():
    """Clicking Reject without touching the dropdown must not say good_fit.

    The options were built straight from the vocabulary, so `good_fit` was
    option zero and therefore selected by default: every reject with an
    untouched dropdown wrote `fit_label='good_fit'` next to it. Those rows are
    auxiliary signal the eval reads, so that is corrupt data, not a cosmetic
    default. Asserted against the source because the suite has no JS runtime,
    and paired with the server-side test below that proves null is accepted.
    """
    source = (STATIC / "panel.js").read_text(encoding="utf-8")
    bar = source[source.index("function decisionBar"):
                 source.index("async function loadItem")]
    fit_block = bar[bar.index('const fit = el("select")'):
                    bar.index('const reason = el("select")')]
    appends = re.findall(r"fit\.appendChild\((.+?)\);", fit_block)
    assert appends, "the fit select is never populated"
    assert re.fullmatch(r'new Option\("\([^"]*\)", ""\)', appends[0]), (
        f"the first option appended to the fit select is {appends[0]!r}; it "
        f"has to be an empty-valued placeholder, or the browser preselects a "
        f"real verdict for a reviewer who never opened the dropdown")
    assert len(appends) > 1, "the fit vocabulary is never offered"
    assert re.search(r"fit_label:\s*fit\.value\s*\|\|\s*null", bar), (
        "an untouched select must send null, not the empty string")
    assert not re.search(r"fit_label:\s*fit\.value\s*[,}]", bar), (
        "fit.value is sent unguarded; the placeholder's empty value would "
        "fail the FitLabel vocabulary check as a 422")


def test_a_decision_with_no_fit_label_records_a_null_not_a_guess(client,
                                                                seeded):
    """The server half of the contract the panel now relies on."""
    path, vid = seeded
    r = client.post(f"/api/item/{vid}/decision",
                    json={"action": "reject", "fit_label": None,
                          "rejection_reason": "seniority"})
    assert r.status_code == 200
    conn = db.connect(path)
    label = db.get_labels(conn, version_id=vid)[0]
    assert label["fit_label"] is None
    assert label["rejection_reason"] == "seniority"
    conn.close()


def test_the_schema_is_brought_up_to_date_once_at_startup(seeded, profile,
                                                          monkeypatch):
    """`conn()` ran `migrate` per request: PRAGMA queries on every GET.

    It is idempotent, so nothing broke -- which is why it survived. It still
    put three PRAGMA round trips and a commit on the read path that belong to
    startup, and it deferred the failure: a database with no schema 500s on
    the first request rather than refusing to come up.
    """
    path, _ = seeded
    calls = []
    real = db.migrate

    def counting(conn):
        calls.append(1)
        return real(conn)

    monkeypatch.setattr(db, "migrate", counting)
    app = create_app(path, profile)
    assert len(calls) == 1, (
        f"create_app must migrate exactly once, ran it {len(calls)} times")
    client = TestClient(app)
    for _ in range(3):
        assert client.get("/api/queue").status_code == 200
    client.get("/api/blind/progress")
    assert len(calls) == 1, (
        f"migrate ran {len(calls) - 1} extra time(s) on the request path")


def test_the_interactive_api_docs_are_not_served(client):
    """FastAPI's default docs pull Swagger and ReDoc from a CDN.

    Same origin as the panel, on a local single-user tool with no API
    consumers to document: a third-party script tag on the page that approves
    jobs, in exchange for nothing. Turned off at the app, not routed around.
    """
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_a_foreign_host_header_is_refused(client):
    """Binding loopback does not stop a web page from reaching loopback.

    `serve` refuses a non-loopback bind, which stops the LAN. It does not stop
    a page in the user's own browser from resolving evil.example.com to
    127.0.0.1 and POSTing a decision to it -- the request arrives on loopback,
    from the browser, with no auth to fail. The Host header is the only thing
    that tells that request apart from the user's own tab.
    """
    assert client.get("/api/queue",
                      headers={"Host": "evil.example.com"}).status_code == 400
    for good in ("127.0.0.1:8000", "localhost:8000", "localhost"):
        assert client.get("/api/queue",
                          headers={"Host": good}).status_code == 200, good


def _app_serve_would_run(seeded, profile, recorded_uvicorn, host):
    """The app `serve` actually hands uvicorn, driven for real.

    `test_serve_passes_an_explicit_loopback_host_through` asserts on the
    kwargs and stops there, so it went on passing for `::1` while every
    request to that bind 400'd. Capturing the app and putting requests
    through it is the only way that assurance is worth anything.
    """
    from offerpilot.panel import app as panel_app
    path, _ = seeded
    recorded_uvicorn.clear()
    panel_app.serve(path, profile, host=host, port=8000)
    return TestClient(recorded_uvicorn[0][0])


def test_an_ipv6_loopback_bind_answers_the_browsers_bracketed_host(
        seeded, profile, recorded_uvicorn):
    """`::1` is a supported bind, so the panel has to answer on it.

    `require_loopback` blesses `::1` and `cli` reads `panel.host` straight out
    of `config.yaml`, so one line of config puts the panel on the IPv6
    loopback. A browser sent there sends `Host: [::1]:8000`, and a Host check
    that splits the port off at the *first* colon reduces that to the literal
    `[` -- 400 on every request including `/`, so the page never loads.
    """
    client = _app_serve_would_run(seeded, profile, recorded_uvicorn, "::1")
    for good in ("[::1]:8000", "[::1]", "::1"):
        assert client.get("/", headers={"Host": good}).status_code == 200, good
        assert client.get("/api/queue",
                          headers={"Host": good}).status_code == 200, good


def test_a_bracketed_ipv6_host_is_matched_exactly_not_waved_through(
        seeded, profile, recorded_uvicorn):
    """Allowlisting the `[` the naive split produces would be strictly worse.

    It would match every bracketed IPv6 literal there is, which is the whole
    class of hosts an attacker gets to choose from. A `::1` bind allows `::1`.
    """
    client = _app_serve_would_run(seeded, profile, recorded_uvicorn, "::1")
    for bad in ("[::2]:8000", "[dead:beef::1]:8000", "[", "[]", "[::1",
                "[::1]evil.example.com", "evil.example.com", ""):
        assert client.get("/api/queue",
                          headers={"Host": bad}).status_code == 400, bad


def test_an_ipv4_bind_does_not_answer_the_ipv6_loopback(client):
    """The allowlist is the bind plus the names that reach it, not `loopback`.

    The default bind is `127.0.0.1`; nothing is listening on `::1`, so a
    request claiming to be for it did not come from the user's own tab.
    """
    assert client.get("/api/queue",
                      headers={"Host": "[::1]:8000"}).status_code == 400


def test_an_unusual_ipv4_loopback_bind_answers_with_and_without_a_port(
        seeded, profile, recorded_uvicorn):
    """The reason `serve` adds the bound host to the allowlist at all."""
    client = _app_serve_would_run(seeded, profile, recorded_uvicorn,
                                  "127.0.0.53")
    for good in ("127.0.0.53:8000", "127.0.0.53"):
        assert client.get("/api/queue",
                          headers={"Host": good}).status_code == 200, good


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", "http://127.0.0.1:8000"),
    ("localhost", "http://localhost:8000"),
    ("127.0.0.53", "http://127.0.0.53:8000"),
    ("::1", "http://[::1]:8000"),
    ("[::1]", "http://[::1]:8000"),
])
def test_panel_url_brackets_an_ipv6_literal(host, expected):
    """`http://::1:8000` is not a URL a browser can open.

    An IPv6 literal has to be bracketed in an authority, or the colons of the
    address and the colon before the port are indistinguishable.
    """
    from offerpilot.panel import app as panel_app
    assert panel_app.panel_url(host, 8000) == expected


def test_panel_cli_prints_an_openable_url_for_an_ipv6_bind(tmp_path,
                                                           monkeypatch, capsys):
    """The banner is the only instruction the user gets; it has to be right."""
    from offerpilot import cli
    from offerpilot.panel import app as panel_app

    cfg = tmp_path / "v6.yaml"
    cfg.write_text("panel:\n  host: '::1'\n  port: 8000\n", encoding="utf-8")
    monkeypatch.setattr(panel_app, "serve", lambda *a, **kw: None)
    cli.main(["panel", "--db", str(tmp_path / "cli.db"), "--config", str(cfg),
              "--profile", "profile.example.yaml"])
    assert "http://[::1]:8000" in capsys.readouterr().out
