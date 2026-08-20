"""The local review panel: HTTP in front of the review queue.

Single user, one machine. `serve` binds `127.0.0.1` and there is no auth, no
CORS and no session -- that is the boundary, not an omission: the panel can
approve a job and write a label, so it must not be reachable from anywhere
but the loopback interface. `require_loopback` enforces that at the bind,
because `cli` reads `panel.host` out of a config file and hands it straight
down.

Everything a route returns is JSON, verbatim. Job text is untrusted, and the
escaping happens exactly once, in the browser, where `panel.js` writes it with
`textContent`. Escaping here as well would double-escape; escaping here
*instead* would leave the DOM path to guess. `tests/test_panel.py` pins both
halves of that contract.

Two pages, two label sources, and the difference between them is the point.
`/` shows the model's work and writes `review_feedback`. `/blind` shows the
posting and the profile and nothing the model produced, and writes
`blind_eval` -- the only labels the eval harness reads, because ground truth
that has already seen the prediction is not ground truth.
"""

import ipaddress
import json
import pathlib
import sqlite3
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from offerpilot.brief import ApplicationBrief
from offerpilot.labels import ActionLabel, FitLabel, RejectionReason
from offerpilot.store import db

STATIC_DIR = pathlib.Path(__file__).parent / "static"

# The only three transitions `pending_review` allows (db.ALLOWED_TRANSITIONS).
_ACTION_TO_STATUS = {"approve": "approved", "reject": "rejected",
                     "save": "saved"}


class Decision(BaseModel):
    action: Literal["approve", "reject", "save"]
    fit_label: Optional[FitLabel] = None
    action_label: Optional[ActionLabel] = None
    rejection_reason: Optional[RejectionReason] = None
    notes: Optional[str] = None


class BriefEdit(BaseModel):
    brief: ApplicationBrief


class BlindLabel(BaseModel):
    fit_label: FitLabel
    action_label: Optional[ActionLabel] = None
    rejection_reason: Optional[RejectionReason] = None
    notes: Optional[str] = None


def _row_to_queue_item(row: sqlite3.Row) -> dict:
    return {"job_version_id": row["job_version_id"],
            "title": row["title"], "company_id": row["company_id"],
            "location": row["location"], "total_score": row["total_score"],
            "url": row["canonical_url"],
            "has_brief": row["brief_json"] is not None}


def create_app(db_path: str, profile) -> FastAPI:
    app = FastAPI(title="OfferPilot Review Panel")

    def conn() -> sqlite3.Connection:
        c = db.connect(db_path)
        db.migrate(c)
        return c

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/blind")
    def blind_page():
        page = STATIC_DIR / "blind.html"
        if not page.exists():
            # Ships next to this module as package data; if a partial install
            # dropped it, an honest 404 beats FileResponse raising at send.
            raise HTTPException(404, "blind labeling page is missing")
        return FileResponse(page)

    @app.get("/api/queue")
    def api_queue():
        with conn() as c:
            rows = db.get_review_queue(c)
        return {"items": [_row_to_queue_item(r) for r in rows]}

    @app.get("/api/item/{version_id}")
    def api_item(version_id: int):
        with conn() as c:
            row = db.get_review_item(c, version_id)
        if row is None:
            raise HTTPException(404, "no review item for that job version")
        match = json.loads(row["match_json"])
        brief_json = row["edited_brief_json"] or row["brief_json"]
        return {
            "job_version_id": row["job_version_id"],
            "status": row["status"],
            "total_score": row["total_score"],
            # `unknown` is never silently a pass: the panel banners it.
            "eligibility_unresolved": match.get("eligibility") == "unknown",
            "match": match,
            "brief": json.loads(brief_json) if brief_json else None,
            "brief_is_edited": row["edited_brief_json"] is not None,
            "job": {"title": row["title"], "location": row["location"],
                    "company_id": row["company_id"], "url": row["canonical_url"],
                    "description_text": row["description_text"]},
        }

    @app.post("/api/item/{version_id}/decision")
    def api_decision(version_id: int, decision: Decision):
        # Refuse before touching the DB: a rejected decision writes nothing,
        # so a 422 leaves the row exactly as reviewable as it was.
        if decision.action == "reject" and not decision.rejection_reason:
            raise HTTPException(422, "rejection_reason is required to reject")
        with conn() as c:
            if db.get_review_item(c, version_id) is None:
                raise HTTPException(404, "no review item for that job version")
            try:
                db.set_status(c, version_id, _ACTION_TO_STATUS[decision.action])
            except ValueError as e:
                # Already decided, or never reached review. A stale browser
                # tab is a conflict, not a server fault -- and the label below
                # must not be written for a decision that did not take.
                raise HTTPException(409, str(e))
            db.record_label(c, version_id, label_source="review_feedback",
                            fit_label=decision.fit_label,
                            action_label=decision.action_label,
                            rejection_reason=decision.rejection_reason,
                            notes=decision.notes)
        return {"ok": True, "status": _ACTION_TO_STATUS[decision.action]}

    @app.put("/api/item/{version_id}/brief")
    def api_edit_brief(version_id: int, payload: BriefEdit):
        with conn() as c:
            if db.get_review_item(c, version_id) is None:
                raise HTTPException(404, "no review item for that job version")
            # Written to a separate column: the model's original stays
            # readable, which is the whole point of keeping edits as labels.
            db.save_edited_brief(c, version_id, payload.brief.model_dump_json())
        return {"ok": True}

    @app.get("/api/trace/{version_id}")
    def api_trace(version_id: int):
        with conn() as c:
            rows = c.execute(
                "SELECT rs.node, rs.attempt, rs.status, rs.started_at, "
                "rs.completed_at, rs.error FROM run_steps rs "
                "JOIN runs r ON r.id = rs.run_id "
                "WHERE r.job_version_id=? ORDER BY rs.id",
                (version_id,)).fetchall()
        return {"steps": [dict(r) for r in rows]}

    def _profile_summary() -> dict:
        """The half of the profile a human needs to judge a job themselves.

        Deliberately assembled field by field rather than dumped: the blind
        page must carry no model output, and `profile.model_dump()` would
        widen automatically the next time a field is added to `Profile`.
        """
        return {"identity": profile.identity.model_dump(),
                "constraints": profile.constraints.model_dump(),
                "skills": profile.skills.model_dump(),
                "experiences": [{"id": e.id, "title": e.title,
                                 "summary": e.summary}
                                for e in profile.experiences]}

    @app.get("/api/blind/next")
    def api_blind_next():
        """One unlabeled job version: posting and profile, nothing else.

        No score, no subscores, no eligibility, no evidence, no brief and no
        status -- a label written after seeing any of those is not independent
        of the model, and the eval's ground truth would be measuring itself.
        Candidates come from `db.get_blind_candidates`, which spans *every*
        job version including `filtered_out` ones, because the eval has to be
        able to count the jobs the prefilter dropped by mistake.
        """
        with conn() as c:
            rows = db.get_blind_candidates(c, limit=1, unlabeled_only=True)
            # Same FROM and same predicate as get_blind_candidates, so
            # `remaining == 0` and "no rows" can never disagree.
            remaining = c.execute(
                "SELECT COUNT(*) n FROM job_versions jv "
                "JOIN jobs j ON j.id = jv.job_id "
                "WHERE NOT EXISTS (SELECT 1 FROM labels l "
                "WHERE l.job_version_id = jv.id "
                "AND l.label_source='blind_eval')").fetchone()["n"]
        summary = _profile_summary()
        if not rows:
            return {"job": None, "remaining": remaining,
                    "profile_summary": summary}
        r = rows[0]
        return {
            "job": {"job_version_id": r["id"], "title": r["title"],
                    "company_id": r["company_id"], "location": r["location"],
                    "description_text": r["description_text"],
                    "url": r["canonical_url"]},
            "remaining": remaining,
            "profile_summary": summary,
        }

    @app.post("/api/blind/{version_id}/label")
    def api_blind_label(version_id: int, label: BlindLabel):
        """Write the human's own verdict once, and touch nothing else.

        No status transition: a blind label is an opinion about the job, not a
        decision about the application, and moving the row would both corrupt
        the review queue and let the label change what the pipeline does.

        That is also why this route has to refuse a repeat itself. `/decision`
        is protected by accident -- `db.set_status` raises on the second
        transition, so its label never runs -- and skipping the transition
        here removes that guard with nothing behind it. The eval reads *every*
        `blind_eval` row, so a second one is not a correction, it is two
        contradictory ground truths for one job. The check is not atomic
        against a concurrent duplicate, but the panel is one human on
        loopback; what it stops is a double-click and a stale tab.
        """
        with conn() as c:
            exists = c.execute("SELECT 1 FROM job_versions WHERE id=?",
                               (version_id,)).fetchone()
            if exists is None:
                raise HTTPException(404, "unknown job version")
            if db.get_labels(c, version_id=version_id,
                             label_source="blind_eval"):
                raise HTTPException(
                    409, "this job version already has a blind label; the "
                         "eval reads them all, so it gets exactly one")
            db.record_label(c, version_id, label_source="blind_eval",
                            fit_label=label.fit_label,
                            action_label=label.action_label,
                            rejection_reason=label.rejection_reason,
                            notes=label.notes)
        return {"ok": True}

    @app.get("/api/blind/progress")
    def api_blind_progress():
        with conn() as c:
            total = c.execute(
                "SELECT COUNT(*) n FROM job_versions").fetchone()["n"]
            labeled = c.execute(
                "SELECT COUNT(DISTINCT job_version_id) n FROM labels "
                "WHERE label_source='blind_eval'").fetchone()["n"]
        # The spec asks for 40-60 blind labels before the eval numbers mean
        # anything; the page shows the target so the count is not just a tally.
        return {"labeled": labeled, "total": total,
                "target_min": 40, "target_max": 60}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def require_loopback(host: str) -> str:
    """Return `host` unchanged, or refuse if it is reachable off this machine.

    Every route above can be called without authenticating, so the bind
    address is the entire access-control story. That makes a one-line edit to
    `panel.host` in `config.yaml` the cheapest way to publish an
    unauthenticated write API to the LAN, and the caller is the wrong place to
    catch it -- `cli` passes the configured value through verbatim.
    """
    try:
        # Strip the brackets a URL-shaped IPv6 literal arrives wrapped in.
        ok = ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        # Not an address at all; a name could resolve anywhere, so only the
        # one that cannot resolve off-box is allowed.
        ok = host == "localhost"
    if not ok:
        raise ValueError(
            f"panel host {host!r} is not a loopback address. The review panel "
            f"approves jobs and writes labels with no auth, so it must not be "
            f"reachable off this machine: bind 127.0.0.1 and tunnel to it.")
    return host


def serve(db_path: str, profile, host: str = "127.0.0.1",
          port: int = 8000) -> None:
    import uvicorn
    require_loopback(host)
    uvicorn.run(create_app(db_path, profile), host=host, port=port)
