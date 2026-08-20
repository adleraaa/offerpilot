"""The local review panel: HTTP in front of the review queue.

Single user, one machine. `serve` binds `127.0.0.1` and there is no auth, no
CORS and no session -- that is the boundary, not an omission: the panel can
approve a job and write a label, so it must not be reachable from anywhere
but the loopback interface.

Everything a route returns is JSON, verbatim. Job text is untrusted, and the
escaping happens exactly once, in the browser, where `panel.js` writes it with
`textContent`. Escaping here as well would double-escape; escaping here
*instead* would leave the DOM path to guess. `tests/test_panel.py` pins both
halves of that contract.
"""

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
            # index.html links here; the blind labeling view is a later
            # milestone. An honest 404 beats FileResponse raising at send time.
            raise HTTPException(404, "blind labeling view is not built yet")
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

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def serve(db_path: str, profile, host: str = "127.0.0.1",
          port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(create_app(db_path, profile), host=host, port=port)
