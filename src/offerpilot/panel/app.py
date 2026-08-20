"""The local review panel: HTTP in front of the review queue.

Single user, one machine. `serve` binds `127.0.0.1` and there is no auth, no
CORS and no session -- that is the boundary, not an omission: the panel can
approve a job and write a label, so it must not be reachable from anywhere
but the loopback interface. `require_loopback` enforces that at the bind,
because `cli` reads `panel.host` out of a config file and hands it straight
down. The bind is only half of it, though, since a page in the user's own
browser reaches loopback too; the Host allowlist in `create_app` is the other
half, and the same reasoning is why the interactive API docs are off.

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
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse

from offerpilot.brief import ApplicationBrief
from offerpilot.labels import ActionLabel, FitLabel, RejectionReason
from offerpilot.store import db

STATIC_DIR = pathlib.Path(__file__).parent / "static"

# The names a browser may legitimately use to reach a loopback bind, plus the
# one `TestClient` sends. `serve` adds whatever host it was actually told to
# bind. See `create_app` for why a Host allowlist is not redundant with the
# loopback bind.
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "testserver")


def _normalize_host(value: str) -> str:
    """The hostname in a `Host` header, with brackets and port removed.

    `[::1]:8000` -> `::1`, `127.0.0.1:8000` -> `127.0.0.1`, `::1` -> `::1`.
    An IPv6 literal is full of colons, so the port is whatever follows the
    closing bracket, or -- unbracketed -- only ever the single colon in a
    name-or-IPv4 host. Anything malformed (unclosed bracket, junk after the
    literal, a non-numeric port) is returned as it arrived, which no
    allowlist entry can equal: refusal by falling through, not by guessing.
    """
    value = value.strip()
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return value
        host, rest = value[1:end], value[end + 1:]
        if rest and not (rest.startswith(":") and rest[1:].isdigit()):
            return value
        return host
    if value.count(":") == 1:
        host, _, port = value.partition(":")
        return host if port.isdigit() else value
    return value


class HostAllowlistMiddleware:
    """Refuse any request whose `Host` is not one this bind answers to.

    Starlette's `TrustedHostMiddleware` cannot express this allowlist: it
    reduces the header with `split(":")[0]`, so a browser pointed at an IPv6
    loopback bind sends `Host: [::1]:8000` and it compares the literal `"["`,
    400ing every request including `/`. Putting `"["` in the allowlist would
    "fix" that by matching every bracketed IPv6 literal in existence, which is
    the entire set an attacker picks from -- so the header is parsed properly
    here instead and matched whole.
    """

    def __init__(self, app, allowed_hosts):
        self.app = app
        self.allowed = {h for h in (_normalize_host(a) for a in allowed_hosts)
                        if h}

    async def __call__(self, scope, receive, send):
        # Only HTTP is checked because only HTTP is served -- there is no
        # websocket route to reach, and a `PlainTextResponse` cannot answer a
        # handshake anyway. Adding one means extending this.
        if scope["type"] == "http":
            host = _normalize_host(Headers(scope=scope).get("host", ""))
            if not host or host not in self.allowed:
                response = PlainTextResponse("Invalid host header",
                                             status_code=400)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


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


def create_app(db_path: str, profile, allowed_hosts=ALLOWED_HOSTS) -> FastAPI:
    # No `/docs`, no `/redoc`, no `/openapi.json`: FastAPI's interactive docs
    # are two <script> tags pointing at a CDN, loaded into the same origin as
    # the page that approves jobs and writes labels. This app has exactly one
    # consumer -- the two static pages next to it -- so the docs buy nothing
    # and the third-party JS is pure surface. Disabled at the app rather than
    # routed around, so there is no second URL that still serves them.
    app = FastAPI(title="OfferPilot Review Panel",
                  docs_url=None, redoc_url=None, openapi_url=None)

    # `serve` refusing a non-loopback bind stops the LAN; it does not stop a
    # page already open in this browser. Any site can point a hostname it
    # controls at 127.0.0.1 and script requests to it -- DNS rebinding -- and
    # those arrive on loopback, from the user's browser, with no auth to fail.
    # The Host header is the only thing that separates them from the user's
    # own tab, so it is checked -- against the bind and the names that reach
    # it, matched whole. Not `TrustedHostMiddleware`: it splits the port off
    # at the first colon, which turns the `Host: [::1]:8000` a browser sends
    # to a `::1` bind into `[` and 400s the page `require_loopback` blessed.
    app.add_middleware(HostAllowlistMiddleware,
                       allowed_hosts=list(allowed_hosts))

    # Once, here, rather than on every request. `db.migrate` is idempotent, so
    # calling it per request broke nothing -- it just put three PRAGMA round
    # trips and a commit on the read path, and deferred a schema-less database
    # into a 500 on the first GET instead of a failure at startup.
    # `init_schema` is `CREATE TABLE IF NOT EXISTS` throughout and ends in
    # `migrate`, so it is that one migrate plus an empty queue rather than a
    # crash when the panel is pointed at a path the CLI has not created yet.
    bootstrap = db.connect(db_path)
    try:
        db.init_schema(bootstrap)
    finally:
        bootstrap.close()

    def conn() -> sqlite3.Connection:
        return db.connect(db_path)

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


def panel_url(host: str, port: int) -> str:
    """The address to actually type at a browser for this bind.

    An IPv6 literal has to be bracketed inside a URL authority -- otherwise
    the colons of the address run into the colon before the port, and
    `http://::1:8000` is not something a browser can open. `require_loopback`
    accepts `::1`, and the banner is the only instruction the user gets.
    """
    bare = host.strip("[]")
    try:
        bracket = ipaddress.ip_address(bare).version == 6
    except ValueError:
        bracket = False
    return f"http://{f'[{bare}]' if bracket else bare}:{port}"


def serve(db_path: str, profile, host: str = "127.0.0.1",
          port: int = 8000) -> None:
    import uvicorn
    require_loopback(host)
    # The bound host joins the Host allowlist: `require_loopback` blesses more
    # spellings of loopback than `ALLOWED_HOSTS` names (127.0.0.53, say), and
    # the panel refusing the address it was just told to bind would be a
    # self-inflicted 400.
    hosts = list(ALLOWED_HOSTS) + ([host] if host not in ALLOWED_HOSTS else [])
    uvicorn.run(create_app(db_path, profile, allowed_hosts=hosts),
                host=host, port=port)
