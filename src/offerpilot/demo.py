"""Key-free demo: synthetic profile, synthetic postings, pre-recorded outputs.

`offerpilot demo` seeds a throwaway SQLite database from `demo/` and serves the
review panel against it. There is no config file, no `profile.yaml`, no API key
and no network call anywhere in this path -- the only thing standing in for the
model is `MockLLM`, which replays outputs recorded by hand.

The demo runs the *same* graph as a real run. That is the point of it: if the
demo took a shortcut around `run_match_for_version`, what it showed would be a
picture of the pipeline rather than the pipeline.
"""

import json
import os
import pathlib
import tempfile

from offerpilot import prefilter
from offerpilot.brief import ApplicationBrief
from offerpilot.graph import run_match_for_version
from offerpilot.llm import PermanentLLMError
from offerpilot.models import MatchResult, NormalizedJob
from offerpilot.profile import Profile, load_profile
from offerpilot.store import db

# src/offerpilot/demo.py -> src/offerpilot -> src -> repo root.
DEMO_DIR = pathlib.Path(__file__).resolve().parents[2] / "demo"
DEMO_THRESHOLD = 60

# The demo's synthetic employers, so the panel has a name to print rather than
# a bare slug. Kept beside the fixtures they describe.
DEMO_COMPANIES = [
    {"id": "examplecorp", "name": "ExampleCorp"},
    {"id": "samplestartup", "name": "SampleStartup"},
    {"id": "excludedcorp", "name": "ExcludedCorp"},
]

_SCHEMA_BY_NODE = {"match": MatchResult, "brief": ApplicationBrief}


class MockLLM:
    """Replays recorded outputs. Never invents one - an unrecorded job raises.

    Two behaviours here are load-bearing and neither is decoration:

    1. `validate` is **called**, exactly as `LLMClient.structured` calls it.
       Handing the validator to the client is what enforces the grounding
       rule; a mock that accepted `**kwargs` and dropped `validate` would
       disarm the citation check for everything that goes through the demo,
       and a recorded output citing an experience id that does not exist would
       walk into `pending_review` in silence. The demo would then be
       advertising a check the demo does not run.
    2. An unrecorded key raises `KeyError` rather than returning a plausible
       default. That is what makes the prefilter's work visible: `demo-2`
       (below the pay floor) and `demo-5` (excluded company) have no recorded
       output at all, so if the deterministic rules ever stopped dropping them
       first, seeding would fail loudly instead of quietly showing a scored
       job that should never have cost a model call.

    Recorded payloads go through the schema the caller asked for, the same way
    the real client parses a reply, so a typo in the fixture fails here rather
    than becoming a mistyped blob in the database.
    """

    def __init__(self, recorded: dict):
        self.recorded = recorded
        # Set by the caller before each job. `structured` also takes
        # `external_id` directly; this attribute is how `seed_demo_db` gets the
        # id to the mock without threading a demo-only argument through the
        # graph, which would mean the demo no longer ran the real graph.
        self.external_id = None
        self.calls: list[str] = []

    def structured(self, *, node: str, run_id, system: str, user: str,
                   schema=None, validate=None, external_id=None):
        key = f"{node}:{external_id or self.external_id}"
        if key not in self.recorded:
            raise KeyError(
                f"no recorded output for {key}. Demo mode never calls a real "
                f"model, so an unrecorded job cannot be scored.")
        self.calls.append(key)
        model = schema or _SCHEMA_BY_NODE.get(node)
        if model is None:
            raise ValueError(f"no schema to parse a {node!r} reply with")
        result = model.model_validate_json(json.dumps(self.recorded[key]))
        if validate is not None:
            try:
                validate(result)
            except ValueError as e:
                # The real client answers a rejection with a repair turn and
                # only gives up after three of them, raising PermanentLLMError.
                # A recording cannot be repaired -- asking again returns the
                # same bytes -- so the first rejection *is* the third, and this
                # raises what the client would have raised. That matters:
                # `run_match_for_version` maps PermanentLLMError to
                # `permanent_error`, so a bad recorded output lands in the same
                # terminal state a bad real reply would, instead of escaping as
                # a ValueError that no caller catches and killing the seed.
                raise PermanentLLMError(
                    f"recorded {node} output for {key} was rejected: {e}") from e
        return result


def _load(name: str):
    path = DEMO_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Demo fixtures live in the repo's demo/ "
            f"directory; run demo mode from a source checkout.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_demo_profile() -> Profile:
    return load_profile(str(DEMO_DIR / "demo_profile.yaml"))


def seed_demo_db(db_path: str) -> tuple[str, Profile]:
    """Collect (from fixtures) -> prefilter -> match -> brief, with no network.

    Everything after the fixture load is the production path: the same
    `prefilter.run_prefilter`, the same `db.set_status` state machine, the same
    compiled graph. Only the two LLM calls are replayed.
    """
    conn = db.connect(db_path)
    db.init_schema(conn)
    profile = load_demo_profile()
    db.upsert_companies(conn, DEMO_COMPANIES)

    llm = MockLLM(_load("recorded_outputs.json"))
    for raw in _load("demo_jobs.json"):
        job = NormalizedJob(**raw)
        _, vid = db.upsert_job(conn, job)
        if vid is None:                       # unchanged since a previous seed
            continue
        results = prefilter.run_prefilter(job, profile)
        db.record_filter_results(conn, vid, results)
        db.set_status(conn, vid, prefilter.decide(results))

    for row in db.get_versions_by_status(conn, "ready_for_match"):
        llm.external_id = conn.execute(
            "SELECT j.external_id e FROM jobs j JOIN job_versions jv "
            "ON jv.job_id = j.id WHERE jv.id=?", (row["id"],)).fetchone()["e"]
        # brief_enabled is explicit: the low-level entry point defaults it off,
        # and a demo with no briefs would show an empty half of the panel.
        run_match_for_version(conn, llm, profile, row,
                              threshold=DEMO_THRESHOLD, max_auto_retries=3,
                              brief_enabled=True)
    conn.close()
    return db_path, profile


def run_demo(*, serve_panel: bool = True, host: str = "127.0.0.1",
             port: int = 8000) -> str:
    tmp = tempfile.mkdtemp(prefix="offerpilot-demo-")
    db_path = os.path.join(tmp, "demo.db")
    _, profile = seed_demo_db(db_path)
    print(f"demo database seeded at {db_path}")
    print("synthetic data only - no API key used, no network calls made")
    if serve_panel:
        from offerpilot.panel import app as panel_app
        # Resolved before the banner: announcing an address the panel then
        # refuses to bind would read as a crash rather than as the refusal.
        host = panel_app.require_loopback(host)
        print(f"review panel on http://{host}:{port}  (ctrl-c to stop)")
        panel_app.serve(db_path, profile, host=host, port=port)
    return db_path
