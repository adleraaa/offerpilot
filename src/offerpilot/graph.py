"""The match pipeline, compiled as a LangGraph `StateGraph`.

Topology:

    START -> match -(gate)-> brief -> persist -> END
                   -(gate)----------> persist -> END

`gate` is a conditional edge, not a node: it is pure routing and writes
nothing. The nodes do all the writing, and every state transition still goes
through `db.set_status`, so `db.ALLOWED_TRANSITIONS` stays the single arbiter
of what may follow what.

`run_match_for_version` keeps its Week 1 signature, its transaction
discipline and its error mapping; only the middle -- one LLM call and a
gate -- is now the compiled graph.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from offerpilot.brief import ApplicationBrief, generate_brief
from offerpilot.models import MatchResult, total_score
from offerpilot.profile import Profile
from offerpilot.prompts import MATCH_SYSTEM, MATCH_USER
from offerpilot.store import db
from offerpilot.llm import (AuthLLMError, RetryableLLMError, PermanentLLMError,
                            SpendCapExceeded)


# A forged delimiter only has to *look* like the real one to the model, so
# matching it by exact bytes loses. Enumerating homoglyphs one at a time loses
# too -- the fullwidth pair has a small-form sibling one codepoint away, and
# the tag name can be broken up by characters that render as nothing at all.
# Instead: fold compatibility variants (NFKC maps U+FF1C and U+FE64 alike to
# "<"), delete every format character (Cf covers ZWSP/ZWNJ/BOM, SOFT HYPHEN and
# INVISIBLE PLUS), and let the tag-name pattern tolerate stray separators such
# as COMBINING GRAPHEME JOINER between its letters.
_TAG = "untrusted_job_posting"
# Bracket lookalikes accepted alongside ASCII "<"/">". NFKC already folds the
# fullwidth (U+FF1C) and small-form (U+FE64) pairs down to ASCII; none of the
# ones below ever reach ASCII, so they are listed out.
_OPENERS = "<\u2039\u00ab\u3008\u300a\u2329\u276e\u2770"
_CLOSERS = ">\u203a\u00bb\u3009\u300b\u232a\u276f\u2771"

# Junk permitted between the letters of the tag name. Excluding alphanumerics
# and brackets makes every gap disjoint from the literal that follows it, so
# the pattern matches without backtracking.
_GAP = r"[^0-9A-Za-z" + re.escape(_OPENERS + _CLOSERS) + r"]{0,4}"
# The underscores are dropped from the literal and absorbed by _GAP, so
# "untrusted job posting" and "untrustedjobposting" are caught as well.
_NAME = _GAP.join(re.escape(c) for c in _TAG.replace("_", ""))

# The closing bracket is optional: an unclosed "<untrusted_job_posting" is
# still a forgery attempt. When it is present, only 64 characters of attribute
# junk may precede it, so a stray bracket far downstream cannot make the
# substitution swallow real job text.
_DELIM_RE = re.compile(
    "[" + re.escape(_OPENERS) + "]" + _GAP + "/?" + _GAP + _NAME
    + "(?:[^" + re.escape(_CLOSERS) + "]{0,64}[" + re.escape(_CLOSERS) + "])?",
    re.I)


def _sanitize(text: str) -> str:
    folded = unicodedata.normalize(
        "NFKC", "".join(ch for ch in (text or "")
                        if unicodedata.category(ch) != "Cf"))
    return _DELIM_RE.sub("[tag-removed]", folded)


def build_prompts(job_row, profile: Profile):
    user = MATCH_USER.format(
        profile_json=profile.model_dump_json(indent=2),
        title=_sanitize(job_row["title"]),
        location=_sanitize(job_row["location"] or ""),
        description=_sanitize(job_row["description_text"]))
    return MATCH_SYSTEM, user


def _start_run(conn, version_id, run_meta=None):
    meta = run_meta or {}
    cur = conn.execute(
        "INSERT INTO runs(run_type, job_version_id, status, git_commit, "
        "config_hash) VALUES('graph', ?, 'running', ?, ?) RETURNING id",
        (version_id, meta.get("git_commit"), meta.get("config_hash")))
    run_id = cur.fetchone()["id"]
    conn.commit()
    return run_id


def _finish_run(conn, run_id, status):
    conn.execute("UPDATE runs SET status=?, completed_at=datetime('now') "
                 "WHERE id=?", (status, run_id))
    conn.commit()


def _log_step(conn, run_id, node, attempt, status, input=None, output=None,
             error=None):
    conn.execute(
        "INSERT INTO run_steps(run_id, node, attempt, status, input_json, "
        "output_json, error, completed_at) VALUES(?,?,?,?,?,?,?,datetime('now'))",
        (run_id, node, attempt, status, input, output, error))
    conn.commit()


def make_evidence_validator(profile: Profile, threshold: int):
    """Reject semantically wrong MatchResults by raising ValueError.

    Handed to `LLMClient.structured(validate=...)`, so a rejection buys the
    model a repair turn inside the client's 3-attempt budget instead of
    parking the version on `permanent_error` at the first offence. The
    messages below are written to be read by the model: they name the mistake
    and the legal alternatives.
    """
    valid = profile.experience_ids()

    def _validate(result: MatchResult) -> None:
        bad = [e.source_id for e in result.evidence if e.source_id not in valid]
        if bad:
            raise ValueError(
                f"evidence source_id {bad} do not exist. Valid ids are: "
                f"{sorted(valid)}. Cite only these, or return an empty "
                f"evidence list and a lower score.")
        # Only jobs the model considers viable can reach review, so only they
        # need profile citations. When eligibility is "fail" the posting is
        # the evidence -- MatchResult already requires
        # eligibility_evidence_excerpt for that verdict -- and citing no
        # experience is the natural, correct answer. Demanding one anyway
        # threw a well-formed result away and parked the version on
        # permanent_error, which db.ALLOWED_TRANSITIONS only lets a human
        # reverse by hand. Keep this guard if this validator moves into the
        # LLM repair loop.
        if (result.eligibility != "fail" and not result.evidence
                and total_score(result) >= threshold):
            raise ValueError(
                "a score at or above the review threshold must cite at least "
                "one evidence source_id from the profile. Either cite the "
                "experience that justifies the score, or lower the subscores.")

    return _validate


@dataclass
class GraphContext:
    """Everything a node needs that is not per-job state.

    It rides in the state dict rather than being closed over, so the compiled
    graph can stay a module-level singleton: compiled once, reused for any
    connection, profile or threshold.

    It carried `max_auto_retries` and no node ever read it. That is not a
    harmless spare field: retries are decided *outside* the graph, in
    `run_match_for_version`'s exception handlers, from its own parameter. A
    field here saying otherwise invites the next reader to branch on it inside
    a node and put the retry budget in two places.
    """
    conn: Any
    llm: Any
    profile: Profile
    threshold: int
    brief_enabled: bool = True


class GraphState(TypedDict, total=False):
    ctx: GraphContext
    version_id: int
    run_id: int
    attempt: int
    job_row: Any
    match: Optional[MatchResult]
    brief: Optional[ApplicationBrief]
    final_status: Optional[str]
    # `final_status` is the version's status, which on the stale path is
    # whatever another writer set -- so it cannot also carry "how did this run
    # end". `run_status` does, and `run_match_for_version` writes it to
    # runs.status. Keeping them apart is what stops a run that lost a race
    # from being filed as "ok".
    run_status: Optional[str]


def _match_node(state: GraphState) -> dict:
    """The one paid call of the pipeline: score this job against the profile."""
    ctx = state["ctx"]
    system, user = build_prompts(state["job_row"], ctx.profile)
    result: MatchResult = ctx.llm.structured(
        node="match", run_id=state["run_id"], system=system, user=user,
        schema=MatchResult,
        validate=make_evidence_validator(ctx.profile, ctx.threshold))
    _log_step(ctx.conn, state["run_id"], "match", state["attempt"], "ok",
              input=json.dumps({"system": system, "user": user}),
              output=result.model_dump_json())
    return {"match": result}


def _gate(state: GraphState) -> str:
    """Pure routing: return the name of the next node. Writes nothing."""
    ctx, match = state["ctx"], state["match"]
    if match.eligibility == "fail" or total_score(match) < ctx.threshold:
        return "persist"
    if not ctx.brief_enabled:
        return "persist"
    try:
        make_evidence_validator(ctx.profile, ctx.threshold)(match)
    except ValueError:
        # `persist` owns this rejection -- it logs the reason and parks the
        # version on permanent_error. All the gate does is decline to spend a
        # second LLM call writing a brief out of evidence that is about to be
        # thrown away.
        return "persist"
    return "brief"


def _brief_node(state: GraphState) -> dict:
    ctx = state["ctx"]
    try:
        brief = generate_brief(ctx.llm, state["run_id"], state["job_row"],
                               ctx.profile, state["match"])
        if not isinstance(brief, ApplicationBrief):
            # review_items.brief_json is read back as an ApplicationBrief by
            # the panel, so a client that returns something else has to fail
            # here rather than leave a mistyped blob in the column.
            raise TypeError(f"expected an ApplicationBrief, got "
                            f"{type(brief).__name__}")
    except Exception as e:
        # A missing brief must never cost us the match result: the match is
        # what a reviewer needs, the brief is only what makes reviewing
        # faster. Broad on purpose -- every failure here is survivable.
        _log_step(ctx.conn, state["run_id"], "brief", state["attempt"],
                  "brief_failed", error=f"{type(e).__name__}: {e}")
        return {"brief": None}
    _log_step(ctx.conn, state["run_id"], "brief", state["attempt"], "ok",
              output=brief.model_dump_json())
    return {"brief": brief}


def _persist_node(state: GraphState) -> dict:
    """The only node that writes job state, and it re-verifies before it does."""
    ctx, vid, match = state["ctx"], state["version_id"], state["match"]
    conn = ctx.conn
    current = conn.execute("SELECT status FROM job_versions WHERE id=?",
                           (vid,)).fetchone()["status"]
    if current != "matching":
        _log_step(conn, state["run_id"], "gate", state["attempt"],
                  "stale_state",
                  error=f"expected status 'matching', found {current!r}")
        return {"final_status": current, "run_status": "stale_state"}

    # Belt to the client's braces. Handing the validator to
    # `structured(validate=...)` is what buys the model a repair turn, but
    # *calling* it is the client's choice: a client that takes **kwargs and
    # drops `validate` would leave the citation rule with no enforcement path
    # at all, and an invented source_id would reach `pending_review` in
    # silence. Re-checking here costs microseconds and makes the gate a
    # property of this graph rather than of whichever client object was
    # passed in. It sits after the stale-state check on purpose: only there is
    # the status known to still be 'matching', which is the one state
    # db.ALLOWED_TRANSITIONS lets us move to 'permanent_error' from.
    try:
        make_evidence_validator(ctx.profile, ctx.threshold)(match)
    except ValueError as e:
        _log_step(conn, state["run_id"], "gate", state["attempt"],
                  "permanent_error", error=str(e))
        db.set_status(conn, vid, "permanent_error")
        return {"final_status": "permanent_error",
                "run_status": "permanent_error"}

    score = total_score(match)
    if match.eligibility == "fail":
        final = "eligibility_failed"
    elif score < ctx.threshold:
        final = "scored_low"
    else:
        conn.execute(
            "INSERT INTO review_items(job_version_id, match_json, "
            "total_score) VALUES(?,?,?)",
            (vid, match.model_dump_json(), score))
        conn.commit()
        if state.get("brief") is not None:
            db.save_brief(conn, vid, state["brief"].model_dump_json())
        final = "pending_review"
    db.set_status(conn, vid, final)
    return {"final_status": final, "run_status": "ok"}


_GRAPH = None


def build_match_graph():
    """The compiled graph, built once and reused."""
    global _GRAPH
    if _GRAPH is None:
        g = StateGraph(GraphState)
        g.add_node("match", _match_node)
        g.add_node("brief", _brief_node)
        g.add_node("persist", _persist_node)
        g.add_edge(START, "match")
        g.add_conditional_edges("match", _gate,
                                {"brief": "brief", "persist": "persist"})
        g.add_edge("brief", "persist")
        g.add_edge("persist", END)
        _GRAPH = g.compile()
    return _GRAPH


def run_match_for_version(conn, llm, profile: Profile, version_row,
                          threshold: int, max_auto_retries: int,
                          brief_enabled: bool = False,
                          run_meta: dict | None = None) -> str:
    """Run one job version through the graph and return its final status.

    `brief_enabled` is opt-in here on purpose. This is the low-level entry
    point -- one job, one call -- and the brief is a second paid LLM call that
    nothing downstream requires. Whether to spend it is the caller's
    decision, and `cli.cmd_match` makes that decision from config
    (`brief.enabled`, default true), so `python -m offerpilot match` does
    write briefs. The default is load-bearing for the Week 1 tests too: their
    doubles answer every node the same way, so a brief turned on here hands
    them a second validator built for a different schema.

    Everything outside the graph -- the attempt counter, the run row, and the
    mapping from LLM exceptions to statuses -- is unchanged from Week 1.
    """
    vid = version_row["id"]
    attempt = version_row["attempt_count"] + 1
    conn.execute("UPDATE job_versions SET attempt_count=? WHERE id=?",
                 (attempt, vid))
    db.set_status(conn, vid, "matching")
    run_id = _start_run(conn, vid, run_meta)
    # Rebuilt here only so a failing call can be logged with what it was
    # given: `_match_node` builds its own from the same pure function off the
    # same row, and never sees this copy.
    system, user = build_prompts(version_row, profile)
    prompt_input = json.dumps({"system": system, "user": user})
    ctx = GraphContext(conn=conn, llm=llm, profile=profile,
                       threshold=threshold, brief_enabled=brief_enabled)
    try:
        out = build_match_graph().invoke({
            "ctx": ctx, "version_id": vid, "run_id": run_id,
            "attempt": attempt, "job_row": version_row})
    except AuthLLMError as e:
        # A rejected key is not this job's fault: hand the version back
        # unspent, so the batch can resume once the key is fixed. Must be
        # caught before PermanentLLMError, which it subclasses.
        _log_step(conn, run_id, "match", attempt, "auth_error",
                  input=prompt_input, error=str(e))
        conn.execute("UPDATE job_versions SET attempt_count=? WHERE id=?",
                     (attempt - 1, vid))
        conn.commit()
        db.set_status(conn, vid, "retryable_error")
        db.set_status(conn, vid, "ready_for_match")
        _finish_run(conn, run_id, "auth_error")
        raise
    except SpendCapExceeded as e:
        _log_step(conn, run_id, "match", attempt, "spend_cap",
                  input=prompt_input, error=str(e))
        conn.execute("UPDATE job_versions SET attempt_count=? WHERE id=?",
                     (attempt - 1, vid))
        conn.commit()
        db.set_status(conn, vid, "retryable_error")
        db.set_status(conn, vid, "ready_for_match")
        _finish_run(conn, run_id, "spend_cap")
        raise
    except RetryableLLMError as e:
        _log_step(conn, run_id, "match", attempt, "retryable_error",
                  input=prompt_input, error=str(e))
        if attempt >= max_auto_retries:
            db.set_status(conn, vid, "retryable_error")
            db.set_status(conn, vid, "permanent_error")
            _finish_run(conn, run_id, "permanent_error")
            return "permanent_error"
        db.set_status(conn, vid, "retryable_error")
        db.set_status(conn, vid, "ready_for_match")
        _finish_run(conn, run_id, "retryable_error")
        return "ready_for_match"
    except PermanentLLMError as e:
        _log_step(conn, run_id, "match", attempt, "permanent_error",
                  input=prompt_input, error=str(e))
        db.set_status(conn, vid, "permanent_error")
        _finish_run(conn, run_id, "permanent_error")
        return "permanent_error"

    _finish_run(conn, run_id, out.get("run_status") or "ok")
    return out["final_status"]
