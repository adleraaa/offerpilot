import json
import re
from offerpilot.models import MatchResult, total_score
from offerpilot.profile import Profile
from offerpilot.prompts import MATCH_SYSTEM, MATCH_USER
from offerpilot.store import db
from offerpilot.llm import RetryableLLMError, PermanentLLMError, SpendCapExceeded


_DELIM_RE = re.compile(r"<\s*/?\s*untrusted_job_posting[^>]*>", re.I)


def _sanitize(text: str) -> str:
    return _DELIM_RE.sub("[tag-removed]", text or "")


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


def run_match_for_version(conn, llm, profile: Profile, version_row,
                          threshold: int, max_auto_retries: int,
                          run_meta: dict | None = None) -> str:
    vid = version_row["id"]
    attempt = version_row["attempt_count"] + 1
    conn.execute("UPDATE job_versions SET attempt_count=? WHERE id=?",
                 (attempt, vid))
    db.set_status(conn, vid, "matching")
    run_id = _start_run(conn, vid, run_meta)
    system, user = build_prompts(version_row, profile)
    prompt_input = json.dumps({"system": system, "user": user})
    try:
        result: MatchResult = llm.structured(
            node="match", run_id=run_id, system=system, user=user,
            schema=MatchResult)
        bad = [e.source_id for e in result.evidence
               if e.source_id not in profile.experience_ids()]
        if bad:
            raise PermanentLLMError(f"invented evidence ids: {bad}")
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

    _log_step(conn, run_id, "match", attempt, "ok",
              input=prompt_input, output=result.model_dump_json())

    current_status = conn.execute(
        "SELECT status FROM job_versions WHERE id=?", (vid,)).fetchone()["status"]
    if current_status != "matching":
        _log_step(conn, run_id, "gate", attempt, "stale_state",
                  error=f"expected status 'matching', found {current_status!r}")
        _finish_run(conn, run_id, "stale_state")
        return current_status

    score = total_score(result)
    if result.eligibility == "fail":
        final = "eligibility_failed"
    elif score < threshold:
        final = "scored_low"
    else:
        conn.execute(
            "INSERT INTO review_items(job_version_id, match_json, "
            "total_score) VALUES(?,?,?)",
            (vid, result.model_dump_json(), score))
        conn.commit()
        final = "pending_review"
    db.set_status(conn, vid, final)
    _finish_run(conn, run_id, "ok")
    return final
