import json
import re
import unicodedata
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
    validate_evidence = make_evidence_validator(profile, threshold)
    try:
        result: MatchResult = llm.structured(
            node="match", run_id=run_id, system=system, user=user,
            schema=MatchResult, validate=validate_evidence)
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
