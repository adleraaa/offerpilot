import hashlib
import json
import sqlite3
from offerpilot.models import NormalizedJob, FilterResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies(
  id TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL, external_id TEXT NOT NULL,
  company_id TEXT NOT NULL, canonical_url TEXT NOT NULL,
  first_seen_at TEXT DEFAULT (datetime('now')),
  last_seen_at TEXT DEFAULT (datetime('now')),
  active INTEGER DEFAULT 1,
  UNIQUE(source, external_id));
CREATE TABLE IF NOT EXISTS job_versions(
  id INTEGER PRIMARY KEY,
  job_id INTEGER NOT NULL REFERENCES jobs(id),
  content_hash TEXT NOT NULL,
  title TEXT, location TEXT, url TEXT, description_text TEXT,
  posted_at TEXT,
  collected_at TEXT DEFAULT (datetime('now')),
  status TEXT NOT NULL DEFAULT 'new',
  attempt_count INTEGER NOT NULL DEFAULT 0,
  processing_started_at TEXT);
CREATE TABLE IF NOT EXISTS filter_results(
  id INTEGER PRIMARY KEY,
  job_version_id INTEGER NOT NULL REFERENCES job_versions(id),
  outcome TEXT NOT NULL, rule TEXT NOT NULL,
  extracted_value TEXT, reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY,
  run_type TEXT NOT NULL, job_version_id INTEGER,
  started_at TEXT DEFAULT (datetime('now')),
  completed_at TEXT, status TEXT,
  git_commit TEXT, config_hash TEXT);
CREATE TABLE IF NOT EXISTS run_steps(
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  node TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 1,
  started_at TEXT DEFAULT (datetime('now')),
  completed_at TEXT, status TEXT,
  input_json TEXT, output_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS review_items(
  id INTEGER PRIMARY KEY,
  job_version_id INTEGER NOT NULL REFERENCES job_versions(id),
  match_json TEXT NOT NULL, total_score INTEGER NOT NULL,
  brief_json TEXT, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS labels(
  id INTEGER PRIMARY KEY,
  job_version_id INTEGER NOT NULL REFERENCES job_versions(id),
  label_source TEXT NOT NULL, fit_label TEXT,
  action_label TEXT, rejection_reason TEXT,
  created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS llm_usage(
  id INTEGER PRIMARY KEY,
  run_id INTEGER, node TEXT, model TEXT NOT NULL,
  prompt_tokens INTEGER, completion_tokens INTEGER,
  estimated_cost_usd REAL, created_at TEXT DEFAULT (datetime('now')));
"""

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "new": {"filtered_out", "ready_for_match"},
    "ready_for_match": {"matching"},
    "matching": {"eligibility_failed", "scored_low", "pending_review",
                 "retryable_error", "permanent_error"},
    "retryable_error": {"ready_for_match", "permanent_error"},
    "pending_review": {"approved", "rejected", "saved"},
}


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _content_hash(job: NormalizedJob) -> str:
    payload = json.dumps([job.title, job.location, job.description_text],
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def upsert_job(conn, job: NormalizedJob):
    cur = conn.execute(
        "INSERT INTO jobs(source, external_id, company_id, canonical_url) "
        "VALUES(?,?,?,?) "
        "ON CONFLICT(source, external_id) DO UPDATE SET "
        "last_seen_at=datetime('now'), active=1 "
        "RETURNING id", (job.source, job.external_id, job.company_id,
                         job.canonical_url))
    job_id = cur.fetchone()["id"]
    h = _content_hash(job)
    latest = conn.execute(
        "SELECT content_hash FROM job_versions WHERE job_id=? "
        "ORDER BY id DESC LIMIT 1", (job_id,)).fetchone()
    if latest and latest["content_hash"] == h:
        conn.commit()
        return job_id, None
    cur = conn.execute(
        "INSERT INTO job_versions(job_id, content_hash, title, location, "
        "url, description_text, posted_at) VALUES(?,?,?,?,?,?,?) "
        "RETURNING id",
        (job_id, h, job.title, job.location, job.url,
         job.description_text, job.posted_at))
    version_id = cur.fetchone()["id"]
    conn.commit()
    return job_id, version_id


def set_status(conn, version_id: int, new_status: str) -> None:
    row = conn.execute("SELECT status FROM job_versions WHERE id=?",
                       (version_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown job_version {version_id}")
    current = row["status"]
    if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal transition {current} -> {new_status}")
    stamp = (", processing_started_at=datetime('now')"
             if new_status == "matching" else "")
    conn.execute(f"UPDATE job_versions SET status=?{stamp} WHERE id=?",
                 (new_status, version_id))
    conn.commit()


def get_versions_by_status(conn, status: str):
    return conn.execute(
        "SELECT * FROM job_versions WHERE status=? ORDER BY id",
        (status,)).fetchall()


def sweep_stale_matching(conn, max_age_minutes: int = 15) -> int:
    cur = conn.execute(
        "UPDATE job_versions SET status='ready_for_match', "
        "processing_started_at=NULL WHERE status='matching' AND "
        "processing_started_at < datetime('now', ?)",
        (f"-{max_age_minutes} minutes",))
    conn.commit()
    return cur.rowcount


def record_filter_results(conn, version_id: int,
                          results: list[FilterResult]) -> None:
    conn.executemany(
        "INSERT INTO filter_results(job_version_id, outcome, rule, "
        "extracted_value, reason) VALUES(?,?,?,?,?)",
        [(version_id, r.outcome, r.rule, r.extracted_value, r.reason)
         for r in results])
    conn.commit()
