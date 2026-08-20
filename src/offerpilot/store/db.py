import hashlib
import json
import sqlite3
from offerpilot.labels import LABEL_SOURCES
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
    "permanent_error": {"ready_for_match"},   # manual reset, spec section
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
    migrate(conn)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def migrate(conn: sqlite3.Connection) -> None:
    """Additive, idempotent column adds for DBs created before Week 2."""
    review_cols = _columns(conn, "review_items")
    if "edited_brief_json" not in review_cols:
        conn.execute("ALTER TABLE review_items ADD COLUMN edited_brief_json TEXT")
    if "edited_at" not in review_cols:
        conn.execute("ALTER TABLE review_items ADD COLUMN edited_at TEXT")
    if "notes" not in _columns(conn, "labels"):
        conn.execute("ALTER TABLE labels ADD COLUMN notes TEXT")
    conn.commit()


def _content_hash(job: NormalizedJob) -> str:
    payload = json.dumps([job.title, job.location, job.description_text],
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def upsert_job(conn: sqlite3.Connection, job: NormalizedJob) -> tuple[int, int | None]:
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


def get_versions_by_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
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


def sweep_stuck_new(conn, profile) -> int:
    """Re-prefilter job versions orphaned at status='new'.

    `collect` writes the version first and prefilters second, so anything
    that throws in between (a bad posting, a crash, a Ctrl-C) leaves a row at
    'new' that no command looks at again: `match` reads 'ready_for_match' and
    the prefilter only ever runs on freshly collected jobs. This is the
    recovery path. It touches nothing but 'new' rows, so re-running it is
    harmless.

    Returns how many rows it actually moved. A row that cannot be
    reconstructed or prefiltered is left at 'new' rather than raising: this
    runs at the start of every `collect`, and one unrecoverable row must not
    take the command with it. Such rows stay visible in `status` as `new`.
    """
    from offerpilot import prefilter
    from offerpilot.models import NormalizedJob

    rows = conn.execute(
        "SELECT jv.*, j.source, j.external_id, j.company_id, j.canonical_url "
        "FROM job_versions jv JOIN jobs j ON j.id = jv.job_id "
        "WHERE jv.status='new'").fetchall()
    swept = 0
    for row in rows:
        try:
            job = NormalizedJob(
                source=row["source"], external_id=row["external_id"],
                company_id=row["company_id"], title=row["title"],
                location=row["location"] or "", url=row["url"],
                canonical_url=row["canonical_url"],
                description_text=row["description_text"] or "",
                posted_at=row["posted_at"])
            results = prefilter.run_prefilter(job, profile)
        except Exception:
            continue
        record_filter_results(conn, row["id"], results)
        set_status(conn, row["id"], prefilter.decide(results))
        swept += 1
    return swept


def record_filter_results(conn, version_id: int,
                          results: list[FilterResult]) -> None:
    conn.executemany(
        "INSERT INTO filter_results(job_version_id, outcome, rule, "
        "extracted_value, reason) VALUES(?,?,?,?,?)",
        [(version_id, r.outcome, r.rule, r.extracted_value, r.reason)
         for r in results])
    conn.commit()


def upsert_companies(conn: sqlite3.Connection, companies: list[dict]) -> int:
    """Record config-declared companies; returns how many were written.

    `companies` comes from user-authored YAML that nothing schema-validates,
    so entries are tolerated rather than trusted.  A bare `name:` parses to
    None -- the key exists, so a `.get(k, default)` default never fires --
    and companies.name is NOT NULL, so the name falls back to the id.  An
    entry with no usable id is skipped rather than aborting the write for its
    well-formed siblings; the caller reports the shortfall.
    """
    rows = []
    for c in companies:
        cid = c.get("id") if isinstance(c, dict) else None
        if not cid:
            continue
        rows.append((str(cid), c.get("name") or str(cid)))
    conn.executemany(
        "INSERT INTO companies(id, name) VALUES(?,?) "
        "ON CONFLICT(id) DO UPDATE SET name=excluded.name", rows)
    conn.commit()
    return len(rows)


def record_label(conn: sqlite3.Connection, version_id: int, *,
                 label_source: str,
                 fit_label: str | None = None, action_label: str | None = None,
                 rejection_reason: str | None = None,
                 notes: str | None = None) -> int:
    if label_source not in LABEL_SOURCES:
        raise ValueError(f"unknown label_source {label_source!r}")
    cur = conn.execute(
        "INSERT INTO labels(job_version_id, label_source, fit_label, "
        "action_label, rejection_reason, notes) VALUES(?,?,?,?,?,?) RETURNING id",
        (version_id, label_source, fit_label, action_label, rejection_reason,
         notes))
    label_id = cur.fetchone()["id"]
    conn.commit()
    return label_id


def get_labels(conn: sqlite3.Connection, *, version_id: int | None = None,
               label_source: str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM labels WHERE 1=1"
    params: list = []
    if version_id is not None:
        sql += " AND job_version_id=?"
        params.append(version_id)
    if label_source is not None:
        sql += " AND label_source=?"
        params.append(label_source)
    return conn.execute(sql + " ORDER BY id", params).fetchall()


_REVIEW_SELECT = """
SELECT ri.id AS review_item_id, ri.job_version_id, ri.match_json,
       ri.total_score, ri.brief_json, ri.edited_brief_json, ri.edited_at,
       ri.created_at,
       jv.title, jv.location, jv.url, jv.description_text, jv.status,
       j.company_id, j.source, j.canonical_url
FROM review_items ri
JOIN job_versions jv ON jv.id = ri.job_version_id
JOIN jobs j ON j.id = jv.job_id
"""


def get_review_queue(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        _REVIEW_SELECT + " WHERE jv.status='pending_review' "
        "ORDER BY ri.total_score DESC, ri.job_version_id").fetchall()


def get_review_item(conn: sqlite3.Connection,
                    version_id: int) -> sqlite3.Row | None:
    return conn.execute(
        _REVIEW_SELECT + " WHERE ri.job_version_id=?", (version_id,)).fetchone()


def save_brief(conn: sqlite3.Connection, version_id: int,
               brief_json: str) -> None:
    conn.execute("UPDATE review_items SET brief_json=? WHERE job_version_id=?",
                 (brief_json, version_id))
    conn.commit()


def save_edited_brief(conn: sqlite3.Connection, version_id: int,
                      brief_json: str) -> None:
    conn.execute(
        "UPDATE review_items SET edited_brief_json=?, "
        "edited_at=datetime('now') WHERE job_version_id=?",
        (brief_json, version_id))
    conn.commit()


def get_blind_candidates(conn: sqlite3.Connection, limit: int = 50, *,
                         unlabeled_only: bool = True) -> list[sqlite3.Row]:
    sql = """
    SELECT jv.id, jv.title, jv.location, jv.description_text, jv.status,
           j.company_id, j.canonical_url
    FROM job_versions jv JOIN jobs j ON j.id = jv.job_id
    """
    if unlabeled_only:
        sql += ("WHERE NOT EXISTS (SELECT 1 FROM labels l "
                "WHERE l.job_version_id = jv.id AND l.label_source='blind_eval') ")
    return conn.execute(sql + "ORDER BY jv.id LIMIT ?", (limit,)).fetchall()
