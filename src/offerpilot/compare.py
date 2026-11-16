"""Run pipeline versions side by side on one blind-labeled database.

Each version gets its own copy of the labeled database, reset to `new` and
prefiltered from scratch under that version's rules, then matched and scored
by the same eval harness. The labeled source file is only ever read, so the
labels both runs are measured against are byte-for-byte the same.
"""

import copy
import os
import sqlite3

from offerpilot import cli, prefilter
from offerpilot.evaluate import run_eval
from offerpilot.models import NormalizedJob
from offerpilot.store import db


def reset_versions(conn, profile, version: str) -> int:
    """Forget any earlier run on this copy and prefilter again under `version`.

    Writes `new` directly instead of going through `db.set_status`: the state
    machine has no edge back to `new`, on purpose, for real runs. This is an
    experiment copy, and starting it over is the point.
    """
    conn.execute("DELETE FROM filter_results")
    conn.execute("DELETE FROM review_items")
    conn.execute("UPDATE job_versions SET status='new', attempt_count=0, "
                 "processing_started_at=NULL")
    conn.commit()
    rows = conn.execute(
        "SELECT jv.*, j.source, j.external_id, j.company_id, j.canonical_url "
        "FROM job_versions jv JOIN jobs j ON j.id = jv.job_id ORDER BY jv.id"
    ).fetchall()
    for row in rows:
        job = NormalizedJob(
            source=row["source"], external_id=row["external_id"],
            company_id=row["company_id"], title=row["title"] or "",
            location=row["location"] or "", url=row["url"] or "",
            canonical_url=row["canonical_url"],
            description_text=row["description_text"] or "",
            posted_at=row["posted_at"])
        results = prefilter.run_prefilter(job, profile)
        db.record_filter_results(conn, row["id"], results)
        db.set_status(conn, row["id"], prefilter.decide(results, version))
    return len(rows)


def _copy_db(src_path: str, dst_path: str) -> None:
    if os.path.exists(dst_path):
        os.remove(dst_path)
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()


def run_comparison(eval_db_path: str, cfg: dict, profile, *, make_llm,
                   versions=("v1", "v2"), out_dir: str,
                   profile_path: str | None = None, run_meta=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(eval_db_path))[0]
    eval_cfg = cfg.get("eval") or {}
    results = {}
    for version in versions:
        path = os.path.join(out_dir, f"{stem}-{version}.db")
        _copy_db(eval_db_path, path)
        conn = db.connect(path)
        db.init_schema(conn)
        reset_versions(conn, profile, version)
        vcfg = copy.deepcopy(cfg)
        vcfg.setdefault("pipeline", {})["version"] = version
        counts = cli.cmd_match(conn, vcfg, profile, make_llm(conn),
                               run_meta=run_meta)
        results[version] = run_eval(
            conn, profile, profile_path=profile_path,
            results_dir=eval_cfg.get("results_dir", out_dir),
            precision_at=eval_cfg.get("precision_at", [5, 10]),
            extra={"pipeline_version": version, "match_counts": counts,
                   "status_counts": cli.cmd_status(conn)})
        conn.close()
    return results
