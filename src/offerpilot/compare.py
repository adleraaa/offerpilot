"""Run pipeline versions side by side on one blind-labeled database.

Each version gets its own copy of the labeled database, reset to `new` and
prefiltered from scratch under that version's rules, then matched and scored
by the same eval harness. The labeled source file is only ever read, so the
labels both runs are measured against are byte-for-byte the same.
"""

import copy
import os
import sqlite3

from offerpilot import cli, evalset, prefilter
from offerpilot.evaluate import run_eval
from offerpilot.models import NormalizedJob
from offerpilot.store import db

_ALL_VERSIONS = (
    "SELECT jv.*, j.source, j.external_id, j.company_id, j.canonical_url "
    "FROM job_versions jv JOIN jobs j ON j.id = jv.job_id ORDER BY jv.id")


def _job_from_row(row) -> NormalizedJob:
    return NormalizedJob(
        source=row["source"], external_id=row["external_id"],
        company_id=row["company_id"], title=row["title"] or "",
        location=row["location"] or "", url=row["url"] or "",
        canonical_url=row["canonical_url"],
        description_text=row["description_text"] or "",
        posted_at=row["posted_at"])


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
    rows = conn.execute(_ALL_VERSIONS).fetchall()
    for row in rows:
        results = prefilter.run_prefilter(_job_from_row(row), profile)
        db.record_filter_results(conn, row["id"], results)
        db.set_status(conn, row["id"], prefilter.decide(results, version))
    return len(rows)


def prefilter_split(conn, profile) -> dict:
    """What each pipeline version's prefilter does with every job in `conn`.

    No model call and no write: the rules run in memory on the stored posting
    text, so this works on any collected database, labeled or not, without a
    key. It answers the first question about v2 before any money is spent on
    matching: how many jobs reach the model that v1 removed, and which rule
    had removed them. `student_level` repeats the split for intern and
    new-grad titles, the jobs the candidate is actually looking for.
    """
    def tally():
        return {"ready_for_match": 0, "filtered_out": 0}

    out = {"total": 0, "v1": tally(), "v2": tally(),
           "student_level": {"total": 0, "v1": tally(), "v2": tally()},
           "kept_by_v2_only": {"total": 0, "student_level": 0, "by_rule": {}},
           "removed_by_v2": {}}
    soft = prefilter.SOFT_RULES["v2"]
    for row in conn.execute(_ALL_VERSIONS).fetchall():
        results = prefilter.run_prefilter(_job_from_row(row), profile)
        fails = sorted({r.rule for r in results if r.outcome == "fail"})
        student = evalset.is_student_level(row["title"])
        decided = {v: prefilter.decide(results, v) for v in ("v1", "v2")}
        out["total"] += 1
        out["student_level"]["total"] += int(student)
        for version, status in decided.items():
            out[version][status] += 1
            if student:
                out["student_level"][version][status] += 1
        if (decided["v1"], decided["v2"]) == ("filtered_out",
                                              "ready_for_match"):
            kept = out["kept_by_v2_only"]
            kept["total"] += 1
            kept["student_level"] += int(student)
            rule = "+".join(fails)
            kept["by_rule"][rule] = kept["by_rule"].get(rule, 0) + 1
        if decided["v2"] == "filtered_out":
            rule = "+".join(f for f in fails if f not in soft)
            out["removed_by_v2"][rule] = out["removed_by_v2"].get(rule, 0) + 1
    return out


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
