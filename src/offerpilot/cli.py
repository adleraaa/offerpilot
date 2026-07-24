import argparse
import os
from offerpilot.config import load_config
from offerpilot.profile import load_profile
from offerpilot.store import db
from offerpilot import prefilter
from offerpilot.collectors import greenhouse, lever
from offerpilot.graph import run_match_for_version
from offerpilot.llm import LLMClient, SpendCapExceeded


def _collect_company(company, cfg):
    if company["ats"] == "greenhouse":
        return greenhouse.parse(greenhouse.fetch(company["ats_slug"]),
                                company_id=company["id"])
    if company["ats"] == "lever":
        return lever.parse(lever.fetch(company["ats_slug"]),
                           company_id=company["id"])
    return []


def cmd_collect(conn, cfg, profile) -> dict:
    inserted = errors = 0
    for company in cfg.get("companies", []):
        try:
            jobs = _collect_company(company, cfg)
        except Exception as e:
            print(f"[collect] {company['id']} failed: {e}")
            errors += 1
            continue
        for job in jobs:
            try:
                _, vid = db.upsert_job(conn, job)
                if vid is None:
                    continue
                inserted += 1
                results = prefilter.run_prefilter(job, profile)
                db.record_filter_results(conn, vid, results)
                db.set_status(conn, vid, prefilter.decide(results))
            except Exception as e:
                print(f"[collect] job {job.external_id} ({company['id']}) failed: {e}")
                errors += 1
    return {"inserted": inserted,
            "companies": len(cfg.get("companies", [])), "errors": errors}


def cmd_match(conn, cfg, profile, llm) -> dict:
    counts: dict[str, int] = {}
    threshold = cfg["match"]["score_threshold"]
    retries = cfg["match"]["max_auto_retries"]
    for row in db.get_versions_by_status(conn, "ready_for_match"):
        try:
            final = run_match_for_version(conn, llm, profile, row,
                                          threshold=threshold,
                                          max_auto_retries=retries)
        except SpendCapExceeded as e:
            print(f"[match] stopped: {e}")
            break
        counts[final] = counts.get(final, 0) + 1
    return counts


def cmd_status(conn) -> dict:
    rows = conn.execute("SELECT status, COUNT(*) c FROM job_versions "
                        "GROUP BY status").fetchall()
    return {r["status"]: r["c"] for r in rows}


def cmd_retry(conn) -> int:
    db.sweep_stale_matching(conn)
    cur = conn.execute(
        "UPDATE job_versions SET status='ready_for_match', attempt_count=0 "
        "WHERE status='permanent_error'")
    conn.commit()
    return cur.rowcount


def main(argv=None):
    p = argparse.ArgumentParser(prog="offerpilot")
    p.add_argument("command",
                   choices=["collect", "match", "status", "retry"])
    p.add_argument("--db", default="data/offerpilot.db")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--profile", default="profile.yaml")
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = db.connect(args.db)
    db.init_schema(conn)
    cfg = load_config(args.config)
    profile_path = (args.profile if os.path.exists(args.profile)
                    else "profile.example.yaml")
    profile = load_profile(profile_path)

    if args.command == "collect":
        print(cmd_collect(conn, cfg, profile))
    elif args.command == "match":
        db.sweep_stale_matching(conn)
        llm = LLMClient(conn, cfg["llm"],
                        os.environ.get("DEEPSEEK_API_KEY", ""))
        print(cmd_match(conn, cfg, profile, llm))
    elif args.command == "status":
        print(cmd_status(conn))
    elif args.command == "retry":
        print({"reset": cmd_retry(conn)})
