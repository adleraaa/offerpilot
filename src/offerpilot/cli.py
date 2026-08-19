import argparse
import os
from offerpilot.config import load_config, config_hash, git_commit
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


def cmd_collect(conn, cfg, profile, limit=None) -> dict:
    db.upsert_companies(conn, cfg.get("companies", []))
    inserted = errors = seen = 0
    for company in cfg.get("companies", []):
        if limit is not None and seen >= limit:
            break
        try:
            jobs = _collect_company(company, cfg)
        except Exception as e:
            print(f"[collect] {company['id']} failed: {e}")
            errors += 1
            continue
        for job in jobs:
            if limit is not None and seen >= limit:
                break
            seen += 1
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


def cmd_match(conn, cfg, profile, llm, limit=None, run_meta=None) -> dict:
    counts: dict[str, int] = {}
    threshold = cfg["match"]["score_threshold"]
    retries = cfg["match"]["max_auto_retries"]
    done = 0
    for row in db.get_versions_by_status(conn, "ready_for_match"):
        if limit is not None and done >= limit:
            break
        try:
            final = run_match_for_version(conn, llm, profile, row,
                                          threshold=threshold,
                                          max_auto_retries=retries,
                                          run_meta=run_meta)
        except SpendCapExceeded as e:
            print(f"[match] stopped: {e}")
            break
        except Exception as e:                      # one bad job, not the batch
            print(f"[match] job_version {row['id']} failed: "
                  f"{type(e).__name__}: {e}")
            counts["error"] = counts.get("error", 0) + 1
            done += 1
            continue
        counts[final] = counts.get(final, 0) + 1
        done += 1
    return counts


def cmd_status(conn) -> dict:
    rows = conn.execute("SELECT status, COUNT(*) c FROM job_versions "
                        "GROUP BY status").fetchall()
    return {r["status"]: r["c"] for r in rows}


def cmd_retry(conn, profile) -> dict:
    stale = db.sweep_stale_matching(conn)
    reset = 0
    for row in db.get_versions_by_status(conn, "permanent_error"):
        db.set_status(conn, row["id"], "ready_for_match")
        conn.execute("UPDATE job_versions SET attempt_count=0 WHERE id=?",
                     (row["id"],))
        conn.commit()
        reset += 1
    return {"reset": reset, "stale_swept": stale}


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="offerpilot",
        description="Human-in-the-loop job-search pipeline. Collects public "
                    "postings, filters them deterministically, scores the "
                    "survivors against your profile, and queues them for your "
                    "approval. Nothing is ever sent to an employer.")
    p.add_argument("command",
                   choices=["collect", "match", "status", "retry"],
                   help="collect: pull postings from configured ATS boards. "
                        "match: score ready jobs with the LLM. "
                        "status: counts by pipeline status. "
                        "retry: reset errored jobs and sweep orphans.")
    p.add_argument("--db", default="data/offerpilot.db",
                   help="SQLite path (default: %(default)s)")
    p.add_argument("--config", default="config.yaml",
                   help="YAML config (default: %(default)s)")
    p.add_argument("--profile", default="profile.yaml",
                   help="candidate profile YAML (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N jobs (collect, match)")
    args = p.parse_args(argv)

    # Read config first: a missing config must fail before we create a DB.
    cfg = load_config(args.config, strict=(args.command in {"collect", "match"}))
    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = db.connect(args.db)
    db.init_schema(conn)

    if args.command == "match":
        if not os.path.exists(args.profile):
            print("profile.yaml not found - refusing to spend LLM budget "
                  "scoring against the synthetic example profile. Create "
                  "profile.yaml (copy profile.example.yaml) first.")
            raise SystemExit(1)
        profile = load_profile(args.profile)
    else:
        if os.path.exists(args.profile):
            profile_path = args.profile
        else:
            profile_path = "profile.example.yaml"
            print("[warn] profile.yaml not found; using profile.example.yaml "
                  "for prefiltering.")
        profile = load_profile(profile_path)

    if args.command == "collect":
        print(cmd_collect(conn, cfg, profile, limit=args.limit))
    elif args.command == "match":
        db.sweep_stale_matching(conn)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("DEEPSEEK_API_KEY is not set; refusing to run match")
            raise SystemExit(1)
        llm = LLMClient(conn, cfg["llm"], api_key)
        run_meta = {"git_commit": git_commit(), "config_hash": config_hash(cfg)}
        print(cmd_match(conn, cfg, profile, llm, limit=args.limit,
                        run_meta=run_meta))
    elif args.command == "status":
        print(cmd_status(conn))
    elif args.command == "retry":
        print(cmd_retry(conn, profile))
