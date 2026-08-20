import argparse
import json
import os
from offerpilot.config import load_config, config_hash, git_commit
from offerpilot.profile import load_profile
from offerpilot.store import db
from offerpilot import prefilter
from offerpilot.collectors import greenhouse, lever
from offerpilot.graph import run_match_for_version
from offerpilot.llm import LLMClient, SpendCapExceeded, AuthLLMError


# Two commands refuse the profile.example.yaml fallback. The reasons differ;
# the failure mode is the same one, which is confident-looking output about a
# person who does not exist.
REFUSES_EXAMPLE_PROFILE = {
    "match": ("refusing to spend LLM budget scoring against the synthetic "
              "example profile"),
    "eval": ("refusing to score against the synthetic example profile -- its "
             "experience ids belong to nobody, so every id cited by a real "
             "brief would count as ungrounded and the metrics would be junk"),
}


def _company_label(company) -> str:
    """A printable identifier that works even for a malformed config entry."""
    if isinstance(company, dict):
        return str(company.get("id") or "<entry with no id>")
    return repr(company)


def _collect_company(company):
    """Fetch and parse one configured board. Reads only this entry.

    Took `cfg` as a second argument and never looked at it. A parameter every
    caller has to supply and nobody reads is a claim about what this function
    depends on, and it was false: routing is decided by `company["ats"]` and
    nothing else.
    """
    if company["ats"] == "greenhouse":
        return greenhouse.parse(greenhouse.fetch(company["ats_slug"]),
                                company_id=company["id"])
    if company["ats"] == "lever":
        return lever.parse(lever.fetch(company["ats_slug"]),
                           company_id=company["id"])
    return []


def cmd_collect(conn, cfg, profile, limit=None) -> dict:
    companies = cfg.get("companies", [])
    # Recording companies must never cost the batch: one malformed config
    # entry is a per-company error below, not a whole-run abort.
    try:
        written = db.upsert_companies(conn, companies)
        if written < len(companies):
            print(f"[collect] {len(companies) - written} company entries "
                  f"skipped: no 'id'")
    except Exception as e:
        print(f"[collect] could not record companies: {type(e).__name__}: {e}")
    inserted = errors = seen = 0
    for company in companies:
        if limit is not None and seen >= limit:
            break
        try:
            jobs = _collect_company(company)
        except Exception as e:
            print(f"[collect] {_company_label(company)} failed: {e}")
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
                print(f"[collect] job {job.external_id} "
                      f"({_company_label(company)}) failed: {e}")
                errors += 1
    return {"inserted": inserted,
            "companies": len(companies), "errors": errors}


def cmd_match(conn, cfg, profile, llm, limit=None, run_meta=None) -> dict:
    counts: dict[str, int] = {}
    threshold = cfg["match"]["score_threshold"]
    retries = cfg["match"]["max_auto_retries"]
    # The brief is the pipeline's second LLM call. `run_match_for_version`
    # leaves that spend to its caller; this is the caller, and config decides.
    brief_enabled = (cfg.get("brief") or {}).get("enabled", True)
    done = 0
    for row in db.get_versions_by_status(conn, "ready_for_match"):
        if limit is not None and done >= limit:
            break
        try:
            final = run_match_for_version(conn, llm, profile, row,
                                          threshold=threshold,
                                          max_auto_retries=retries,
                                          brief_enabled=brief_enabled,
                                          run_meta=run_meta)
        except AuthLLMError as e:
            # Every remaining job would fail this way too; one rejected call
            # is the whole diagnosis. Caught before the generic handler
            # below, which would otherwise count 191 identical "errors".
            print(f"[match] aborted - credentials rejected: {e}")
            break
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


# Both error states, not just the terminal one. `retryable_error` looks
# transient in `graph.py` -- every branch that writes it writes the next status
# immediately after -- but those are two separately committed transitions, so a
# batch killed in between (Ctrl-C, OOM, a dropped connection) leaves the row
# parked there. Nothing else sweeps it: `sweep_stale_matching` looks at
# `matching` and `sweep_stuck_new` at `new`. `ALLOWED_TRANSITIONS` has always
# allowed `retryable_error -> ready_for_match`; until this list had two entries
# nothing ever asked for it, and the job sat in the database unreachable.
_RESET_TO_READY = ("permanent_error", "retryable_error")


def cmd_retry(conn, profile) -> dict:
    stale = db.sweep_stale_matching(conn)
    orphans = db.sweep_stuck_new(conn, profile)
    reset = 0
    for status in _RESET_TO_READY:
        for row in db.get_versions_by_status(conn, status):
            db.set_status(conn, row["id"], "ready_for_match")
            conn.execute("UPDATE job_versions SET attempt_count=0 WHERE id=?",
                         (row["id"],))
            conn.commit()
            reset += 1
    return {"reset": reset, "stale_swept": stale,
            "orphans_prefiltered": orphans}


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="offerpilot",
        description="Human-in-the-loop job-search pipeline. Collects public "
                    "postings, filters them deterministically, scores the "
                    "survivors against your profile, and queues them for your "
                    "approval. Nothing is ever sent to an employer.")
    p.add_argument("command",
                   choices=["collect", "match", "status", "retry", "panel",
                            "eval", "demo"],
                   help="collect: pull postings from configured ATS boards. "
                        "match: score ready jobs with the LLM. "
                        "status: counts by pipeline status. "
                        "retry: reset errored jobs and sweep orphans. "
                        "panel: serve the local review panel. "
                        "eval: score the pipeline against blind labels. "
                        "demo: seed a throwaway database with synthetic data "
                        "and serve the panel; needs no config and no API key.")
    p.add_argument("--db", default="data/offerpilot.db",
                   help="SQLite path (default: %(default)s)")
    p.add_argument("--config", default="config.yaml",
                   help="YAML config (default: %(default)s)")
    p.add_argument("--profile", default="profile.yaml",
                   help="candidate profile YAML (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N jobs (collect, match)")
    args = p.parse_args(argv)

    # Intercepted before everything below, because the whole promise of demo
    # mode is that it runs on a clean checkout: no config.yaml, no
    # profile.yaml, no API key, and no database at `--db`. It brings its own
    # synthetic profile and seeds a temp database of its own.
    if args.command == "demo":
        from offerpilot.demo import run_demo
        run_demo(host="127.0.0.1", port=8000)
        return

    # Read config first: a missing config must fail before we create a DB.
    cfg = load_config(args.config, strict=(args.command in {"collect", "match"}))

    # Before `db.connect`, which would otherwise create the file this is
    # checking for. An eval over a database that does not exist reports zero
    # labels and all-None metrics, which looks like a finding rather than a
    # typo. `python run_eval.py` routes through here, so this is the one
    # implementation of that guard and the two entry points cannot drift.
    if args.command == "eval" and not os.path.exists(args.db):
        print(f"{args.db} not found. Run `offerpilot collect` and label some "
              f"jobs in the panel first.")
        raise SystemExit(1)

    if (args.command in REFUSES_EXAMPLE_PROFILE
            and not os.path.exists(args.profile)):
        print(f"{args.profile} not found - "
              f"{REFUSES_EXAMPLE_PROFILE[args.command]}. Create "
              f"{args.profile} (copy profile.example.yaml) first.")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = db.connect(args.db)
    db.init_schema(conn)

    if os.path.exists(args.profile):
        profile_path = args.profile
    else:
        profile_path = "profile.example.yaml"
        print(f"[warn] {args.profile} not found; using "
              f"{profile_path} instead.")
    profile = load_profile(profile_path)

    if args.command == "collect":
        # A run that crashed between upsert_job and the prefilter left rows
        # at 'new' that nothing else reads; heal them before collecting more.
        orphans = db.sweep_stuck_new(conn, profile)
        if orphans:
            print(f"[collect] re-prefiltered {orphans} orphaned job "
                  f"version(s) from an earlier run")
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
    elif args.command == "eval":
        # Imported here for the same reason as the panel below: the
        # pipeline subcommands should not pay to import the eval harness.
        from offerpilot.evaluate import run_eval as _run_eval
        eval_cfg = cfg.get("eval") or {}
        # Same rendering as `python run_eval.py`, which is the same call.
        # `profile_path` goes into the committed result file: which profile
        # the groundedness counts were computed against decides what they mean.
        print(json.dumps(
            _run_eval(conn, profile, profile_path=profile_path,
                      results_dir=eval_cfg.get("results_dir", "evals/results"),
                      precision_at=eval_cfg.get("precision_at", [5, 10])),
            indent=2))
    elif args.command == "panel":
        # Imported here so the other subcommands never pay for
        # FastAPI's import time.
        from offerpilot.panel import app as panel_app
        panel_cfg = cfg.get("panel") or {}
        port = int(panel_cfg.get("port", 8000))
        # The panel opens its own short-lived connection per request; holding
        # this one open would just be a second writer on the same WAL file.
        conn.close()
        try:
            # Before the banner: announcing an address the panel then refuses
            # to bind would read as a crash rather than as the refusal it is.
            host = panel_app.require_loopback(panel_cfg.get("host", "127.0.0.1"))
        except ValueError as e:
            print(e)
            raise SystemExit(1)
        print(f"review panel on {panel_app.panel_url(host, port)}  "
              f"(ctrl-c to stop)")
        panel_app.serve(args.db, profile, host=host, port=port)
