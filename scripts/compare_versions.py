"""Score pipeline v1 and v2 on the same blind-labeled database.

    set DEEPSEEK_API_KEY=...
    python scripts/compare_versions.py --db data/eval-2026-11.db \
        --config configs/poster-2026-11.yaml --out data/compare

Each version runs on its own copy under --out; the labeled database is only
read. Result files land in the config's eval.results_dir, tagged -v1 / -v2.
"""

import argparse
import json
import os
import sys

from offerpilot import compare
from offerpilot.config import config_hash, git_commit, load_config
from offerpilot.llm import LLMClient
from offerpilot.profile import load_profile
from offerpilot.store import db


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--profile", default="profile.yaml")
    p.add_argument("--out", default="data/compare")
    p.add_argument("--versions", nargs="+", default=["v1", "v2"])
    a = p.parse_args(argv)
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set")
    if not os.path.exists(a.profile):
        sys.exit(f"{a.profile} not found")
    conn = db.connect(a.db)
    if not db.get_labels(conn, label_source="blind_eval"):
        sys.exit(f"{a.db} has no blind_eval labels yet; import them first")
    conn.close()
    cfg = load_config(a.config, strict=True)
    profile = load_profile(a.profile)
    out = compare.run_comparison(
        a.db, cfg, profile, make_llm=lambda c: LLMClient(c, cfg["llm"], key),
        versions=a.versions, out_dir=a.out, profile_path=a.profile,
        run_meta={"git_commit": git_commit(), "config_hash": config_hash(cfg)})
    for v, r in out.items():
        c = r["classification"]
        print(f"{v}: precision={c['precision']} recall={c['recall']} "
              f"f1={c['f1']} (tp={c['tp']} fp={c['fp']} fn={c['fn']} "
              f"tn={c['tn']}) prefilter_fn={r['prefilter_false_negatives']} "
              f"-> {r['written_to']}")
    print(json.dumps({v: r["ranking"] for v, r in out.items()}, indent=2))


if __name__ == "__main__":
    main()
