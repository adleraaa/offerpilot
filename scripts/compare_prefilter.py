"""Count what the v1 and v2 prefilters each do with a collected database.

    python scripts/compare_prefilter.py --db data/poster-2026-11.db \
        [--profile profile.yaml] [--out evals/results/prefilter-split.json]

No API key and no writes: the database is opened read-only and the rules run
in memory on the stored posting text. Prints the split and, with --out, saves
it as JSON next to the eval result files.
"""

import argparse
import datetime
import json
import os
import sqlite3
import sys

from offerpilot import compare
from offerpilot.config import git_commit
from offerpilot.profile import load_profile


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--profile", default="profile.yaml")
    p.add_argument("--out")
    a = p.parse_args(argv)
    for path in (a.db, a.profile):
        if not os.path.exists(path):
            sys.exit(f"{path} not found")
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    split = compare.prefilter_split(conn, load_profile(a.profile))
    conn.close()

    s = split["student_level"]
    kept = split["kept_by_v2_only"]
    print(f"{split['total']} job versions, {s['total']} with a student-level "
          f"title")
    for v in ("v1", "v2"):
        print(f"{v}: {split[v]['ready_for_match']} reach the model "
              f"({s[v]['ready_for_match']} student-level), "
              f"{split[v]['filtered_out']} removed "
              f"({s[v]['filtered_out']} student-level)")
    print(f"kept by v2 only: {kept['total']} ({kept['student_level']} "
          f"student-level), by rule: {kept['by_rule']}")
    print(f"removed by v2, by hard rule: {split['removed_by_v2']}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        record = {"generated_at": datetime.datetime.now(
                      datetime.timezone.utc).isoformat(timespec="seconds"),
                  "git_commit": git_commit(),
                  "database": os.path.basename(a.db), **split}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
