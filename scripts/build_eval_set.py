"""Sample the second blind eval set and write one labeling sheet per labeler.

    python scripts/build_eval_set.py --source data/poster-2026-11.db \
        --out data/eval-2026-11.db --sheets labeling/ --labelers A B

Writes the eval database (sampled versions only, status `new`) and
`labeling/label_<name>.xlsx` for each labeler, each in its own shuffled order.
"""

import argparse
import os
import sys

from offerpilot import evalset
from offerpilot.profile import load_profile
from offerpilot.store import db


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sheets", required=True)
    p.add_argument("--labelers", nargs="+", default=["A", "B"])
    p.add_argument("--profile", default="profile.yaml")
    p.add_argument("--n-student", type=int, default=60)
    p.add_argument("--n-other", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261112)
    a = p.parse_args(argv)
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; refusing to overwrite a set that may "
                 f"already carry labels")
    src = db.connect(a.source)
    ids = evalset.sample_version_ids(src, n_student=a.n_student,
                                     n_other=a.n_other, seed=a.seed)
    out = evalset.export_eval_db(src, a.out, ids)
    profile = load_profile(a.profile)
    os.makedirs(a.sheets, exist_ok=True)
    for i, name in enumerate(a.labelers):
        path = os.path.join(a.sheets, f"label_{name}.xlsx")
        n = evalset.write_label_sheet(out, path, profile, seed=a.seed + i + 1)
        print(f"{path}: {n} jobs")
    student = sum(evalset.is_student_level(r["title"]) for r in out.execute(
        "SELECT title FROM job_versions"))
    print(f"{a.out}: {len(ids)} versions ({student} student-level, "
          f"{len(ids) - student} other), seed {a.seed}")


if __name__ == "__main__":
    main()
