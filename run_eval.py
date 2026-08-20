"""Spec-named entry point: python run_eval.py [--db ...] [--profile ...]

The spec names this file, so it exists; it delegates to `offerpilot eval`
rather than repeating the config/profile/database loading sequence, so the two
entry points cannot drift into producing different numbers from the same
database.
"""

import argparse
import os
import sys


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run_eval",
        description="Score the pipeline against blind labels and write "
                    "evals/results/eval-<timestamp>.json. Equivalent to "
                    "`python -m offerpilot eval`.")
    p.add_argument("--db", default="data/offerpilot.db")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--profile", default="profile.yaml")
    args = p.parse_args(argv)

    # An eval over a database that does not exist reports zero labels and
    # all-None metrics, which looks like a finding rather than a typo.
    if not os.path.exists(args.db):
        raise SystemExit(f"{args.db} not found. Run `offerpilot collect` and "
                         f"label some jobs in the panel first.")

    from offerpilot.cli import main as cli_main
    cli_main(["eval", "--db", args.db, "--config", args.config,
              "--profile", args.profile])


if __name__ == "__main__":
    sys.exit(main())
