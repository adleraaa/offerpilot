"""Spec-named entry point: python run_eval.py [--db ...] [--profile ...]

The spec names this file, so it exists; it delegates to `offerpilot eval`
rather than repeating the config/profile/database loading sequence, so the two
entry points cannot drift into producing different numbers from the same
database.
"""

import argparse
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

    # Every guard -- missing database, missing profile.yaml -- lives in the
    # eval branch of `offerpilot.cli`, so this shim cannot be the lenient way
    # in. Adding a check here would be a second implementation to keep in
    # sync, which is the drift this file exists to avoid.
    from offerpilot.cli import main as cli_main
    cli_main(["eval", "--db", args.db, "--config", args.config,
              "--profile", args.profile])


if __name__ == "__main__":
    sys.exit(main())
