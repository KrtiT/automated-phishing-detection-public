"""Run the authenticated development-only family and verify the worker's evidence."""

import argparse
import sys
from pathlib import Path

from automated_phishing_detection.development_runner import (
    DevelopmentRunPaths,
    run_development,
    run_development_process,
)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--expected-revision", required=True)
    command.add_argument("--expected-profile-sha256", required=True)
    command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for name in (
        "train",
        "validation",
        "suffix-rules",
        "logistic-l1",
        "gmm",
        "attempt",
        "public-summary",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    paths = DevelopmentRunPaths(
        args.train,
        args.validation,
        args.suffix_rules,
        args.logistic_l1,
        args.gmm,
        args.attempt,
        args.public_summary,
    )
    try:
        run = run_development if args.worker else run_development_process
        run(
            args.repo_root,
            expected_revision=args.expected_revision,
            expected_profile_sha256=args.expected_profile_sha256,
            paths=paths,
        )
    except Exception as exc:
        print(f"Development execution stopped: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        "Development evidence published."
        if args.worker
        else "Development completion verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
