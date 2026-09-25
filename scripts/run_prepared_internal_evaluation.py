"""Consume retained study preparation only after both access gates are frozen."""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths
from automated_phishing_detection.internal_failure import (
    failure_exit_code,
    failure_kind,
)
from automated_phishing_detection.prepared_internal_runner import (
    PreparedInternalRunPaths,
    run_prepared_internal_evaluation,
    run_prepared_internal_process_with_evidence,
)


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for name in (
        "expected-revision",
        "expected-contract-sha256",
        "expected-preparation-reservation-sha256",
        "expected-preparation-completion-sha256",
    ):
        command.add_argument(f"--{name}", required=True)
    for name in (
        "preparation",
        "attempt",
        "public-summary",
        *(
            member.name.replace("_", "-")
            for group in (ArtifactPaths, SecondaryArtifactPaths)
            for member in fields(group)
        ),
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    return command


def _paths(args):
    artifacts = tuple(
        group(*(getattr(args, member.name) for member in fields(group)))
        for group in (ArtifactPaths, SecondaryArtifactPaths)
    )
    return PreparedInternalRunPaths(
        args.preparation, *artifacts, args.attempt, args.public_summary
    )


def main(argv=None):
    args = parser().parse_args(argv)
    run = (
        run_prepared_internal_evaluation
        if args.worker
        else run_prepared_internal_process_with_evidence
    )
    try:
        run(
            args.repo_root,
            expected_revision=args.expected_revision,
            expected_contract_sha256=args.expected_contract_sha256,
            paths=_paths(args),
            expected_preparation_reservation_sha256=args.expected_preparation_reservation_sha256,
            expected_preparation_completion_sha256=args.expected_preparation_completion_sha256,
        )
    except BaseException as error:
        print(f"Internal execution stopped: {failure_kind(error)}", file=sys.stderr)
        return failure_exit_code(error)
    print(
        "Internal evidence published."
        if args.worker
        else "Internal completion verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
