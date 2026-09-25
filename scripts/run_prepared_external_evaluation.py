"""Publish one external worker's evidence from retained preparation; access closed."""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from automated_phishing_detection._external_source_records import (
    PreparedExternalRunPaths,
)
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths
from automated_phishing_detection.external_source_runner import (
    run_prepared_external_evaluation,
)
from automated_phishing_detection.internal_failure import (
    failure_exit_code,
    failure_kind,
)

_ARTIFACT_GROUPS = (ArtifactPaths, SecondaryArtifactPaths, DriftArtifactPaths)


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument("--repo-root", type=Path, required=True)
    for name in (
        "expected-revision",
        "expected-contract-sha256",
        "expected-handoff-sha256",
        "expected-preparation-reservation-sha256",
        "expected-preparation-completion-sha256",
    ):
        command.add_argument(f"--{name}", required=True)
    names = (
        "preparation",
        *(
            member.name.replace("_", "-")
            for group in _ARTIFACT_GROUPS
            for member in fields(group)
        ),
        "attempt",
        "public-summary",
        "internal-transport",
    )
    for name in names:
        command.add_argument(f"--{name}", type=Path, required=True)
    return command


def _paths(arguments):
    groups = (
        group(*(getattr(arguments, member.name) for member in fields(group)))
        for group in _ARTIFACT_GROUPS
    )
    return PreparedExternalRunPaths(
        arguments.preparation, *groups, arguments.attempt, arguments.public_summary
    )


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        run_prepared_external_evaluation(
            arguments.repo_root,
            expected_revision=arguments.expected_revision,
            expected_contract_sha256=arguments.expected_contract_sha256,
            paths=_paths(arguments),
            internal_transport=arguments.internal_transport,
            expected_handoff_sha256=arguments.expected_handoff_sha256,
            expected_preparation_reservation_sha256=arguments.expected_preparation_reservation_sha256,
            expected_preparation_completion_sha256=arguments.expected_preparation_completion_sha256,
        )
    except BaseException as error:
        print(f"External execution stopped: {failure_kind(error)}", file=sys.stderr)
        return failure_exit_code(error)
    print("External evidence published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
