"""Publish one worker's external evidence behind the closed access gate.

Publication is not observed acceptance; only the same observing parent can
accept completion. No override permits protected-data access.
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from automated_phishing_detection._external_source_records import ExternalRunPaths
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths
from automated_phishing_detection.external_source_runner import run_external_evaluation
from automated_phishing_detection.internal_failure import (
    failure_exit_code,
    failure_kind,
)

_ARTIFACT_GROUPS = (ArtifactPaths, SecondaryArtifactPaths, DriftArtifactPaths)


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--expected-revision", required=True)
    command.add_argument("--expected-contract-sha256", required=True)
    path_names = (
        "archive",
        "suffix-rules",
        *(
            field.name.replace("_", "-")
            for group in _ARTIFACT_GROUPS
            for field in fields(group)
        ),
        "attempt",
        "public-summary",
        "internal-transport",
    )
    for name in path_names:
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--expected-handoff-sha256", required=True)
    return command


def _paths(arguments):
    groups = (
        group(*(getattr(arguments, field.name) for field in fields(group)))
        for group in _ARTIFACT_GROUPS
    )
    return ExternalRunPaths(
        arguments.archive,
        arguments.suffix_rules,
        *groups,
        arguments.attempt,
        arguments.public_summary,
    )


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        run_external_evaluation(
            arguments.repo_root,
            expected_revision=arguments.expected_revision,
            expected_contract_sha256=arguments.expected_contract_sha256,
            paths=_paths(arguments),
            internal_transport=arguments.internal_transport,
            expected_handoff_sha256=arguments.expected_handoff_sha256,
        )
    except BaseException as error:
        print(f"External execution stopped: {failure_kind(error)}", file=sys.stderr)
        return failure_exit_code(error)
    print("External evidence published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
