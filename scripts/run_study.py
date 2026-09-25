"""Run one fresh same-parent study; all current protected-access gates are closed."""

import argparse
import asyncio
import json
import sys
from dataclasses import fields
from pathlib import Path

from automated_phishing_detection._study_cli_protocol import (
    ARGUMENTS,
    PATH_ARGUMENTS,
)


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for option in ARGUMENTS:
        command.add_argument(
            option, type=Path if option[2:] in PATH_ARGUMENTS else str, required=True
        )
    return command


def _artifacts(arguments):
    from automated_phishing_detection.bound_drift import DriftArtifactPaths
    from automated_phishing_detection.bound_models import ArtifactPaths
    from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths

    return tuple(
        group(*(getattr(arguments, member.name) for member in fields(group)))
        for group in (ArtifactPaths, SecondaryArtifactPaths, DriftArtifactPaths)
    )


def _preparation(arguments):
    from automated_phishing_detection._study_preparation_records import (
        StudyPreparationPaths,
    )

    return StudyPreparationPaths(
        arguments.source_csv,
        arguments.suffix_rules,
        arguments.archive,
        arguments.preparation_attempt,
    )


def _source_paths(arguments):
    from automated_phishing_detection._prepared_external_records import (
        PreparedExternalRunPaths,
    )
    from automated_phishing_detection._prepared_internal_records import (
        PreparedInternalRunPaths,
    )

    primary, secondary, drift = _artifacts(arguments)
    internal = PreparedInternalRunPaths(
        arguments.preparation_attempt,
        primary,
        secondary,
        arguments.internal_attempt,
        arguments.internal_public_summary,
    )
    external = PreparedExternalRunPaths(
        arguments.preparation_attempt,
        primary,
        secondary,
        drift,
        arguments.external_attempt,
        arguments.external_public_summary,
    )
    return _preparation(arguments), internal, external


def _paths(arguments):
    from automated_phishing_detection.study_runner import StudyRunPaths

    return StudyRunPaths(
        *_source_paths(arguments),
        arguments.attempt,
        arguments.public_summary,
        arguments.accepted_inputs_dir,
        arguments.cells_dir,
    )


async def _run(arguments):
    from automated_phishing_detection.study_runner import run_study

    return await run_study(
        arguments.repo_root,
        expected_revision=arguments.expected_revision,
        expected_contract_sha256=arguments.expected_contract_sha256,
        expected_operational_profile_sha256=arguments.expected_operational_profile_sha256,
        paths=_paths(arguments),
    )


def _message(result):
    status = json.loads(result.snapshot.payload("public-summary.json"))["status"]
    return {
        "whole_study_hold": "Study held: insufficient population capacity.",
        "study_evidence_published": "Study evidence published.",
    }[status]


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        message = _message(asyncio.run(_run(arguments)))
    except BaseException as error:
        from automated_phishing_detection.internal_failure import (
            failure_exit_code,
            failure_kind,
        )

        print(f"Study execution stopped: {failure_kind(error)}", file=sys.stderr)
        return failure_exit_code(error)
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
