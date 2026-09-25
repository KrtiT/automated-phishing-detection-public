"""Fixed operational child options; no deadline or workload override."""

import argparse
import asyncio
import sys
from dataclasses import fields
from pathlib import Path

from ._operational_cell_protocol import COMMON_ARGUMENTS, SERVICE_ARGUMENTS
from .bound_models import ArtifactPaths
from .internal_failure import failure_exit_code, failure_kind


def parser(role):
    command = argparse.ArgumentParser(
        description=f"Run one bound operational {role}; access closed.",
        allow_abbrev=False,
    )
    names = SERVICE_ARGUMENTS if role == "service" else COMMON_ARGUMENTS
    for name in names:
        kind = str if name.startswith("--expected-") else Path
        command.add_argument(name, type=kind, required=True)
    return command


def _keywords(arguments, role):
    result = {
        name: getattr(arguments, name)
        for name in (
            "expected_revision",
            "expected_contract_sha256",
            "expected_operational_profile_sha256",
            "expected_binding_sha256",
        )
    }
    result.update(
        accepted_inputs_directory=arguments.accepted_inputs_dir,
        cell_input_directory=arguments.cell_input_dir,
    )
    if role == "service":
        result["artifacts"] = ArtifactPaths(
            *(getattr(arguments, member.name) for member in fields(ArtifactPaths))
        )
    return result


def main(role, runner, argv=None):
    arguments = parser(role).parse_args(argv)
    try:
        asyncio.run(runner(arguments.repo_root, **_keywords(arguments, role)))
    except BaseException as error:
        print(f"Operational child stopped: {failure_kind(error)}", file=sys.stderr)
        return failure_exit_code(error)
    print("Operational child completed; parent verification required.")
    return 0
