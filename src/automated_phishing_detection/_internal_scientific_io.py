"""Descriptor-checked create-only scientific checkpoint installation."""

import json
import os
import stat

from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._internal_scientific_protocol import SCIENTIFIC_DIRECTORY, require


def guard(directory: receipt._Directory) -> None:
    require(
        all(
            receipt._entry(directory, name) is None
            for name in (
                "finalize.claim",
                "evidence",
                "outcome.json",
                "failure-progress.json",
            )
        )
    )


def validate_identity(
    attempt: receipt.Attempt, directory: receipt._Directory, identity: dict
) -> None:
    reservation = json.loads(receipt._read_reservation(directory))
    require(canonical_bytes(reservation["identity"]) == canonical_bytes(identity))
    require(reservation["directory"] == str(attempt.directory))


def start(
    attempt: receipt.Attempt, identity: dict, contents: dict[str, bytes]
) -> tuple[tuple[int, int], tuple[int, int]]:
    with receipt._attempt_directory(attempt) as directory:
        guard(directory)
        validate_identity(attempt, directory, identity)
        receipt._require_absent(directory, SCIENTIFIC_DIRECTORY)
        with receipt._staging_directory(directory, SCIENTIFIC_DIRECTORY) as (
            staging,
            name,
        ):
            for filename, content in contents.items():
                receipt._write_file(staging, filename, content, 0o600)
            receipt._sync_directory(staging)
            receipt._publish(directory, name, directory, SCIENTIFIC_DIRECTORY)
            receipt._sync_directory(directory)
            return receipt._identity(os.fstat(directory.descriptor)), receipt._identity(
                os.fstat(staging.descriptor)
            )


def append(
    attempt: receipt.Attempt,
    identities: tuple,
    confirmed: dict[str, str],
    name: str,
    content: bytes,
) -> None:
    with receipt._attempt_directory(attempt) as directory:
        guard(directory)
        require(receipt._identity(os.fstat(directory.descriptor)) == identities[0])
        with receipt._directory(directory.path / SCIENTIFIC_DIRECTORY) as checkpoints:
            require(
                receipt._identity(os.fstat(checkpoints.descriptor)) == identities[1]
            )
            require(stat.S_IMODE(os.fstat(checkpoints.descriptor).st_mode) == 0o700)
            require(set(os.listdir(checkpoints.descriptor)) == set(confirmed))
            receipt._require_absent(checkpoints, name)
            receipt._install_record(checkpoints, name, content)
