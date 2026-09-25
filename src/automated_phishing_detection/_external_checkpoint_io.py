"""Private create-only external checkpoints bound to their reservation and inode."""

import json
import os
import stat
from contextlib import contextmanager

from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._external_checkpoint_protocol import DIRECTORY, require


@contextmanager
def _preserved(manager):
    original = None
    try:
        with manager as value:
            try:
                yield value
            except BaseException as error:
                original = error
                raise
    except BaseException:
        if original is not None and not isinstance(original, Exception):
            raise original from None
        raise


def _guard(directory, *, started):
    require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
    expected = {"reservation.json", DIRECTORY} if started else {"reservation.json"}
    require(set(os.listdir(directory.descriptor)) == expected)


@contextmanager
def _authenticated(attempt):
    require(type(attempt) is receipt.Attempt)
    with _preserved(receipt._directory(attempt.directory)) as directory:
        receipt._authenticate(attempt, directory)
        yield directory


def start(attempt, identity, contents):
    with _authenticated(attempt) as directory:
        _guard(directory, started=False)
        reservation = json.loads(receipt._read_reservation(directory))
        require(canonical_bytes(reservation["identity"]) == canonical_bytes(identity))
        with _preserved(receipt._staging_directory(directory, DIRECTORY)) as (
            staging,
            name,
        ):
            for filename, content in contents.items():
                receipt._write_file(staging, filename, content, 0o600)
            receipt._sync_directory(staging)
            receipt._publish(directory, name, directory, DIRECTORY)
            receipt._sync_directory(directory)
            return (
                receipt._identity(os.fstat(directory.descriptor)),
                receipt._identity(os.fstat(staging.descriptor)),
            )


@contextmanager
def _checked(attempt, identities, confirmed):
    with _authenticated(attempt) as directory:
        _guard(directory, started=True)
        require(receipt._identity(os.fstat(directory.descriptor)) == identities[0])
        with _preserved(receipt._directory(directory.path / DIRECTORY)) as checkpoints:
            require(
                receipt._identity(os.fstat(checkpoints.descriptor)) == identities[1]
            )
            require(stat.S_IMODE(os.fstat(checkpoints.descriptor).st_mode) == 0o700)
            require(set(os.listdir(checkpoints.descriptor)) == set(confirmed))
            yield checkpoints


def append(attempt, identities, confirmed, name, content):
    with _checked(attempt, identities, confirmed) as directory:
        receipt._require_absent(directory, name)
        _install_record(directory, name, content)


def _install_record(directory, name, content):
    with _preserved(receipt._staging_directory(directory, name)) as (staging, unused):
        receipt._write_file(staging, "record.json", content, 0o600)
        receipt._sync_directory(staging)
        receipt._publish(staging, "record.json", directory, name)
        receipt._sync_directory(directory)


def validate(attempt, identities, confirmed):
    with _checked(attempt, identities, confirmed):
        pass
