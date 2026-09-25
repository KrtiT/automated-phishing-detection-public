"""Closed private failure fields and receipt-backed create-only persistence."""

import base64
import json
import os
import stat
from hashlib import sha256

from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._external_checkpoint_protocol import PROTOCOL, PROVENANCE_ORDER, SCIENTIFIC_ORDER

SIDECAR = "external-failure.json"
KINDS = {
    "cancelled",
    "keyboard_interrupt",
    "system_exit",
    "execution_failed",
    "interrupted",
}
_CHECKPOINT_KEYS = {
    "schema_version",
    "protocol_id",
    "reservation_sha256",
    "status",
    "confirmed_sha256",
    "pending_checkpoint_bytes",
}


class ExternalSourceFailureError(ValueError):
    """Symbolic private-retention failure with no diagnostic source content."""


def require(condition):
    if not condition:
        raise ExternalSourceFailureError("invalid_external_source_failure")


def base64_value(value):
    require(type(value) is str)
    require(
        base64.b64encode(base64.b64decode(value, validate=True)).decode("ascii")
        == value
    )


def checkpoint(value, reservation):
    if value is None:
        return
    require(type(value) is dict and set(value) == _CHECKPOINT_KEYS)
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(
        value["protocol_id"] == PROTOCOL and value["reservation_sha256"] == reservation
    )
    require(value["status"] in {"pending", "failed", "complete"})
    confirmed, pending = value["confirmed_sha256"], value["pending_checkpoint_bytes"]
    require(type(confirmed) is dict and type(pending) is dict)
    order = PROVENANCE_ORDER + SCIENTIFIC_ORDER
    require(set(confirmed) == set(order[: len(confirmed)]))
    require(not 0 < len(confirmed) < len(PROVENANCE_ORDER))
    _pending(value, confirmed, pending, order)
    for digest in confirmed.values():
        require(type(digest) is str and receipt._SHA256.fullmatch(digest) is not None)
    for content in pending.values():
        base64_value(content)
    if value["status"] == "complete":
        require(len(confirmed) == len(order) and not pending)


def _pending(value, confirmed, pending, order):
    expected = (
        set(PROVENANCE_ORDER)
        if not confirmed
        else set(order[len(confirmed) : len(confirmed) + 1])
    )
    require(not pending or (value["status"] == "failed" and set(pending) == expected))
    require(value["status"] != "pending" or len(confirmed) < len(order))


def _private_file(directory, name):
    metadata = receipt._entry(directory, name)
    require(metadata is not None and stat.S_ISREG(metadata.st_mode))
    require(metadata.st_nlink == 1 and stat.S_IMODE(metadata.st_mode) == 0o600)
    return metadata


def _guard(directory, *, installed):
    directory.check()
    require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
    names = set(os.listdir(directory.descriptor))
    required = {"reservation.json", SIDECAR} if installed else {"reservation.json"}
    require(required <= names <= required | {"checkpoints"})
    reservation = _private_file(directory, "reservation.json")
    checkpoints = receipt._entry(directory, "checkpoints")
    if checkpoints is not None:
        require(stat.S_ISDIR(checkpoints.st_mode))
        require(stat.S_IMODE(checkpoints.st_mode) == 0o700)
    if installed:
        _private_file(directory, SIDECAR)
    return receipt._identity(reservation), receipt._identity(
        checkpoints
    ) if checkpoints else None


def _authenticate(attempt, directory, record):
    receipt._authenticate(attempt, directory)
    content = receipt._read_reservation(directory)
    require(sha256(content).hexdigest() == attempt.reservation_sha256)
    require(
        receipt._json_bytes(json.loads(content)["identity"], "identity")
        == receipt._json_bytes(record["execution"], "execution")
    )


def _install(directory, content):
    with receipt._staging_directory(directory, SIDECAR) as (staging, unused):
        receipt._write_file(staging, "record.json", content, 0o600)
        receipt._sync_directory(staging)
        identity = receipt._identity(_private_file(staging, "record.json"))
        receipt._publish(staging, "record.json", directory, SIDECAR)
        receipt._sync_directory(directory)
        return identity


def _state(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _verify_installed(directory, content, identity):
    before = _private_file(directory, SIDECAR)
    require(receipt._identity(before) == identity)
    descriptor = os.open(
        SIDECAR,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=directory.descriptor,
    )
    with preserve_cleanup(lambda: os.close(descriptor)):
        require(_state(os.fstat(descriptor)) == _state(before))
        with CleanupStack() as cleanup:
            stream = cleanup.enter_context(os.fdopen(descriptor, "rb", closefd=False))
            require(stream.read() == content)
        require(_state(os.fstat(descriptor)) == _state(before))
    after = receipt._entry(directory, SIDECAR)
    require(after is not None and _state(after) == _state(before))
    directory.check()


def retain(attempt, content, record):
    require(type(attempt) is receipt.Attempt)
    require(record["reservation_sha256"] == attempt.reservation_sha256)
    with receipt._attempt_directory(attempt) as directory:
        identities = _guard(directory, installed=False)
        _authenticate(attempt, directory, record)
        receipt._require_absent(directory, SIDECAR)
        identity = _install(directory, content)
        _authenticate(attempt, directory, record)
        require(_guard(directory, installed=True) == identities)
        _verify_installed(directory, content, identity)
