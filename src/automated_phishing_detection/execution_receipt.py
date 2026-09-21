"""Durable, single-attempt evidence publication without recovery or resumption.

Reserve successfully before protected access. An installed reservation is never
removed, and an installed finalization claim is never retried. Completion writes
private evidence and its outcome before the public marker. Acceptance requires a
successful producer exit AND independently verified artifacts/marker: a marker
can exist after a failed post-rename sync or a crash, without proving durability.

Callers enforce identity bindings, public aggregate redaction, and the absence of
URLs or private values from public summaries. These primitives do not perform
research execution or authenticate those caller-supplied scientific bindings.
"""

from __future__ import annotations

import ctypes
import errno
import json
import math
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from . import transformer_pipeline

_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExecutionReceiptError(ValueError):
    """An attempt or publication payload violates its immutable boundary."""


@dataclass(frozen=True)
class Attempt:
    directory: Path
    reservation_sha256: str


def _json_value(value: object) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ExecutionReceiptError("payload must contain only strict finite JSON values")


def _json_bytes(value: dict, name: str) -> bytes:
    if type(value) is not dict or not value:
        raise ExecutionReceiptError(f"{name} must be a nonempty dictionary")
    try:
        _json_value(value)
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ExecutionReceiptError(f"{name} must be strict finite UTF-8 JSON") from exc


def _absolute_path(path: Path) -> Path:
    if not isinstance(path, Path) or ".." in path.parts:
        raise ExecutionReceiptError("use a Path without parent traversal")
    return path.absolute()


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _open_directory(path: Path) -> int:
    """Walk every component without following links, retaining only the final fd."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ExecutionReceiptError(
            "directory-relative no-follow access is unavailable"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ExecutionReceiptError(
            "directory and parents must exist without aliases"
        ) from exc


@dataclass(frozen=True)
class _Directory:
    path: Path
    descriptor: int

    def check(self) -> None:
        current = _open_directory(self.path)
        try:
            if _identity(os.fstat(current)) != _identity(os.fstat(self.descriptor)):
                raise ExecutionReceiptError(
                    "directory identity changed during publication"
                )
        finally:
            os.close(current)


@contextmanager
def _directory(path: Path):
    absolute = _absolute_path(path)
    descriptor = _open_directory(absolute)
    try:
        yield _Directory(absolute, descriptor)
    finally:
        os.close(descriptor)


def _entry(directory: _Directory, name: str):
    try:
        return os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _require_absent(directory: _Directory, name: str) -> None:
    directory.check()
    if not name or _entry(directory, name) is not None:
        raise ExecutionReceiptError("destination already exists; no overwrite or retry")


def _sync_directory(directory: _Directory) -> None:
    directory.check()
    transformer_pipeline._sync_descriptor(directory.descriptor)
    directory.check()


def _write_file(directory: _Directory, name: str, content: bytes, mode: int) -> None:
    directory.check()
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
        dir_fd=directory.descriptor,
    )
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(content)
        stream.flush()
        transformer_pipeline._sync_descriptor(stream.fileno())
    directory.check()


def _rename_noreplace(
    source: _Directory, source_name: str, destination: _Directory, destination_name: str
) -> None:
    """The existing no-replace rename flags, bound to pinned directory fds."""
    if sys.platform == "darwin":
        function = ctypes.CDLL(None, use_errno=True).renameatx_np
        flag = 4
    elif sys.platform.startswith("linux"):
        try:
            function = ctypes.CDLL(None, use_errno=True).renameat2
        except AttributeError as exc:
            raise ExecutionReceiptError(
                "directory-relative no-replace rename is unavailable"
            ) from exc
        flag = 1
    else:
        raise ExecutionReceiptError(
            "directory-relative no-replace rename is unavailable"
        )
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    if (
        function(
            source.descriptor,
            os.fsencode(source_name),
            destination.descriptor,
            os.fsencode(destination_name),
            flag,
        )
        != 0
    ):
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise ExecutionReceiptError(
                "destination already exists; no overwrite or retry"
            )
        raise OSError(error, os.strerror(error))


def _publish(
    source: _Directory, source_name: str, destination: _Directory, destination_name: str
) -> None:
    source.check()
    destination.check()
    before = _entry(source, source_name)
    if before is None or not (
        stat.S_ISREG(before.st_mode) or stat.S_ISDIR(before.st_mode)
    ):
        raise ExecutionReceiptError("publication source must not be an alias")
    _rename_noreplace(source, source_name, destination, destination_name)
    source.check()
    destination.check()
    after = _entry(destination, destination_name)
    if after is None or _identity(after) != _identity(before):
        raise ExecutionReceiptError("installed artifact identity changed")


@contextmanager
def _staging_directory(parent: _Directory, prefix: str):
    parent.check()
    name = f".{prefix}.tmp-{secrets.token_hex(16)}"
    os.mkdir(name, 0o700, dir_fd=parent.descriptor)
    before = _entry(parent, name)
    descriptor = os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent.descriptor
    )
    staging = _Directory(parent.path / name, descriptor)
    try:
        if before is None or _identity(before) != _identity(os.fstat(descriptor)):
            raise ExecutionReceiptError("staging directory identity changed")
        parent.check()
        staging.check()
        yield staging, name
    finally:
        # Never enumerate/clean an fd whose directory has already been installed.
        current = _entry(parent, name)
        if current is not None and _identity(current) == _identity(
            os.fstat(descriptor)
        ):
            try:
                for child in os.listdir(descriptor):
                    os.unlink(child, dir_fd=descriptor)
                os.rmdir(name, dir_fd=parent.descriptor)
            except OSError:
                pass
        os.close(descriptor)


def _install_record(directory: _Directory, name: str, content: bytes) -> None:
    with _staging_directory(directory, name) as (staging, _):
        _write_file(staging, "record.json", content, 0o600)
        _sync_directory(staging)
        _publish(staging, "record.json", directory, name)
        _sync_directory(directory)


def reserve_attempt(directory: Path, *, identity: dict) -> Attempt:
    """Install a new private reservation before any future protected access.

    A failure after installation leaves the reservation permanently occupied.
    No attempt handle is returned unless installation and parent sync succeeded.
    """
    identity_bytes = _json_bytes(identity, "identity")
    destination = _absolute_path(directory)
    reservation = _json_bytes(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(destination),
            "identity": json.loads(identity_bytes),
        },
        "reservation",
    )
    with _directory(destination.parent) as parent:
        _require_absent(parent, destination.name)
        with _staging_directory(parent, destination.name) as (staging, name):
            _write_file(staging, "reservation.json", reservation, 0o600)
            _sync_directory(staging)
            _publish(parent, name, parent, destination.name)
            _sync_directory(parent)
    return Attempt(destination, sha256(reservation).hexdigest())


def _read_reservation(directory: _Directory) -> bytes:
    directory.check()
    before_path = _entry(directory, "reservation.json")
    descriptor = os.open(
        "reservation.json",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=directory.descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if (
            before_path is None
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _identity(before) != _identity(before_path)
        ):
            raise ExecutionReceiptError(
                "reservation must be a regular file without aliases"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = _entry(directory, "reservation.json")
    before_state = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    )
    after_state = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    )
    if (
        before_state != after_state
        or len(content) != before.st_size
        or after_path is None
        or _identity(after_path) != _identity(before)
    ):
        raise ExecutionReceiptError("reservation changed while being read")
    directory.check()
    return content


def _authenticate(attempt: Attempt, directory: _Directory) -> None:
    if (
        type(attempt) is not Attempt
        or type(attempt.reservation_sha256) is not str
        or _SHA256.fullmatch(attempt.reservation_sha256) is None
    ):
        raise ExecutionReceiptError("use an authenticated Attempt")
    try:
        content = _read_reservation(directory)
        if sha256(content).hexdigest() != attempt.reservation_sha256:
            raise ExecutionReceiptError("reservation hash does not match Attempt")
        record = json.loads(content)
        if (
            type(record) is not dict
            or set(record) != {"schema_version", "status", "directory", "identity"}
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or record["status"] != "reserved"
            or record["directory"] != str(directory.path)
            or content != _json_bytes(record, "reservation")
        ):
            raise ExecutionReceiptError(
                "reservation schema or directory binding differs"
            )
        _json_bytes(record["identity"], "identity")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ExecutionReceiptError(
            "reservation cannot be safely authenticated"
        ) from exc


@contextmanager
def _attempt_directory(attempt: Attempt):
    if type(attempt) is not Attempt:
        raise ExecutionReceiptError("use an authenticated Attempt")
    with _directory(attempt.directory) as directory:
        _authenticate(attempt, directory)
        yield directory


def _claim(attempt: Attempt, directory: _Directory, operation: str) -> None:
    _authenticate(attempt, directory)
    if any(
        _entry(directory, name) is not None
        for name in ("finalize.claim", "evidence", "outcome.json")
    ):
        raise ExecutionReceiptError(
            "attempt is already finalized or incomplete; no retry"
        )
    _install_record(
        directory,
        "finalize.claim",
        _json_bytes(
            {
                "schema_version": 1,
                "reservation_sha256": attempt.reservation_sha256,
                "operation": operation,
            },
            "claim",
        ),
    )


def record_failure(attempt: Attempt, *, stage: str, error_type: str) -> Path:
    """Consume finalization and record symbolic failure fields, never exception text."""
    for name, value in (("stage", stage), ("error_type", error_type)):
        if type(value) is not str or _SYMBOL.fullmatch(value) is None:
            raise ExecutionReceiptError(f"{name} must be a simple symbolic identifier")
    with _attempt_directory(attempt) as directory:
        content = _json_bytes(
            {
                "schema_version": 1,
                "status": "failed",
                "reservation_sha256": attempt.reservation_sha256,
                "stage": stage,
                "error_type": error_type,
            },
            "failure",
        )
        _claim(attempt, directory, "failure")
        _install_record(directory, "outcome.json", content)
        return directory.path / "outcome.json"


def _private_payloads(outputs: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(outputs, Mapping) or not outputs:
        raise ExecutionReceiptError("private_outputs must be a nonempty mapping")
    snapshot = dict(outputs)
    for name, content in snapshot.items():
        if type(name) is not str or _FILENAME.fullmatch(name) is None:
            raise ExecutionReceiptError("private output names must be simple filenames")
        if type(content) is not bytes:
            raise ExecutionReceiptError("private outputs must contain exact bytes")
    return snapshot


def publish_completion(
    attempt: Attempt,
    *,
    private_outputs: Mapping[str, bytes],
    public_summary: dict,
    public_path: Path,
) -> Path:
    """Publish private evidence, a prepared outcome, then the public marker last.

    Validate payloads before claiming. Every failure after the claim consumes the
    attempt; installed artifacts remain in place, even if publication or fsync
    raises after a rename. This API never creates a replacement failure outcome.
    The public payload is exactly the caller's aggregate JSON, without automatic
    insertion of private paths, raw output bytes, or other private metadata.
    """
    outputs = _private_payloads(private_outputs)
    summary = _json_bytes(public_summary, "public_summary")
    destination = _absolute_path(public_path)
    with (
        _attempt_directory(attempt) as directory,
        _directory(destination.parent) as public_parent,
    ):
        _require_absent(public_parent, destination.name)
        if destination.is_relative_to(directory.path):
            raise ExecutionReceiptError(
                "public marker must be outside the attempt directory"
            )
        outcome = _json_bytes(
            {
                "schema_version": 1,
                "status": "completion_prepared",
                "reservation_sha256": attempt.reservation_sha256,
                "public_summary_sha256": sha256(summary).hexdigest(),
                "private_sha256": {
                    name: sha256(content).hexdigest()
                    for name, content in outputs.items()
                },
            },
            "outcome",
        )
        _claim(attempt, directory, "completion")
        temporary_name = f".{destination.name}.tmp-{secrets.token_hex(16)}"
        try:
            with _staging_directory(directory, "evidence") as (staging, staging_name):
                for name, content in sorted(outputs.items()):
                    _write_file(staging, name, content, 0o600)
                _sync_directory(staging)
                _write_file(public_parent, temporary_name, summary, 0o644)
                _publish(directory, staging_name, directory, "evidence")
                _sync_directory(directory)
                _install_record(directory, "outcome.json", outcome)
                _authenticate(attempt, directory)
                _publish(public_parent, temporary_name, public_parent, destination.name)
                _sync_directory(public_parent)
                directory.check()
        finally:
            try:
                os.unlink(temporary_name, dir_fd=public_parent.descriptor)
            except FileNotFoundError:
                pass
    return destination
