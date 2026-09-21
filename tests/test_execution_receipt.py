"""Durable execution-receipt tests use temporary synthetic bytes only."""

import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from hashlib import sha256
from importlib import import_module

import pytest


@pytest.fixture
def receipt():
    return import_module("automated_phishing_detection.execution_receipt")


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def reserve(receipt, tmp_path):
    return receipt.reserve_attempt(
        tmp_path / "attempt", identity={"run": "synthetic-1"}
    )


def complete(receipt, attempt, public_path):
    return receipt.publish_completion(
        attempt,
        private_outputs={"scores.bin": b"private synthetic bytes"},
        public_summary={"status": "completed", "rows": 3},
        public_path=public_path,
    )


def test_reservation_is_canonical_private_and_immutable(receipt, tmp_path):
    attempt = reserve(receipt, tmp_path)
    expected = canonical(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(tmp_path / "attempt"),
            "identity": {"run": "synthetic-1"},
        }
    )
    assert attempt.directory == tmp_path / "attempt"
    assert (attempt.directory / "reservation.json").read_bytes() == expected
    assert attempt.reservation_sha256 == sha256(expected).hexdigest()
    assert stat.S_IMODE(attempt.directory.stat().st_mode) == 0o700
    assert (
        stat.S_IMODE((attempt.directory / "reservation.json").stat().st_mode) == 0o600
    )
    with pytest.raises(FrozenInstanceError):
        attempt.reservation_sha256 = "a" * 64


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling"])
def test_existing_attempt_or_tombstone_is_never_reused(receipt, tmp_path, kind):
    destination = tmp_path / "attempt"
    if kind == "directory":
        destination.mkdir()
    elif kind == "file":
        destination.write_bytes(b"prior reservation")
    else:
        target = tmp_path / "target"
        if kind == "symlink":
            target.mkdir()
        destination.symlink_to(target, target_is_directory=True)
    before = destination.lstat()
    with pytest.raises(receipt.ExecutionReceiptError):
        reserve(receipt, tmp_path)
    assert destination.lstat().st_ino == before.st_ino


@pytest.mark.parametrize(
    "identity",
    [
        None,
        {},
        [],
        {1: "bad"},
        {"n": float("nan")},
        {"nested": [float("inf")]},
        {"bad": b"bytes"},
        {"tuple": (1, 2)},
    ],
)
def test_identity_is_nonempty_strict_finite_json_before_reservation(
    receipt, tmp_path, identity
):
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.reserve_attempt(tmp_path / "attempt", identity=identity)
    assert not (tmp_path / "attempt").exists()


def test_reservation_requires_real_existing_parent(receipt, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    for path in (
        alias / "attempt",
        tmp_path / "missing" / "attempt",
        tmp_path / "child" / ".." / "attempt",
    ):
        with pytest.raises(receipt.ExecutionReceiptError):
            receipt.reserve_attempt(path, identity={"run": 1})


def test_concurrent_reservations_have_exactly_one_winner(receipt, tmp_path):
    def attempt_reservation(index):
        try:
            return receipt.reserve_attempt(
                tmp_path / "attempt", identity={"run": index}
            )
        except receipt.ExecutionReceiptError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt_reservation, range(4)))
    winners = [result for result in results if type(result) is receipt.Attempt]
    assert len(winners) == 1
    assert (
        sha256((winners[0].directory / "reservation.json").read_bytes()).hexdigest()
        == winners[0].reservation_sha256
    )


def test_failure_receipt_is_sanitized_and_excludes_exception_text(receipt, tmp_path):
    attempt = reserve(receipt, tmp_path)
    path = receipt.record_failure(
        attempt, stage="model_scoring", error_type="RuntimeError"
    )
    assert path == attempt.directory / "outcome.json"
    assert json.loads(path.read_bytes()) == {
        "schema_version": 1,
        "status": "failed",
        "reservation_sha256": attempt.reservation_sha256,
        "stage": "model_scoring",
        "error_type": "RuntimeError",
    }
    assert set(entry.name for entry in attempt.directory.iterdir()) == {
        "reservation.json",
        "finalize.claim",
        "outcome.json",
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, tmp_path / "public.json")
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.record_failure(attempt, stage="retry", error_type="ValueError")


@pytest.mark.parametrize(
    "field,bad",
    [
        ("stage", ""),
        ("stage", "private/path"),
        ("stage", "https://private.example/"),
        ("error_type", "RuntimeError: private message"),
        ("error_type", RuntimeError("private")),
        ("error_type", None),
    ],
)
def test_failure_fields_are_symbolic_not_free_text(receipt, tmp_path, field, bad):
    attempt = reserve(receipt, tmp_path)
    values = {"stage": "scoring", "error_type": "RuntimeError", field: bad}
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.record_failure(attempt, **values)
    assert not (attempt.directory / "finalize.claim").exists()


@pytest.mark.parametrize(
    "change", ["bytes", "symlink", "hardlink", "directory", "moved"]
)
def test_reservation_is_authenticated_without_following_aliases(
    receipt, tmp_path, change
):
    attempt = reserve(receipt, tmp_path)
    path = attempt.directory / "reservation.json"
    content = path.read_bytes()
    if change == "bytes":
        path.write_bytes(content + b"\n")
    elif change == "moved":
        moved = tmp_path / "moved"
        attempt.directory.rename(moved)
        attempt = receipt.Attempt(moved, attempt.reservation_sha256)
    else:
        path.unlink()
        target = tmp_path / "original.json"
        target.write_bytes(content)
        if change == "symlink":
            path.symlink_to(target)
        elif change == "hardlink":
            os.link(target, path)
        else:
            path.mkdir()
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.record_failure(attempt, stage="scoring", error_type="RuntimeError")
    assert not (attempt.directory / "finalize.claim").exists()


def test_completion_installs_private_evidence_and_outcome_before_public_marker(
    receipt, tmp_path, monkeypatch
):
    attempt = reserve(receipt, tmp_path)
    public_parent = tmp_path / "public"
    public_parent.mkdir()
    public_path = public_parent / "summary.json"
    original = receipt._publish
    destinations = []

    def observe(source, source_name, destination, destination_name):
        destinations.append(destination_name)
        if destination.path / destination_name == public_path:
            assert (
                attempt.directory / "evidence" / "scores.bin"
            ).read_bytes() == b"private synthetic bytes"
            outcome = json.loads((attempt.directory / "outcome.json").read_bytes())
            assert outcome["status"] == "completion_prepared"
        original(source, source_name, destination, destination_name)

    monkeypatch.setattr(receipt, "_publish", observe)
    assert complete(receipt, attempt, public_path) == public_path
    assert destinations == [
        "finalize.claim",
        "evidence",
        "outcome.json",
        "summary.json",
    ]
    public_bytes = canonical({"status": "completed", "rows": 3})
    assert public_path.read_bytes() == public_bytes
    assert b"private" not in public_bytes and str(tmp_path).encode() not in public_bytes
    assert json.loads((attempt.directory / "outcome.json").read_bytes()) == {
        "schema_version": 1,
        "status": "completion_prepared",
        "reservation_sha256": attempt.reservation_sha256,
        "public_summary_sha256": sha256(public_bytes).hexdigest(),
        "private_sha256": {
            "scores.bin": sha256(b"private synthetic bytes").hexdigest()
        },
    }
    assert stat.S_IMODE((attempt.directory / "evidence").stat().st_mode) == 0o700
    assert (
        stat.S_IMODE((attempt.directory / "evidence" / "scores.bin").stat().st_mode)
        == 0o600
    )
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o644
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, tmp_path / "another.json")


@pytest.mark.parametrize(
    "outputs",
    [
        {},
        {"../escape": b"x"},
        {"/absolute": b"x"},
        {"a/b": b"x"},
        {"a\\b": b"x"},
        {".": b"x"},
        {"..": b"x"},
        {"x": bytearray(b"x")},
        {"x": "not bytes"},
        {1: b"x"},
        None,
    ],
)
def test_private_output_payloads_are_validated_before_claim(receipt, tmp_path, outputs):
    attempt = reserve(receipt, tmp_path)
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.publish_completion(
            attempt,
            private_outputs=outputs,
            public_summary={"rows": 1},
            public_path=tmp_path / "public.json",
        )
    assert not (attempt.directory / "finalize.claim").exists()


@pytest.mark.parametrize(
    "summary", [{}, {"n": float("nan")}, {"private": b"raw bytes"}, {1: 2}, [], None]
)
def test_public_summary_is_validated_before_claim(receipt, tmp_path, summary):
    attempt = reserve(receipt, tmp_path)
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.publish_completion(
            attempt,
            private_outputs={"x": b"x"},
            public_summary=summary,
            public_path=tmp_path / "public.json",
        )
    assert not (attempt.directory / "finalize.claim").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "file",
        "directory",
        "symlink",
        "dangling",
        "inside",
        "parent_alias",
        "missing_parent",
    ],
)
def test_public_destination_cannot_overwrite_or_alias_private_evidence(
    receipt, tmp_path, kind
):
    attempt = reserve(receipt, tmp_path)
    path = tmp_path / "public.json"
    if kind == "file":
        path.write_bytes(b"keep")
    elif kind == "directory":
        path.mkdir()
    elif kind in ("symlink", "dangling"):
        path.symlink_to(
            attempt.directory / ("reservation.json" if kind == "symlink" else "missing")
        )
    elif kind == "inside":
        path = attempt.directory / "public.json"
    elif kind == "parent_alias":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / "public.json"
    else:
        path = tmp_path / "missing" / "public.json"
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, path)
    assert not (attempt.directory / "finalize.claim").exists()


def test_failure_and_completion_compete_for_one_permanent_claim(receipt, tmp_path):
    attempt = reserve(receipt, tmp_path)
    public_path = tmp_path / "public.json"

    def finalize(operation):
        try:
            if operation == "failure":
                return receipt.record_failure(
                    attempt, stage="scoring", error_type="RuntimeError"
                )
            return complete(receipt, attempt, public_path)
        except receipt.ExecutionReceiptError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(finalize, ("failure", "completion")))
    assert (
        sum(isinstance(result, receipt.ExecutionReceiptError) for result in results)
        == 1
    )
    status = json.loads((attempt.directory / "outcome.json").read_bytes())["status"]
    assert public_path.exists() == (status == "completion_prepared")


@pytest.mark.parametrize("attempt", [None, {}, "attempt"])
def test_failure_requires_typed_authenticated_attempt(receipt, attempt):
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.record_failure(attempt, stage="scoring", error_type="RuntimeError")


@pytest.mark.parametrize("name", ["finalize.claim", "evidence", "outcome.json"])
def test_finalization_tombstones_are_never_replaced(receipt, tmp_path, name):
    attempt = reserve(receipt, tmp_path)
    tombstone = attempt.directory / name
    tombstone.write_bytes(b"prior interrupted attempt")
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, tmp_path / "public.json")
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.record_failure(attempt, stage="scoring", error_type="RuntimeError")
    assert tombstone.read_bytes() == b"prior interrupted attempt"


def test_racing_public_marker_is_not_overwritten_or_cleaned_up(
    receipt, tmp_path, monkeypatch
):
    attempt = reserve(receipt, tmp_path)
    public_path = tmp_path / "public.json"
    original = receipt._publish

    def install_competing_marker(source, source_name, destination, destination_name):
        if destination.path / destination_name == public_path:
            public_path.write_bytes(b"competing immutable marker")
        original(source, source_name, destination, destination_name)

    monkeypatch.setattr(receipt, "_publish", install_competing_marker)
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, public_path)
    assert public_path.read_bytes() == b"competing immutable marker"
    assert (attempt.directory / "evidence" / "scores.bin").is_file()
    assert (attempt.directory / "outcome.json").is_file()


@pytest.mark.parametrize(
    "destination_name,after",
    [
        ("evidence", False),
        ("evidence", True),
        ("outcome.json", False),
        ("public.json", False),
        ("public.json", True),
    ],
)
def test_publication_failures_leave_consumed_attempt_and_installed_evidence(
    receipt, tmp_path, monkeypatch, destination_name, after
):
    attempt = reserve(receipt, tmp_path)
    original = receipt._publish

    def interrupted(source, source_name, destination, installed_name):
        if installed_name != destination_name:
            return original(source, source_name, destination, installed_name)
        if after:
            original(source, source_name, destination, installed_name)
        raise OSError("synthetic interrupted publication")

    monkeypatch.setattr(receipt, "_publish", interrupted)
    with pytest.raises(OSError, match="synthetic"):
        complete(receipt, attempt, tmp_path / "public.json")
    assert (attempt.directory / "reservation.json").is_file()
    assert (attempt.directory / "finalize.claim").is_file()
    assert (attempt.directory / "evidence").exists() == (
        destination_name != "evidence" or after
    )
    assert (tmp_path / "public.json").exists() == (
        destination_name == "public.json" and after
    )
    monkeypatch.setattr(receipt, "_publish", original)
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, tmp_path / "retry.json")
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.record_failure(attempt, stage="retry", error_type="OSError")
    with pytest.raises(receipt.ExecutionReceiptError):
        reserve(receipt, tmp_path)


@pytest.mark.parametrize("after", [False, True])
def test_public_parent_swap_at_rename_never_redirects_publication(
    receipt, tmp_path, monkeypatch, after
):
    attempt = reserve(receipt, tmp_path)
    public_parent = tmp_path / "public"
    public_parent.mkdir()
    moved = tmp_path / "original-public"
    public_path = public_parent / "summary.json"
    original = receipt._rename_noreplace

    def replace_at_rename(source, source_name, destination, destination_name):
        if destination_name != "summary.json":
            return original(source, source_name, destination, destination_name)
        if after:
            original(source, source_name, destination, destination_name)
        public_parent.rename(moved)
        public_parent.symlink_to(attempt.directory, target_is_directory=True)
        if not after:
            original(source, source_name, destination, destination_name)

    monkeypatch.setattr(receipt, "_rename_noreplace", replace_at_rename)
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, public_path)
    assert not (attempt.directory / "summary.json").exists()
    assert (moved / "summary.json").is_file()
    assert (attempt.directory / "evidence" / "scores.bin").is_file()
    assert (attempt.directory / "outcome.json").is_file()


def test_attempt_parent_swap_during_private_write_does_not_write_into_replacement(
    receipt, tmp_path, monkeypatch
):
    attempt = reserve(receipt, tmp_path)
    original_directory = tmp_path / "original-attempt"
    replacement = tmp_path / "replacement"
    original_open = receipt.os.open
    swapped = False

    def swap_before_private_write(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and flags & os.O_CREAT and os.fspath(path) == "scores.bin":
            swapped = True
            shutil.copytree(attempt.directory, replacement)
            attempt.directory.rename(original_directory)
            attempt.directory.symlink_to(replacement, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(receipt.os, "open", swap_before_private_write)
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, tmp_path / "public.json")
    assert swapped
    assert list(replacement.rglob("scores.bin")) == []
    assert (original_directory / "reservation.json").is_file()
    assert (original_directory / "finalize.claim").is_file()
    assert not (tmp_path / "public.json").exists()


def test_reservation_parent_swap_at_install_uses_pinned_directory_and_fails_closed(
    receipt, tmp_path, monkeypatch
):
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "original-parent"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original = receipt._rename_noreplace

    def replace_at_install(source, source_name, destination, destination_name):
        if destination_name == "attempt":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)
        original(source, source_name, destination, destination_name)

    monkeypatch.setattr(receipt, "_rename_noreplace", replace_at_install)
    with pytest.raises(receipt.ExecutionReceiptError):
        reserve(receipt, parent)
    assert (moved / "attempt" / "reservation.json").is_file()
    assert not (replacement / "attempt").exists()


@pytest.mark.parametrize("phase", ["reservation", "claim", "private", "public"])
def test_post_install_fsync_failure_preserves_one_sided_evidence(
    receipt, tmp_path, monkeypatch, phase
):
    original = receipt._sync_directory
    attempt = None if phase == "reservation" else reserve(receipt, tmp_path)
    directory = tmp_path / "attempt"
    public_path = tmp_path / "public.json"

    def fail_after_install(path):
        installed = {
            "reservation": directory.exists(),
            "claim": (directory / "finalize.claim").exists(),
            "private": (directory / "evidence").exists(),
            "public": public_path.exists(),
        }[phase]
        target = directory if phase in {"claim", "private"} else tmp_path
        if installed and path.path == target:
            raise OSError("synthetic fsync failure")
        original(path)

    monkeypatch.setattr(receipt, "_sync_directory", fail_after_install)
    with pytest.raises(OSError, match="fsync"):
        if phase == "reservation":
            reserve(receipt, tmp_path)
        else:
            complete(receipt, attempt, public_path)
    assert (directory / "reservation.json").is_file()
    assert (directory / "evidence").exists() == (phase in {"private", "public"})
    assert public_path.exists() == (phase == "public")
    with pytest.raises(receipt.ExecutionReceiptError):
        reserve(receipt, tmp_path)
    if attempt is not None:
        with pytest.raises(receipt.ExecutionReceiptError):
            complete(receipt, attempt, tmp_path / "retry.json")


def test_reservation_survives_an_exception_after_atomic_install(
    receipt, tmp_path, monkeypatch
):
    original = receipt._publish

    def interrupted(source, source_name, destination, destination_name):
        original(source, source_name, destination, destination_name)
        raise OSError("synthetic interruption after reservation install")

    monkeypatch.setattr(receipt, "_publish", interrupted)
    with pytest.raises(OSError, match="interruption"):
        reserve(receipt, tmp_path)
    assert (tmp_path / "attempt" / "reservation.json").is_file()
    with pytest.raises(receipt.ExecutionReceiptError):
        reserve(receipt, tmp_path)


def test_preinstall_reservation_failure_does_not_remove_existing_content(
    receipt, tmp_path, monkeypatch
):
    original = receipt._write_file

    def fail_write(directory, name, content, mode):
        original(directory, name, content, mode)
        raise OSError("synthetic file sync failure")

    monkeypatch.setattr(receipt, "_write_file", fail_write)
    with pytest.raises(OSError):
        reserve(receipt, tmp_path)
    assert not (tmp_path / "attempt").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_public_parent_replacement_before_temp_creation_fails_without_redirect(
    receipt, tmp_path, monkeypatch, replacement
):
    attempt = reserve(receipt, tmp_path)
    public_parent = tmp_path / "public"
    public_parent.mkdir()
    moved_parent = tmp_path / "original-public"
    public_path = public_parent / "summary.json"
    original_open = receipt.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and flags & os.O_CREAT
            and ".summary.json.tmp-" in os.fspath(path)
        ):
            swapped = True
            public_parent.rename(moved_parent)
            if replacement == "symlink":
                public_parent.symlink_to(attempt.directory, target_is_directory=True)
            else:
                public_parent.mkdir()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(receipt.os, "open", swap_before_open)
    with pytest.raises(receipt.ExecutionReceiptError):
        complete(receipt, attempt, public_path)
    assert swapped
    assert not public_path.exists()
    assert not (attempt.directory / "summary.json").exists()
    assert not any(
        entry.name.startswith(".summary.json.tmp-")
        for entry in attempt.directory.iterdir()
    )
    assert (attempt.directory / "reservation.json").is_file()
    assert (attempt.directory / "finalize.claim").is_file()
