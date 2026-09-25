"""Create-only ordered scientific retention survives ordinary writer failures."""

import base64
import json
import stat
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from internal_scientific_fixtures import (
    ORDER,
    PROTOCOL,
    checkpoint_module,
    complete,
    writer,
)
from internal_scientific_fixtures import (
    scientific as scientific,
)

from automated_phishing_detection import execution_receipt


def test_scientific_checkpoint_protocol_is_available() -> None:
    module = checkpoint_module()
    assert module.SCIENTIFIC_CHECKPOINT_PROTOCOL == PROTOCOL
    assert module.SCIENTIFIC_CHECKPOINT_ORDER == ORDER


@pytest.mark.parametrize("field", ["identity", "source_checkpoint_sha256"])
def test_constructor_serialization_errors_are_symbolic(
    tmp_path: Path, field: str
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"fixture": 1}
    )
    private_type = type("private_canary", (), {})
    arguments = {"identity": {}, "source_checkpoint_sha256": {}, "record_ids": ("row",)}
    arguments[field] = {"value": private_type()}
    with pytest.raises(module.ScientificCheckpointError) as caught:
        module.ScientificCheckpointWriter(attempt, **arguments)
    assert "private_canary" not in str(caught.value)


def test_complete_writer_retains_exact_private_inventory(
    tmp_path: Path, scientific: SimpleNamespace
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    retain = writer(module, attempt, scientific)
    hashes = complete(retain, scientific)
    directory = attempt.directory / "scientific-checkpoints"
    assert set(path.name for path in directory.iterdir()) == set(ORDER)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert hashes == {
        name: sha256((directory / name).read_bytes()).hexdigest() for name in ORDER
    }
    assert all(
        stat.S_IMODE((directory / name).stat().st_mode) == 0o600 for name in ORDER
    )
    context = json.loads((directory / "context.json").read_bytes())
    assert context["protocol_id"] == PROTOCOL
    assert context["reservation_sha256"] == attempt.reservation_sha256
    assert context["record_ids"] == list(scientific.record_ids)
    assert context["checkpoint_order"] == list(ORDER)
    assert json.loads(retain.snapshot())["status"] == "complete"


@pytest.mark.parametrize(
    "name", ["manifests.json", "context.json", "../private", "completion.json"]
)
def test_writer_rejects_unexpected_first_name(
    tmp_path: Path, scientific: SimpleNamespace, name: str
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    retain = writer(module, attempt, scientific)
    with pytest.raises(module.ScientificCheckpointError):
        retain(name, b"{}\n")
    with pytest.raises(module.ScientificCheckpointError):
        retain("bindings.json", scientific.payloads["bindings.json"])
    assert not (attempt.directory / "scientific-checkpoints").exists()


def _failing_installer(
    monkeypatch: pytest.MonkeyPatch, after_install: bool
) -> list[str]:
    original = execution_receipt._install_record
    calls: list[str] = []

    def fail(directory: object, name: str, content: bytes) -> None:
        calls.append(name)
        if after_install:
            original(directory, name, content)
        raise OSError("private path")

    monkeypatch.setattr(execution_receipt, "_install_record", fail)
    return calls


@pytest.mark.parametrize("after_install", [False, True])
def test_failed_install_retains_ambiguous_bytes_and_never_retries(
    tmp_path: Path,
    scientific: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    after_install: bool,
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    retain = writer(module, attempt, scientific)
    retain("bindings.json", scientific.payloads["bindings.json"])
    calls = _failing_installer(monkeypatch, after_install)
    with pytest.raises(
        module.ScientificCheckpointError, match="scientific_checkpoint_write_failed"
    ):
        retain("manifests.json", scientific.payloads["manifests.json"])
    snapshot = json.loads(retain.snapshot())
    assert (
        base64.b64decode(snapshot["pending_checkpoint_bytes"]["manifests.json"])
        == scientific.payloads["manifests.json"]
    )
    with pytest.raises(module.ScientificCheckpointError):
        retain("manifests.json", scientific.payloads["manifests.json"])
    assert calls == ["manifests.json"]


def test_finalized_attempt_cannot_start_scientific_retention(
    tmp_path: Path, scientific: SimpleNamespace
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    execution_receipt.record_failure(attempt, stage="fixture", error_type="ValueError")
    retain = writer(module, attempt, scientific)
    with pytest.raises(module.ScientificCheckpointError):
        retain("bindings.json", scientific.payloads["bindings.json"])


@pytest.mark.parametrize("kind", ["symlink", "replacement", "extra_file"])
def test_changed_scientific_directory_stops_next_write(
    tmp_path: Path, scientific: SimpleNamespace, kind: str
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    retain = writer(module, attempt, scientific)
    retain("bindings.json", scientific.payloads["bindings.json"])
    directory = attempt.directory / "scientific-checkpoints"
    if kind == "extra_file":
        (directory / "unexpected").write_bytes(b"private")
    else:
        directory.rename(attempt.directory / "original")
        if kind == "symlink":
            directory.symlink_to(
                attempt.directory / "original", target_is_directory=True
            )
        else:
            directory.mkdir(mode=0o700)
    with pytest.raises(module.ScientificCheckpointError):
        retain("manifests.json", scientific.payloads["manifests.json"])
