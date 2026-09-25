"""Private failure sidecars retain known state without accepting an execution."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from internal_scientific_fixtures import PROTOCOL, checkpoint_module

from automated_phishing_detection import evaluation_producer, execution_receipt


@pytest.fixture
def scientific() -> SimpleNamespace:
    return SimpleNamespace(
        identity={"scientific_checkpoint_protocol": PROTOCOL},
        source_hashes={
            name: "a" * 64
            for name in (
                "group_test.jsonl",
                "source-overlap.json",
                "source-reconstruction.json",
            )
        },
    )


def _failure(attempt: object, scientific: SimpleNamespace, checkpoints: object) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "failed",
        "reservation_sha256": attempt.reservation_sha256,
        "execution": scientific.identity,
        "source_checkpoint_sha256": scientific.source_hashes,
        "stage": "scoring",
        "producer": {"inference_counts": None},
        "checkpoints": checkpoints,
        "failure": "cancelled",
        "cleanup_failed": False,
    }


def test_failure_sidecar_precedes_finalization_and_cannot_be_rewritten(
    tmp_path: Path, scientific: SimpleNamespace
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    content = evaluation_producer._json_bytes(_failure(attempt, scientific, None))
    module.retain_failure_progress(attempt, content)
    assert (attempt.directory / "failure-progress.json").read_bytes() == content
    with pytest.raises(module.ScientificCheckpointError):
        module.retain_failure_progress(attempt, content)
    execution_receipt.record_failure(attempt, stage="scoring", error_type="cancelled")
    assert (
        json.loads((attempt.directory / "outcome.json").read_bytes())["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "field", ["reservation_sha256", "protocol_id", "confirmed_sha256"]
)
def test_failure_sidecar_rejects_forged_nested_writer_linkage(
    tmp_path: Path, scientific: SimpleNamespace, field: str
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    snapshot = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "reservation_sha256": attempt.reservation_sha256,
        "status": "pending",
        "confirmed_sha256": {},
        "pending_checkpoint_bytes": {},
    }
    snapshot[field] = "private.canary"
    content = evaluation_producer._json_bytes(_failure(attempt, scientific, snapshot))
    with pytest.raises(module.ScientificCheckpointError) as caught:
        module.retain_failure_progress(attempt, content)
    assert "private.canary" not in str(caught.value)


@pytest.mark.parametrize("field", ["failure", "stage", "cleanup_failed", "private"])
def test_failure_sidecar_envelope_is_closed_and_private_safe(
    tmp_path: Path, scientific: SimpleNamespace, field: str
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    value = _failure(attempt, scientific, None)
    value[field] = "private.canary"
    with pytest.raises(module.ScientificCheckpointError) as caught:
        module.retain_failure_progress(attempt, evaluation_producer._json_bytes(value))
    assert "private.canary" not in str(caught.value)


def test_failure_sidecar_is_forbidden_after_finalization(
    tmp_path: Path, scientific: SimpleNamespace
) -> None:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    execution_receipt.record_failure(attempt, stage="scoring", error_type="cancelled")
    with pytest.raises(module.ScientificCheckpointError):
        module.retain_failure_progress(
            attempt,
            evaluation_producer._json_bytes(_failure(attempt, scientific, None)),
        )
