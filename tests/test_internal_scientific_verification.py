"""Successful scientific checkpoints agree with every saved scientific byte."""

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from internal_scientific_fixtures import (
    ORDER,
    checkpoint_module,
    complete,
    verification_module,
    writer,
)
from internal_scientific_fixtures import (
    scientific as scientific,
)

from automated_phishing_detection import evaluation_producer, execution_receipt


def _snapshot(tmp_path: Path, scientific: SimpleNamespace) -> tuple:
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity=scientific.identity
    )
    retain = writer(module, attempt, scientific)
    complete(retain, scientific)
    outputs = {
        name: (attempt.directory / "scientific-checkpoints" / name).read_bytes()
        for name in ORDER
    }
    return attempt, outputs


def _verify(
    module: object, attempt: object, outputs: dict, scientific: SimpleNamespace
) -> None:
    module.verify_scientific_checkpoints(
        outputs,
        scientific.private,
        identity=scientific.identity,
        reservation_sha256=attempt.reservation_sha256,
        source_checkpoint_sha256=scientific.source_hashes,
    )


def _refresh_completion(outputs: dict[str, bytes]) -> None:
    completion = json.loads(outputs["completion.json"])
    completion["context_sha256"] = sha256(outputs["context.json"]).hexdigest()
    completion["checkpoint_sha256"] = {
        name: sha256(content).hexdigest()
        for name, content in outputs.items()
        if name != "completion.json"
    }
    outputs["completion.json"] = evaluation_producer._json_bytes(completion)


def test_successful_verification_is_byte_only(
    tmp_path: Path, scientific: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = verification_module()
    attempt, outputs = _snapshot(tmp_path, scientific)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("scientific checkpoint verification attempted I/O or scoring")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(evaluation_producer, "score_primary_url", forbidden)
    _verify(module, attempt, outputs, scientific)


@pytest.mark.parametrize("name", ORDER)
def test_missing_scientific_checkpoint_is_rejected(
    tmp_path: Path, scientific: SimpleNamespace, name: str
) -> None:
    module = verification_module()
    attempt, outputs = _snapshot(tmp_path, scientific)
    del outputs[name]
    with pytest.raises(module.ScientificCheckpointVerificationError):
        _verify(module, attempt, outputs, scientific)


@pytest.mark.parametrize("name", ORDER)
def test_mutated_scientific_checkpoint_is_rejected(
    tmp_path: Path, scientific: SimpleNamespace, name: str
) -> None:
    module = verification_module()
    attempt, outputs = _snapshot(tmp_path, scientific)
    outputs[name] += b" "
    with pytest.raises(module.ScientificCheckpointVerificationError):
        _verify(module, attempt, outputs, scientific)


@pytest.mark.parametrize(
    "field", ["execution", "source_checkpoint_sha256", "record_ids"]
)
def test_rehashed_context_forgery_is_rejected(
    tmp_path: Path, scientific: SimpleNamespace, field: str
) -> None:
    module = verification_module()
    attempt, outputs = _snapshot(tmp_path, scientific)
    context = json.loads(outputs["context.json"])
    context[field] = "private.canary"
    outputs["context.json"] = evaluation_producer._json_bytes(context)
    _refresh_completion(outputs)
    with pytest.raises(module.ScientificCheckpointVerificationError) as caught:
        _verify(module, attempt, outputs, scientific)
    assert "private.canary" not in str(caught.value)


@pytest.mark.parametrize("name", ORDER[5:17])
def test_rehashed_completed_column_must_match_final_score_projection(
    tmp_path: Path, scientific: SimpleNamespace, name: str
) -> None:
    module = verification_module()
    attempt, outputs = _snapshot(tmp_path, scientific)
    column = json.loads(outputs[name])
    score = column["column"]["scores"][0]
    probability = "probability" if "probability" in score else "transformer_probability"
    score[probability] = 0.125 if score[probability] != 0.125 else 0.25
    outputs[name] = evaluation_producer._json_bytes(column)
    _refresh_completion(outputs)
    with pytest.raises(module.ScientificCheckpointVerificationError):
        _verify(module, attempt, outputs, scientific)


@pytest.mark.parametrize(
    "name",
    [
        "bindings.json",
        "manifests.json",
        "predictions.jsonl",
        "routing.json",
        "secondary.json",
    ],
)
def test_final_private_inventory_is_exact_and_matches_scientific_copy(
    tmp_path: Path, scientific: SimpleNamespace, name: str
) -> None:
    module = verification_module()
    attempt, outputs = _snapshot(tmp_path, scientific)
    scientific.private[name] += b" "
    with pytest.raises(module.ScientificCheckpointVerificationError):
        _verify(module, attempt, outputs, scientific)


@pytest.mark.parametrize(
    "field",
    [
        "row_count",
        "partition_sha256",
        "bindings_sha256",
        "primary_scores_sha256",
        "inference_counts",
    ],
)
def test_rehashed_primary_completion_must_match_complete_observations(
    tmp_path: Path, scientific: SimpleNamespace, field: str
) -> None:
    module = verification_module()
    attempt, outputs = _snapshot(tmp_path, scientific)
    receipt = json.loads(outputs["primary-completion.json"])
    receipt[field] = "private.canary"
    outputs["primary-completion.json"] = evaluation_producer._json_bytes(receipt)
    _refresh_completion(outputs)
    with pytest.raises(module.ScientificCheckpointVerificationError) as caught:
        _verify(module, attempt, outputs, scientific)
    assert "private.canary" not in str(caught.value)
