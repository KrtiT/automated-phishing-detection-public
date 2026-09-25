"""Invented completed internal observations, without source or model files."""

import importlib
import importlib.util
import json
from dataclasses import asdict, replace
from hashlib import sha256
from types import ModuleType, SimpleNamespace

import pytest
from test_evaluation_producer import parse, synthetic_session

from automated_phishing_detection import bound_secondary, evaluation_producer
from automated_phishing_detection._external_secondary_checkpoints import (
    SECONDARY_CHECKPOINTS,
    column_bytes,
    project_member_bindings,
)

PROTOCOL = "internal-scientific-checkpoints-v1"
ORDER = (
    "context.json",
    "bindings.json",
    "manifests.json",
    "primary-scores.jsonl",
    "primary-completion.json",
    *SECONDARY_CHECKPOINTS,
    "predictions.jsonl",
    "routing.json",
    "secondary.json",
    "completion.json",
)
PRIVATE_NAMES = (
    "bindings.json",
    "predictions.jsonl",
    "routing.json",
    "manifests.json",
    "secondary.json",
)


def checkpoint_module() -> ModuleType:
    name = "automated_phishing_detection.internal_scientific_checkpoints"
    assert importlib.util.find_spec(name) is not None, "scientific writer missing"
    return importlib.import_module(name)


def verification_module() -> ModuleType:
    name = "automated_phishing_detection.internal_scientific_verification"
    assert importlib.util.find_spec(name) is not None, "scientific verifier missing"
    return importlib.import_module(name)


def _columns(produced: object, bindings: dict, primary: bytes) -> dict[str, bytes]:
    rows = produced.rows
    count = len(rows)
    columns = tuple(
        bound_secondary.CompletedTabularColumn(
            name, tuple(row.secondary_tabular[index] for row in rows), count
        )
        for index, name in enumerate(bound_secondary._TABULAR_NAMES)
    ) + tuple(
        bound_secondary.CompletedSeedColumn(
            seed,
            tuple(row.secondary_seeds[index] for row in rows),
            0 if seed == 42 else count,
            count if seed == 42 else 0,
        )
        for index, seed in enumerate(bound_secondary._SEEDS)
    )
    members = project_member_bindings(bindings["secondary"])
    return {
        name: column_bytes(
            sha256(primary).hexdigest(),
            tuple(row.record.record_id for row in rows),
            member,
            column,
        )
        for name, member, column in zip(
            SECONDARY_CHECKPOINTS, members, columns, strict=True
        )
    }


def _phase_outputs(produced: object) -> dict[str, bytes]:
    from automated_phishing_detection.source_runner import _secondary

    bindings = json.loads(produced.private_outputs["bindings.json"])
    primary = b"".join(
        evaluation_producer._json_bytes(
            asdict(replace(row, secondary_tabular=(), secondary_seeds=()))
        )
        for row in produced.rows
    )
    receipt = evaluation_producer._json_bytes(
        {
            "schema_version": 1,
            "phase": "internal_primary",
            "row_count": len(produced.rows),
            "partition_sha256": bindings["partition_sha256"],
            "bindings_sha256": sha256(
                produced.private_outputs["bindings.json"]
            ).hexdigest(),
            "primary_scores_sha256": sha256(primary).hexdigest(),
            "inference_counts": asdict(produced.inference_counts),
        }
    )
    return {
        **produced.private_outputs,
        **_columns(produced, bindings, primary),
        "primary-scores.jsonl": primary,
        "primary-completion.json": receipt,
        "secondary.json": evaluation_producer._json_bytes(_secondary(produced)),
    }


@pytest.fixture
def scientific(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    prepared = parse(evaluation_producer)
    session, *unused = synthetic_session(evaluation_producer, monkeypatch)
    produced = evaluation_producer.produce_internal_evidence(prepared, session)
    payloads = _phase_outputs(produced)
    private = {name: payloads[name] for name in PRIVATE_NAMES}
    return SimpleNamespace(
        identity={
            "kind": "internal_evaluation",
            "scientific_checkpoint_protocol": PROTOCOL,
        },
        source_hashes={
            name: sha256(name.encode()).hexdigest()
            for name in (
                "group_test.jsonl",
                "source-overlap.json",
                "source-reconstruction.json",
            )
        },
        record_ids=tuple(row.record.record_id for row in produced.rows),
        counts=produced.inference_counts,
        secondary_counts=produced.secondary_inference_counts,
        payloads=payloads,
        private=private,
    )


def writer(module: ModuleType, attempt: object, scientific: SimpleNamespace) -> object:
    return module.ScientificCheckpointWriter(
        attempt,
        identity=scientific.identity,
        source_checkpoint_sha256=scientific.source_hashes,
        record_ids=scientific.record_ids,
    )


def complete(retain: object, scientific: SimpleNamespace) -> dict[str, str]:
    for name in ORDER[1:-1]:
        retain(name, scientific.payloads[name])
    return retain.complete(
        inference_counts=scientific.counts,
        secondary_inference_counts=scientific.secondary_counts,
    )
