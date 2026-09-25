"""Closed byte schemas for consistency with observing-parent expectations."""

import re
from hashlib import sha256

from . import source_checkpoint_verification, source_overlap
from ._checkpoint_codec import canonical_bytes
from ._internal_scientific_protocol import (
    SCIENTIFIC_CHECKPOINT_NAMES,
    SCIENTIFIC_CHECKPOINT_PROTOCOL,
    SCIENTIFIC_OUTPUT_NAMES,
    SOURCE_CHECKPOINT_NAMES,
)

OVERLAP_NAME = "attempt/checkpoints/source-overlap.json"
SOURCE_LINKS = {
    "source_spec_sha256": "source/data/sources.json",
    "preparation_summary_sha256": "source/reports/phiusiil-preparation-summary.json",
    "reservation_sha256": "attempt/reservation.json",
    "partition_sha256": "attempt/checkpoints/group_test.jsonl",
}
SNAPSHOT_NAMES = frozenset(
    {
        "public-summary.json",
        "attempt/reservation.json",
        "attempt/finalize.claim",
        "attempt/outcome.json",
        "source/data/sources.json",
        "source/reports/phiusiil-preparation-summary.json",
        *(f"attempt/evidence/{name}" for name in SCIENTIFIC_OUTPUT_NAMES),
        *(f"attempt/checkpoints/{name}" for name in SOURCE_CHECKPOINT_NAMES),
        *(
            f"attempt/scientific-checkpoints/{name}"
            for name in SCIENTIFIC_CHECKPOINT_NAMES
        ),
    }
)
EXECUTION_CONSTANTS = {
    "kind": "internal_evaluation",
    "source_interface": "original_csv_reconstruction_v1",
    "scientific_checkpoint_protocol": SCIENTIFIC_CHECKPOINT_PROTOCOL,
}
EXECUTION_HASHES = frozenset(
    {
        *SOURCE_LINKS,
        "execution_contract_sha256",
        "runtime_sha256",
        "source_csv_sha256",
        "suffix_rules_sha256",
    }
)
OVERLAP_PINS = frozenset(
    {
        "source_csv_sha256",
        "suffix_rules_sha256",
        "source_spec_sha256",
        "preparation_summary_sha256",
    }
)
PREPARATION_NAMES = frozenset(
    {
        "train.jsonl",
        "validation.jsonl",
        "group_test.jsonl",
        "quarantine.jsonl",
        "SHA256SUMS",
    }
)


class InternalHandoffError(ValueError):
    """Symbolic consistency failure without source values or private paths."""


def require(condition: bool, reason: str = "invalid_internal_handoff") -> None:
    if not condition:
        raise InternalHandoffError(reason)


def keys(value: object, expected) -> None:
    require(type(value) is dict and set(value) == set(expected))


def digest(value: object) -> None:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def hash_map(value: object, expected) -> None:
    keys(value, expected)
    for item in value.values():
        digest(item)


def loads(content: bytes) -> dict:
    require(type(content) is bytes)
    value = source_checkpoint_verification._json(content)
    require(type(value) is dict and canonical_bytes(value) == content)
    return value


def worker(value: object) -> None:
    keys(value, {"command_sha256", "exit", "stdout_sha256", "stderr_sha256"})
    for name in ("command_sha256", "stdout_sha256", "stderr_sha256"):
        digest(value[name])
    exit_value = value["exit"]
    keys(exit_value, {"pid", "exit_observed", "exit_code"})
    require(type(exit_value["pid"]) is int and exit_value["pid"] > 0)
    require(exit_value["exit_observed"] is True)
    require(type(exit_value["exit_code"]) is int and exit_value["exit_code"] == 0)


def execution(value: object, snapshot_hashes: dict) -> None:
    require(type(value) is dict)
    constants, hashes = EXECUTION_CONSTANTS.copy(), EXECUTION_HASHES
    if value.get("source_interface") == "retained_study_preparation_v1":
        constants["source_interface"] = "retained_study_preparation_v1"
        hashes = hashes | {
            "study_preparation_reservation_sha256",
            "study_preparation_complete_sha256",
        }
    keys(value, {*constants, *hashes, "revision"})
    require(all(value[name] == item for name, item in constants.items()))
    require(type(value["revision"]) is str)
    require(re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is not None)
    for name in hashes:
        digest(value[name])
    require(
        all(value[name] == snapshot_hashes[path] for name, path in SOURCE_LINKS.items())
    )


def envelope(value: object) -> None:
    keys(value, {"schema_version", "kind", "execution", "worker", "snapshot_sha256"})
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["kind"] == "same-parent-internal-handoff-v1")
    hash_map(value["snapshot_sha256"], SNAPSHOT_NAMES)
    worker(value["worker"])
    execution(value["execution"], value["snapshot_sha256"])


def _overlap_identity(value: dict, identity: dict) -> None:
    keys(
        value,
        {
            "schema_version",
            "algorithm_id",
            "scope",
            "input_hashes",
            "reconstructed_output_sha256",
            "rows",
            "domains",
        },
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["algorithm_id"] == source_overlap._ALGORITHM_ID)
    require(value["scope"] == source_overlap._SCOPE)
    hash_map(value["input_hashes"], OVERLAP_PINS)
    require(value["input_hashes"] == {name: identity[name] for name in OVERLAP_PINS})
    hash_map(value["reconstructed_output_sha256"], PREPARATION_NAMES)
    require(
        value["reconstructed_output_sha256"]["group_test.jsonl"]
        == identity["partition_sha256"]
    )


def overlap(content: bytes, handoff: dict) -> frozenset[str]:
    require(sha256(content).hexdigest() == handoff["snapshot_sha256"][OVERLAP_NAME])
    value = loads(content)
    _overlap_identity(value, handoff["execution"])
    require(type(value["rows"]) is list)
    valid = source_checkpoint_verification._membership(
        value["rows"], handoff["execution"], len(value["rows"])
    )
    domains = sorted({domain for _, domain in valid.values()})
    require(value["domains"] == domains)
    return frozenset(domains)
