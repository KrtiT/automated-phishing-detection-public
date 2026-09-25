"""Closed scientific-checkpoint schemas, separate from execution acceptance."""

import json
from dataclasses import asdict
from hashlib import sha256

from . import fixed_cascade
from ._checkpoint_codec import canonical_bytes
from .bound_secondary import _SEEDS, _TABULAR_NAMES, SecondaryInferenceCounts
from .selective_inference import InferenceCounts

SCIENTIFIC_CHECKPOINT_PROTOCOL = "internal-scientific-checkpoints-v1"
SCIENTIFIC_DIRECTORY = "scientific-checkpoints"
SCIENTIFIC_CHECKPOINT_ORDER = (
    "context.json",
    "bindings.json",
    "manifests.json",
    "primary-scores.jsonl",
    "primary-completion.json",
    *(f"secondary-tabular-{name}.json" for name in _TABULAR_NAMES),
    *(f"secondary-seed-{seed}.json" for seed in _SEEDS),
    "predictions.jsonl",
    "routing.json",
    "secondary.json",
    "completion.json",
)
SCIENTIFIC_CHECKPOINT_NAMES = frozenset(SCIENTIFIC_CHECKPOINT_ORDER)
SCIENTIFIC_OUTPUT_NAMES = frozenset(
    {
        "predictions.jsonl",
        "routing.json",
        "manifests.json",
        "bindings.json",
        "secondary.json",
    }
)
SOURCE_CHECKPOINT_NAMES = frozenset(
    {"group_test.jsonl", "source-overlap.json", "source-reconstruction.json"}
)


class ScientificCheckpointError(ValueError):
    """Scientific retention failed without exposing private values."""


def require(condition: bool) -> None:
    if not condition:
        raise ScientificCheckpointError("invalid_scientific_checkpoint")


def loads(content: bytes) -> dict:
    require(type(content) is bytes)
    value = json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )
    require(type(value) is dict and canonical_bytes(value) == content)
    return value


def hashes(contents: dict[str, bytes]) -> dict[str, str]:
    return {name: sha256(content).hexdigest() for name, content in contents.items()}


def validate_source_hashes(value: object) -> None:
    require(type(value) is dict and set(value) == SOURCE_CHECKPOINT_NAMES)
    for digest in value.values():
        fixed_cascade._lowercase_sha256(digest, "source_checkpoint_hash")


def _validate_context(
    identity: dict,
    reservation_sha256: str,
    source_hashes: dict[str, str],
    bindings: bytes,
    record_ids: tuple[str, ...],
) -> None:
    require(type(identity) is dict)
    require(
        identity.get("scientific_checkpoint_protocol") == SCIENTIFIC_CHECKPOINT_PROTOCOL
    )
    validate_source_hashes(source_hashes)
    fixed_cascade._lowercase_sha256(reservation_sha256, "reservation_hash")
    require(type(bindings) is bytes and type(record_ids) is tuple)
    require(
        bool(record_ids)
        and all(type(value) is str and bool(value) for value in record_ids)
    )
    require(len(set(record_ids)) == len(record_ids))


def context_bytes(
    identity: dict,
    reservation_sha256: str,
    source_hashes: dict[str, str],
    bindings: bytes,
    record_ids: tuple[str, ...],
) -> bytes:
    _validate_context(identity, reservation_sha256, source_hashes, bindings, record_ids)
    return canonical_bytes(
        {
            "schema_version": 1,
            "protocol_id": SCIENTIFIC_CHECKPOINT_PROTOCOL,
            "reservation_sha256": reservation_sha256,
            "execution": identity,
            "source_checkpoint_sha256": source_hashes,
            "bindings_sha256": sha256(bindings).hexdigest(),
            "record_ids": record_ids,
            "checkpoint_order": SCIENTIFIC_CHECKPOINT_ORDER,
        }
    )


def expected_counts(count: int) -> tuple[InferenceCounts, SecondaryInferenceCounts]:
    return InferenceCounts(count, count, count, 0), SecondaryInferenceCounts(
        tuple((name, count) for name in _TABULAR_NAMES),
        tuple((seed, 0 if seed == 42 else count) for seed in _SEEDS),
        count,
    )


def _validate_counts(
    contents: dict[str, bytes],
    inference_counts: InferenceCounts,
    secondary_inference_counts: SecondaryInferenceCounts,
) -> None:
    require(set(contents) == SCIENTIFIC_CHECKPOINT_NAMES - {"completion.json"})
    require(type(inference_counts) is InferenceCounts)
    require(type(secondary_inference_counts) is SecondaryInferenceCounts)
    count = len(loads(contents["context.json"])["record_ids"])
    primary, secondary = expected_counts(count)
    require(
        canonical_bytes(asdict(inference_counts)) == canonical_bytes(asdict(primary))
    )
    require(
        canonical_bytes(asdict(secondary_inference_counts))
        == canonical_bytes(asdict(secondary))
    )


def completion_bytes(
    contents: dict[str, bytes],
    reservation_sha256: str,
    inference_counts: InferenceCounts,
    secondary_inference_counts: SecondaryInferenceCounts,
) -> bytes:
    _validate_counts(contents, inference_counts, secondary_inference_counts)
    return canonical_bytes(
        {
            "schema_version": 1,
            "phase": "internal_scientific",
            "reservation_sha256": reservation_sha256,
            "context_sha256": sha256(contents["context.json"]).hexdigest(),
            "checkpoint_sha256": hashes(contents),
            "private_sha256": hashes(
                {name: contents[name] for name in SCIENTIFIC_OUTPUT_NAMES}
            ),
            "inference_counts": asdict(inference_counts),
            "secondary_inference_counts": asdict(secondary_inference_counts),
        }
    )
