"""Compare complete scientific checkpoints with immutable final evidence bytes."""

from dataclasses import asdict
from hashlib import sha256

from . import bound_secondary as secondary
from ._checkpoint_codec import canonical_bytes, column_bytes, project_member_bindings
from ._external_secondary_validation import validate_column
from ._internal_scientific_protocol import (
    SCIENTIFIC_CHECKPOINT_NAMES,
    SCIENTIFIC_CHECKPOINT_ORDER,
    SCIENTIFIC_OUTPUT_NAMES,
    completion_bytes,
    context_bytes,
    expected_counts,
    loads,
    require,
)


class ScientificCheckpointVerificationError(ValueError):
    """A successful checkpoint inventory does not match its linked final evidence."""


def _inventory(contents: object, names: frozenset[str]) -> None:
    require(type(contents) is dict and set(contents) == names)
    require(
        all(
            type(name) is str and type(value) is bytes
            for name, value in contents.items()
        )
    )


def _primary(rows: tuple[dict, ...], binding: dict, contents: dict[str, bytes]) -> None:
    primary = b"".join(
        canonical_bytes({**row, "secondary_tabular": [], "secondary_seeds": []})
        for row in rows
    )
    require(primary == contents["primary-scores.jsonl"])
    counts, unused = expected_counts(len(rows))
    receipt = canonical_bytes(
        {
            "schema_version": 1,
            "phase": "internal_primary",
            "row_count": len(rows),
            "partition_sha256": binding["partition_sha256"],
            "bindings_sha256": sha256(contents["bindings.json"]).hexdigest(),
            "primary_scores_sha256": sha256(primary).hexdigest(),
            "inference_counts": asdict(counts),
        }
    )
    require(receipt == contents["primary-completion.json"])


def _column(rows: tuple[dict, ...], index: int) -> object:
    count = len(rows)
    if index < len(secondary._TABULAR_NAMES):
        name = secondary._TABULAR_NAMES[index]
        scores = tuple(
            secondary.SecondaryTabularScore(**row["secondary_tabular"][index])
            for row in rows
        )
        return secondary.CompletedTabularColumn(name, scores, count)
    seed_index = index - len(secondary._TABULAR_NAMES)
    seed = secondary._SEEDS[seed_index]
    scores = tuple(
        secondary.SecondarySeedScore(**row["secondary_seeds"][seed_index])
        for row in rows
    )
    return secondary.CompletedSeedColumn(
        seed, scores, 0 if seed == 42 else count, count if seed == 42 else 0
    )


def _columns(rows: tuple[dict, ...], binding: dict, contents: dict[str, bytes]) -> None:
    members = project_member_bindings(binding["secondary"])
    names = SCIENTIFIC_CHECKPOINT_ORDER[5:17]
    primary_hash = sha256(contents["primary-scores.jsonl"]).hexdigest()
    record_ids = tuple(row["record"]["record_id"] for row in rows)
    for index, (name, member) in enumerate(zip(names, members, strict=True)):
        column = _column(rows, index)
        validate_column(column, index, len(rows))
        require(
            contents[name] == column_bytes(primary_hash, record_ids, member, column)
        )


def _completion(
    contents: dict[str, bytes], reservation_sha256: str, count: int
) -> None:
    primary, secondary_counts = expected_counts(count)
    preceding = {
        name: content for name, content in contents.items() if name != "completion.json"
    }
    require(
        contents["completion.json"]
        == completion_bytes(preceding, reservation_sha256, primary, secondary_counts)
    )


def _verify(
    contents: dict[str, bytes],
    private_outputs: dict[str, bytes],
    identity: dict,
    reservation_sha256: str,
    source_checkpoint_sha256: dict[str, str],
) -> None:
    _inventory(contents, SCIENTIFIC_CHECKPOINT_NAMES)
    _inventory(private_outputs, SCIENTIFIC_OUTPUT_NAMES)
    require(
        all(contents[name] == private_outputs[name] for name in SCIENTIFIC_OUTPUT_NAMES)
    )
    rows = tuple(loads(line) for line in contents["predictions.jsonl"].splitlines(True))
    record_ids = tuple(row["record"]["record_id"] for row in rows)
    require(
        contents["context.json"]
        == context_bytes(
            identity,
            reservation_sha256,
            source_checkpoint_sha256,
            contents["bindings.json"],
            record_ids,
        )
    )
    binding = loads(contents["bindings.json"])
    _primary(rows, binding, contents)
    _columns(rows, binding, contents)
    _completion(contents, reservation_sha256, len(rows))


def verify_scientific_checkpoints(
    contents: dict[str, bytes],
    private_outputs: dict[str, bytes],
    *,
    identity: dict,
    reservation_sha256: str,
    source_checkpoint_sha256: dict[str, str],
) -> None:
    """Check linkage and score projections only; scientific replay remains separate."""
    try:
        _verify(
            contents,
            private_outputs,
            identity,
            reservation_sha256,
            source_checkpoint_sha256,
        )
    except Exception:
        raise ScientificCheckpointVerificationError(
            "invalid_scientific_checkpoints"
        ) from None
