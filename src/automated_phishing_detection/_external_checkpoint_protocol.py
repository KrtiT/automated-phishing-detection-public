"""External retention inventory; completion does not imply execution acceptance."""

from hashlib import sha256

from ._external_secondary_checkpoints import SECONDARY_CHECKPOINTS

PROTOCOL = "external-source-checkpoints-v1"
DIRECTORY = "checkpoints"
PROVENANCE_ORDER = (
    "publisher-source.json",
    "publisher-summary.json",
    "suffix-rules.dat",
    "internal-source-overlap.json",
    "internal-source-handoff.json",
    "external-source-reconstruction.json",
)
SCIENTIFIC_ORDER = (
    "retained-test.jsonl",
    "quarantine.jsonl",
    "inventory.json",
    "preparation-summary.json",
    "bindings.json",
    "drift-accepted-report.json",
    "drift-preparation-summary.json",
    "drift-source-spec.json",
    "training-reference.json",
    "validation-audit.json",
    "primary-scores.jsonl",
    "primary-completion.json",
    *SECONDARY_CHECKPOINTS,
    "secondary-completion.json",
    "all-scores.jsonl",
    "routing.json",
    "monitors.json",
    "predictions.jsonl",
    "secondary.json",
)


class ExternalCheckpointError(ValueError):
    """A symbolic checkpoint failure with no private path or producer text."""


def require(condition):
    if not condition:
        raise ExternalCheckpointError("invalid_external_checkpoint")


def hashes(contents):
    return {name: sha256(content).hexdigest() for name, content in contents.items()}


def snapshot_mapping(contents, names):
    require(type(contents) is dict and set(contents) == set(names))
    require(all(type(name) is str for name in contents))
    require(all(type(content) is bytes for content in contents.values()))
    return {name: contents[name] for name in names}
