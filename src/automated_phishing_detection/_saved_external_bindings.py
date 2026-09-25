"""Closed external byte inventories and already accepted drift identity chains."""

from dataclasses import asdict
from hashlib import sha256

from . import fixed_cascade, saved_evidence
from ._external_preparation_outputs import _PUBLIC_INPUTS
from ._external_secondary_checkpoints import SECONDARY_CHECKPOINTS
from .retained_drift import RetainedDriftReference
from .retained_external_drift import restore_external_drift

PRIVATE_OUTPUTS = frozenset(
    (
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
)
_BINDING_FIELDS = frozenset(
    (
        "schema_version",
        "source_binding",
        "preparation",
        "artifact_hashes",
        "thresholds",
        "secondary",
        "gmm_audit",
        "replay_artifacts",
        "drift",
    )
)
_DRIFT_OUTPUTS = (
    *(target for unused, target in _PUBLIC_INPUTS),
    "training-reference.json",
    "validation-audit.json",
)


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("invalid_saved_external_binding")


def snapshot_outputs(
    private_outputs: dict[str, bytes], public_summary: bytes
) -> tuple[dict[str, bytes], dict]:
    _require(type(private_outputs) is dict and type(public_summary) is bytes)
    _require(all(type(name) is str for name in private_outputs))
    snapshot = private_outputs.copy()
    _require(set(snapshot) == PRIVATE_OUTPUTS)
    _require(all(type(content) is bytes for content in snapshot.values()))
    summary = saved_evidence._loads(public_summary, "external_summary")
    hashes = {name: sha256(content).hexdigest() for name, content in snapshot.items()}
    _require(
        type(summary) is dict
        and fixed_cascade._matches_exactly(summary.get("private_sha256"), hashes)
    )
    return snapshot, summary


def restore_bindings(outputs: dict[str, bytes]) -> dict:
    binding = saved_evidence._loads(outputs["bindings.json"], "external_bindings")
    _require(
        type(binding) is dict
        and set(binding) == _BINDING_FIELDS
        and type(binding["schema_version"]) is int
        and binding["schema_version"] == 1
        and binding["source_binding"] == "caller_supplied_preparation_only"
    )
    saved_evidence._validate_binding_core(binding)
    preparation = saved_evidence._loads(
        outputs["preparation-summary.json"], "external_preparation"
    )
    _require(fixed_cascade._matches_exactly(binding["preparation"], preparation))
    return binding


def _drift_envelope(outputs: dict[str, bytes], binding: dict) -> None:
    drift = binding["drift"]
    hashes = {name: sha256(outputs[name]).hexdigest() for name in _DRIFT_OUTPUTS}
    _require(
        type(drift) is dict
        and set(drift) == {"pins", "portable_state_sha256", "private_sha256"}
        and fixed_cascade._matches_exactly(drift["private_sha256"], hashes)
        and hashes["drift-accepted-report.json"]
        == binding["secondary"]["accepted_report_sha256"]["tabular"]
    )


def restore_reference(
    outputs: dict[str, bytes], binding: dict
) -> RetainedDriftReference:
    _drift_envelope(outputs, binding)
    public = tuple((name, outputs[target]) for name, target in _PUBLIC_INPUTS)
    restored = restore_external_drift(
        outputs["training-reference.json"], outputs["validation-audit.json"], public
    )
    reference = restored.reference
    _require(
        fixed_cascade._matches_exactly(binding["drift"]["pins"], asdict(reference.pins))
        and binding["drift"]["portable_state_sha256"] == reference.portable_state_sha256
    )
    return reference
