"""Compose unchanged checkpoint and scientific kernels for historical bytes."""

from dataclasses import asdict

from . import _study_run_schema as schema
from . import saved_evidence, source_completion, source_runner
from ._internal_scientific_protocol import (
    SCIENTIFIC_CHECKPOINT_NAMES,
    SCIENTIFIC_OUTPUT_NAMES,
    SOURCE_CHECKPOINT_NAMES,
)
from .evaluation_producer import _manifest_summary
from .internal_scientific_verification import verify_scientific_checkpoints
from .internal_source_handoff import VerifiedInternalSnapshot
from .source_checkpoint_verification import verify_source_checkpoints


def _members(values, directory, names):
    return {name: values[f"attempt/{directory}/{name}"] for name in names}


def _checkpoints(values, execution, source, report):
    identity = {
        name: value for name, value in execution.items() if name != "reservation_sha256"
    }
    reservation = execution["reservation_sha256"]
    public = source_runner._json(values["public-summary.json"])
    private = _members(values, "evidence", SCIENTIFIC_OUTPUT_NAMES)
    source_completion._private_bindings(private["bindings.json"], identity)
    schema.same(source_runner._json(private["secondary.json"]), public["secondary"])
    overlap = verify_source_checkpoints(
        _members(values, "checkpoints", SOURCE_CHECKPOINT_NAMES),
        public,
        source,
        report,
        identity,
        reservation,
        private["predictions.jsonl"],
    )
    verify_scientific_checkpoints(
        _members(values, "scientific-checkpoints", SCIENTIFIC_CHECKPOINT_NAMES),
        private,
        identity=identity,
        reservation_sha256=reservation,
        source_checkpoint_sha256=public["checkpoint_sha256"],
    )
    return public, private, overlap


def _projection(reconstructed):
    return {
        "row_count": reconstructed.row_count,
        "domain_count": reconstructed.domain_count,
        "class_counts": reconstructed.class_counts,
        "offline_inference_counts": asdict(reconstructed.inference_counts),
        "offline_secondary_inference_counts": asdict(
            reconstructed.secondary_inference_counts
        ),
        "manifests": {
            str(prevalence): _manifest_summary(outcome)
            for prevalence, outcome in reconstructed.manifests.items()
        },
        "primary": asdict(reconstructed.primary),
        "secondary": reconstructed.secondary,
    }


def _population(projection, source):
    for name in ("row_count", "domain_count", "class_counts"):
        schema.same(projection[name], source[f"expected_{name}"])


def restore(values, execution, source, report):
    public, private, overlap = _checkpoints(values, execution, source, report)
    reconstructed, population = (
        saved_evidence.reconstruct_internal_evidence_and_population(
            private["predictions.jsonl"],
            private["manifests.json"],
            private["bindings.json"],
            private["routing.json"],
        )
    )
    projection = _projection(reconstructed)
    _population(projection, source)
    schema.same(projection, {name: public[name] for name in projection})
    return VerifiedInternalSnapshot(
        tuple(sorted(values.items())),
        tuple(population.records),
        tuple(
            sorted(
                (name, tuple(column)) for name, column in population.predictions.items()
            )
        ),
        overlap,
        tuple(sorted(reconstructed.manifests.items())),
    )
