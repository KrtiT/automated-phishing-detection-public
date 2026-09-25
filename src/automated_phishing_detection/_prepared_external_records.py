"""Closed retained-input joins; supplied records grant no execution authority."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from . import _external_records_validation as validation
from ._checkpoint_codec import canonical_bytes
from .bound_drift import DriftArtifactPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths
from .study_preparation_context import preparation_identity


@dataclass(frozen=True)
class PreparedExternalRunPaths:
    preparation: Path
    artifacts: ArtifactPaths
    secondary_artifacts: SecondaryArtifactPaths
    drift_artifacts: DriftArtifactPaths
    attempt: Path
    public_summary: Path


def outputs_outside_preparation(paths):
    if type(paths) is PreparedExternalRunPaths:
        validation.require(
            not any(
                path.absolute().is_relative_to(paths.preparation.absolute())
                for path in (paths.attempt, paths.public_summary)
            )
        )


def _expected_preparation(binding, profile, internal):
    return preparation_identity(
        binding,
        profile,
        {
            "expected_sha256": internal["partition_sha256"],
            "source_csv_sha256": internal["source_csv_sha256"],
            "suffix_rules_sha256": internal["suffix_rules_sha256"],
        },
    )


def preparation_links(binding, profile, handoff, internal, preparation):
    from .retained_study_preparation import RestoredStudyPreparation

    validation.require(type(preparation) is RestoredStudyPreparation)
    expected = _expected_preparation(binding, profile, internal)
    execution = preparation.execution
    validation.require(canonical_bytes(execution) == canonical_bytes(expected))
    links = preparation.scoring_source
    for name in (
        "study_preparation_reservation_sha256",
        "study_preparation_complete_sha256",
    ):
        validation.digest(links[name])
    joined = links | {
        name: execution[name]
        for name in (
            "source_csv_sha256",
            "partition_sha256",
            "suffix_rules_sha256",
        )
    }
    validation.require(
        canonical_bytes({name: internal[name] for name in joined})
        == canonical_bytes(joined)
    )
    validation.require(
        handoff.overlap_bytes == preparation.payload("source-overlap.json")
    )
    return links


def match_prepared_outputs(outputs, preparation):
    names = (
        "publisher-source.json",
        "publisher-summary.json",
        "suffix-rules.dat",
        "retained-test.jsonl",
        "quarantine.jsonl",
        "inventory.json",
        "preparation-summary.json",
    )
    validation.require(
        all(outputs[name] == preparation.payload(name) for name in names)
    )
    validation.require(
        outputs["internal-source-overlap.json"]
        == preparation.payload("source-overlap.json")
    )
    validation.require(
        sha256(preparation.payload("preparation-complete.json")).hexdigest()
        == preparation.completion_sha256
    )
