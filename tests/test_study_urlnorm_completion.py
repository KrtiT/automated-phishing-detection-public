"""Derived completion versions join independently retained publisher ancestry."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from retained_study_derivation_fixtures import (
    derivation_case,
    inputs,
    preparation_api,
    preparation_case,
    run,
    runner,
)

from automated_phishing_detection import retained_study_preparation as retained
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["derivation_case", "inputs", "preparation_api", "preparation_case", "runner"]


def restore(case, snapshot):
    complete = snapshot.payload("preparation-complete.json")
    return retained.restore_study_preparation(
        snapshot,
        expected_identity=json.loads(complete)["execution"],
        expected_reservation_sha256=snapshot.reservation_sha256,
        expected_completion_sha256=sha256(complete).hexdigest(),
        source_spec_bytes=(case.binding.root / "data/sources.json").read_bytes(),
        preparation_summary_bytes=(
            case.binding.root / "reports/phiusiil-preparation-summary.json"
        ).read_bytes(),
    )


def mutation(snapshot, update):
    complete = json.loads(snapshot.payload("preparation-complete.json"))
    update(complete)
    return replace(
        snapshot,
        payloads=tuple(
            (
                name,
                canonical_bytes(complete)
                if name == "preparation-complete.json"
                else value,
            )
            for name, value in snapshot.payloads
        ),
    )


def test_derived_preparation_restores_without_original_inputs(derivation_case):
    snapshot = run(derivation_case)
    restored = restore(derivation_case, snapshot)
    assert restored.payloads == snapshot.payloads
    assert restored.internal == derivation_case.prior.internal
    assert restored.publisher.public_summary["schema_version"] == 2
    assert restored.execution["protocol"] == "study-preparation-v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("schema_version", True),
        ("schema_version", 3),
        ("protocol", "study-preparation-v1"),
        ("scoring_authorized", True),
        ("protected_evaluation_authorized", True),
    ],
)
def test_mixed_or_authorizing_derived_completion_rejects(derivation_case, field, value):
    snapshot = mutation(
        run(derivation_case), lambda record: record.update({field: value})
    )
    with pytest.raises(retained.StudyPreparationRestoreError):
        restore(derivation_case, snapshot)


@pytest.mark.parametrize(
    "field,value",
    [
        ("representation", "raw_url"),
        ("publisher_source_sha256", "0" * 64),
        ("publisher_summary_sha256", "0" * 64),
        ("diagnostic_sha256", "bad"),
        ("prior_profile", {}),
        ("extra", "0" * 64),
    ],
)
def test_derived_lineage_is_closed_and_matches_both_parents(
    derivation_case, field, value
):
    snapshot = mutation(
        run(derivation_case), lambda record: record["derivation"].update({field: value})
    )
    with pytest.raises(retained.StudyPreparationRestoreError):
        restore(derivation_case, snapshot)


def test_derived_publisher_cannot_hide_under_legacy_completion(derivation_case):
    def downgrade(record):
        record.update(schema_version=1, protocol="study-preparation-v1")
        del record["derivation"]

    snapshot = mutation(run(derivation_case), downgrade)
    with pytest.raises(retained.StudyPreparationRestoreError):
        restore(derivation_case, snapshot)
