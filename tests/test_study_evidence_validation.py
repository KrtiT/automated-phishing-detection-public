"""Malformed supplied evidence rejects symbolically rather than becoming absent."""

from dataclasses import replace

import pytest
from study_evidence_fixtures import evidence
from study_evidence_fixtures import study as study

from automated_phishing_detection.hypothesis_evaluation import SavedPopulation
from automated_phishing_detection.paired_evaluation import BinaryPrediction


@pytest.mark.parametrize("field", ["internal", "external", "reference", "http"])
def test_untyped_supplied_components_reject(study, field):
    with pytest.raises(study.StudyEvidenceError):
        study.reduce_study_evidence(**{field: {"private-sensitive": True}})


@pytest.mark.parametrize(
    "mutation", ["misaligned", "duplicate", "boolean", "unknown_model", "no_models"]
)
def test_invalid_internal_predictions_are_not_dropped(study, mutation):
    supplied = evidence()
    internal = supplied["internal"]
    predictions = dict(internal.predictions)
    column = predictions["transformer"]
    if mutation == "misaligned":
        predictions["transformer"] = (
            BinaryPrediction("private-sensitive", 1),
            *column[1:],
        )
    elif mutation == "duplicate":
        predictions["transformer"] = (column[0], *column[:-1])
    elif mutation == "boolean":
        predictions["transformer"] = (replace(column[0], decision=True), *column[1:])
    elif mutation == "unknown_model":
        predictions["private-sensitive-model"] = column
    else:
        predictions = {}
    supplied["internal"] = SavedPopulation(internal.records, predictions)
    with pytest.raises(study.StudyEvidenceError) as failure:
        study.reduce_study_evidence(**supplied)
    assert "private-sensitive" not in str(failure.value)


def test_overlapping_external_control_identity_is_rejected(study):
    supplied = evidence()
    external = supplied["external"]
    ids = (
        external.populations["gold"].records[0].record_id,
        *external.controls.record_ids[1:],
    )
    controls = replace(
        external.controls,
        record_ids=ids,
        predictions={
            model: tuple(
                BinaryPrediction(identity, prediction.decision)
                for identity, prediction in zip(ids, column, strict=True)
            )
            for model, column in external.controls.predictions.items()
        },
    )
    supplied["external"] = replace(external, controls=controls)
    with pytest.raises(study.StudyEvidenceError):
        study.reduce_study_evidence(**supplied)


@pytest.mark.parametrize("field", ["populations", "controls", "external_windows"])
def test_malformed_external_components_do_not_become_missing(study, field):
    supplied = evidence()
    supplied["external"] = replace(supplied["external"], **{field: None})
    with pytest.raises(study.StudyEvidenceError):
        study.reduce_study_evidence(**supplied)


@pytest.mark.parametrize(
    "mutation", ["wrong_label", "secondary", "misaligned", "extra_control"]
)
def test_invalid_external_role_or_column_is_rejected(study, mutation):
    supplied = evidence()
    external = supplied["external"]
    gold = external.populations["gold"]
    if mutation == "extra_control":
        controls = replace(
            external.controls,
            predictions=external.controls.predictions
            | {"policy": external.controls.predictions["cascade"]},
        )
        external = replace(external, controls=controls)
    elif mutation == "secondary":
        external = replace(
            external, populations=external.populations | {"secondary": gold}
        )
    else:
        if mutation == "wrong_label":
            gold = replace(
                gold, records=(replace(gold.records[0], label=0), *gold.records[1:])
            )
        else:
            gold = replace(
                gold,
                predictions=gold.predictions
                | {"policy": gold.predictions["policy"][:-1]},
            )
        external = replace(external, populations=external.populations | {"gold": gold})
    with pytest.raises(study.StudyEvidenceError):
        study.reduce_study_evidence(**(supplied | {"external": external}))
