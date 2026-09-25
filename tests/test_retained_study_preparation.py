import importlib.util
from dataclasses import FrozenInstanceError

import pytest
from phishvn_source_fixtures import record
from retained_study_preparation_fixtures import api, make_case, restore, retained_case

__all__ = ["api", "retained_case"]


def test_restore_api_exists():
    assert importlib.util.find_spec(
        "automated_phishing_detection.retained_study_preparation"
    ), "missing retained preparation restoration"


def test_original_bytes_and_typed_views_roundtrip(api, retained_case):
    restored = restore(api, retained_case)
    assert type(restored) is api.RestoredStudyPreparation
    assert restored.payloads == retained_case.snapshot.payloads
    assert restored.reservation_sha256 == retained_case.reservation
    assert restored.completion_sha256 == retained_case.completion
    assert restored.internal == retained_case.internal
    assert restored.external == retained_case.external
    assert restored.publisher == retained_case.publisher
    assert restored.reconstructed_internal == retained_case.reconstructed
    assert restored.execution == retained_case.identity
    assert restored.overlap_domains == retained_case.reconstructed.overlap_domains
    assert "invalid-label.example" in restored.overlap_domains
    assert "conflict.example" in restored.overlap_domains
    assert restored.feasibility["shortages"]
    assert restored.scoring_source == {
        "source_interface": "retained_study_preparation_v1",
        "study_preparation_reservation_sha256": retained_case.reservation,
        "study_preparation_complete_sha256": retained_case.completion,
    }


def test_result_is_immutable_and_missing_payload_is_not_fabricated(api, retained_case):
    restored = restore(api, retained_case)
    with pytest.raises(FrozenInstanceError):
        restored.payloads = ()
    with pytest.raises(KeyError):
        restored.payload("not-retained")
    assert (
        restored.payload("source-reconstruction.json")
        is dict(restored.payloads)["source-reconstruction.json"]
    )


@pytest.mark.parametrize("name", ["execution", "feasibility", "scoring_source"])
def test_json_views_are_fresh(api, retained_case, name):
    restored = restore(api, retained_case)
    expected = getattr(restored, name)
    getattr(restored, name).clear()
    assert getattr(restored, name) == expected


@pytest.mark.parametrize("name", ["external", "publisher", "reconstructed_internal"])
def test_nested_typed_views_are_fresh(api, retained_case, name):
    restored = restore(api, retained_case)
    expected = getattr(restored, name)
    changed = getattr(restored, name)
    changed.public_summary.clear()
    changed.private_outputs.clear()
    if name == "publisher":
        changed.published_split_counts.clear()
    assert getattr(restored, name) == expected
    assert restored.payloads == retained_case.snapshot.payloads


def test_empty_external_retention_remains_empty(api):
    case = make_case([record("invalid", url="bare.example")])
    restored = restore(api, case)
    assert restored.external.retained == ()
    assert len(restored.external.quarantine) == 1
    assert restored.feasibility["counts"]["external"]["retained_test_rows"] == 0
