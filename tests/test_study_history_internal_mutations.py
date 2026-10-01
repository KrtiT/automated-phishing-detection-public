"""Rehashed invented histories must still satisfy unchanged scientific checks."""

import json

import pytest
from study_history_internal_fixtures import (
    historical_internal,
    inputs,
    published,
    repin,
    republish,
    republish_private,
    runner,
    verify,
)

__all__ = ["historical_internal", "inputs", "published", "runner"]


@pytest.mark.parametrize("change", ["missing", "extra", "not_dict", "nonbytes"])
def test_snapshot_inventory_is_exact(historical_internal, change):
    case = historical_internal
    if change == "missing":
        case.payloads.pop("attempt/scientific-checkpoints/completion.json")
    elif change == "extra":
        case.payloads["injected.json"] = b"{}"
    elif change == "not_dict":
        case.payloads = tuple(case.payloads.items())
    else:
        case.payloads["public-summary.json"] = bytearray(
            case.payloads["public-summary.json"]
        )
    with pytest.raises(ValueError, match="^invalid_historical_internal_science$"):
        verify(case)


@pytest.mark.parametrize("field", ["revision", "runtime_sha256", "reservation_sha256"])
def test_independent_original_execution_is_required(historical_internal, field):
    case = historical_internal
    execution = case.execution | {field: "0" * (40 if field == "revision" else 64)}
    with pytest.raises(ValueError):
        verify(case, expected_execution=execution)


@pytest.mark.parametrize(
    "name", ["data/sources.json", "reports/phiusiil-preparation-summary.json"]
)
def test_source_files_require_independent_pins(historical_internal, name):
    case = historical_internal
    with pytest.raises(ValueError):
        verify(case, expected_source_sha256=case.sources | {name: "0" * 64})


def test_original_attempt_path_is_required(historical_internal):
    with pytest.raises(ValueError):
        verify(
            historical_internal, expected_attempt_directory="/invented/other-attempt"
        )


@pytest.mark.parametrize(
    "field",
    [
        "row_count",
        "domain_count",
        "class_counts",
        "offline_inference_counts",
        "offline_secondary_inference_counts",
        "manifests",
        "primary",
        "secondary",
    ],
)
def test_all_public_scientific_fields_match_reconstruction(historical_internal, field):
    case = historical_internal
    public = json.loads(case.payloads["public-summary.json"])
    public[field] = public[field] + 1 if type(public[field]) is int else {}
    republish(case, public)
    with pytest.raises(ValueError):
        verify(case)


def test_rehashed_routing_lie_fails_real_kernel(historical_internal):
    case = historical_internal
    routing = json.loads(case.payloads["attempt/evidence/routing.json"])
    routing["rows"][0]["logical_band"] = not routing["rows"][0]["logical_band"]
    republish_private(case, "routing.json", routing)
    with pytest.raises(ValueError):
        verify(case)


def test_rehashed_manifest_lie_fails_real_kernel(historical_internal):
    case = historical_internal
    manifests = json.loads(case.payloads["attempt/evidence/manifests.json"])
    manifests["10"]["available"] += 1
    republish_private(case, "manifests.json", manifests)
    with pytest.raises(ValueError):
        verify(case)


def test_matching_public_and_private_secondary_lie_fails_reconstruction(
    historical_internal,
):
    republish_private(historical_internal, "secondary.json", {}, public_value={})
    with pytest.raises(ValueError):
        verify(historical_internal)


@pytest.mark.parametrize(
    "member",
    [
        "attempt/checkpoints/source-overlap.json",
        "attempt/scientific-checkpoints/context.json",
    ],
)
def test_changed_original_checkpoint_context_rejects(historical_internal, member):
    case = historical_internal
    case.payloads[member] += b" "
    repin(case)
    with pytest.raises(ValueError):
        verify(case)


def test_republished_valid_metadata_cannot_replace_original_byte_pins(
    historical_internal,
):
    case = historical_internal
    original = case.pins.copy()
    public = json.loads(case.payloads["public-summary.json"])
    public["protected_evaluation_authorized"] = True
    republish(case, public)
    with pytest.raises(ValueError):
        verify(case, expected_snapshot_sha256=original)
