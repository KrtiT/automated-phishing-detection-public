"""Closed saved records reject defaults, coercion, and noncanonical input."""

import pytest
from http_run_codec_fixtures import api, decode, document, wire_bytes

from automated_phishing_detection._checkpoint_codec import canonical_bytes

RUN_FIELDS = (
    "manifest_sha256",
    "prevalence_basis_points",
    "concurrency",
    "run_index",
    "warmup",
    "measured",
    "initial",
    "after_warmup",
    "after_measured",
    "workload",
    "measured_elapsed_ms",
    "measured_drain_ms",
)
OUTCOME_FIELDS = (
    "record_id",
    "request_id",
    "elapsed_ms",
    "status_code",
    "error",
    "response",
)
RESPONSE_FIELDS = (
    "request_id",
    "admission_sequence",
    "action",
    "probability",
    "stage2_invoked",
)
DRAIN_FIELDS = (
    "admitted_requests",
    "completed_requests",
    "failed_requests",
    "transformer_forward_attempts",
    "successful_transformer_scores",
)


@pytest.mark.parametrize("field", RUN_FIELDS)
def test_every_run_field_is_required(field):
    value = document()
    del value["run"][field]
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize("field", OUTCOME_FIELDS)
def test_every_outcome_field_is_required_even_nullable_ones(field):
    value = document()
    del value["run"]["measured"][0][field]
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize("field", RESPONSE_FIELDS)
def test_every_response_field_is_required(field):
    value = document()
    del value["run"]["warmup"][0]["response"][field]
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize("field", DRAIN_FIELDS)
def test_every_counter_is_required(field):
    value = document()
    del value["run"]["initial"][field]
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize("level", ["envelope", "run", "outcome", "response", "drain"])
def test_extra_fields_are_not_ignored(level):
    value = document()
    nested = {
        "envelope": value,
        "run": value["run"],
        "outcome": value["run"]["warmup"][0],
        "response": value["run"]["warmup"][0]["response"],
        "drain": value["run"]["initial"],
    }
    nested[level]["unexpected"] = "private-example"
    with pytest.raises(api().HttpRunCodecError, match="^invalid_http_run_record$"):
        decode(canonical_bytes(value))


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("protocol", "http-progress-v1"),
        ("run", []),
        ("run", None),
    ],
)
def test_closed_envelope(field, invalid):
    value = document()
    value[field] = invalid
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("prevalence_basis_points", True),
        ("concurrency", 64.0),
        ("run_index", True),
        ("workload", None),
        ("warmup", {}),
        ("measured", None),
        ("initial", []),
        ("measured_elapsed_ms", None),
        ("measured_drain_ms", None),
    ],
)
def test_run_field_types_are_exact(field, invalid):
    value = document()
    value["run"][field] = invalid
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("record_id", 0),
        ("request_id", 0),
        ("elapsed_ms", True),
        ("elapsed_ms", "1"),
        ("status_code", True),
        ("error", []),
        ("response", []),
    ],
)
def test_outcome_field_types_are_exact(field, invalid):
    value = document()
    value["run"]["warmup"][0][field] = invalid
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("admission_sequence", True),
        ("probability", True),
        ("stage2_invoked", 1),
        ("action", 1),
        ("request_id", 1),
    ],
)
def test_response_types_do_not_coerce(field, invalid):
    value = document()
    value["run"]["warmup"][0]["response"][field] = invalid
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize("invalid", [True, 0.0, "0", -1])
def test_counter_types_and_bounds_are_strict(invalid):
    value = document()
    value["run"]["initial"]["admitted_requests"] = invalid
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value[:-1],
        lambda value: b" " + value,
        lambda value: value.replace(
            b'"schema_version":1', b'"schema_version":1,"schema_version":1'
        ),
        lambda value: value.replace(b'"elapsed_ms":1.0', b'"elapsed_ms":NaN', 1),
        lambda value: value.replace(b'"elapsed_ms":1.0', b'"elapsed_ms":1e999', 1),
        lambda value: value.replace(
            b'"elapsed_ms":1.0', b'"elapsed_ms":' + b"9" * 400, 1
        ),
        lambda value: value.replace(b'"elapsed_ms":1.0', b'"elapsed_ms":1.00', 1),
        lambda value: value.decode("ascii"),
        lambda value: bytearray(value),
        lambda value: b"[]\n",
        lambda value: b"\xff",
    ],
)
def test_noncanonical_nonfinite_overflow_and_nonbytes_are_rejected(mutation):
    with pytest.raises(api().HttpRunCodecError, match="^invalid_http_run_record$"):
        decode(mutation(wire_bytes()))
