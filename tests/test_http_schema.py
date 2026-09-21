from importlib import import_module

import pytest
from pydantic import ValidationError


@pytest.fixture
def schema():
    return import_module("automated_phishing_detection.http_schema")


def test_request_preserves_raw_url_and_has_no_invented_length_limit(schema):
    raw = "HTTPS://Example.test:443/a%2Fb?x=" + "x" * 10000
    request = schema.ScanRequest(request_id="scan-1", url=raw)
    assert request.model_dump() == {"request_id": "scan-1", "url": raw}
    assert schema.ScanRequest(request_id="1", url=" ").url == " "


@pytest.mark.parametrize(
    "request_id", ["", "a b", "a\nb", "a\tb", "a\x00b", "a\u200bb", 1, True, None]
)
def test_request_id_is_strict_nonempty_printable_without_whitespace(schema, request_id):
    with pytest.raises(ValidationError):
        schema.ScanRequest(request_id=request_id, url="https://example.test")
    with pytest.raises(ValidationError):
        schema.ScanResponse(
            request_id=request_id,
            admission_sequence=1,
            action="allow",
            probability=0.1,
            stage2_invoked=False,
        )


@pytest.mark.parametrize("url", ["", 1, True, None, ["https://example.test"]])
def test_request_url_is_strict_nonempty_string(schema, url):
    with pytest.raises(ValidationError):
        schema.ScanRequest(request_id="scan-1", url=url)


def test_request_forbids_unknown_fields(schema):
    with pytest.raises(ValidationError):
        schema.ScanRequest(request_id="scan-1", url="raw", drift_override=True)


def test_response_has_exact_public_fields(schema):
    response = schema.ScanResponse(
        request_id="scan-1",
        admission_sequence=1,
        action="alert",
        probability=1,
        stage2_invoked=True,
    )
    assert response.model_dump() == {
        "request_id": "scan-1",
        "admission_sequence": 1,
        "action": "alert",
        "probability": 1.0,
        "stage2_invoked": True,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("admission_sequence", True),
        ("admission_sequence", "1"),
        ("admission_sequence", 0),
        ("admission_sequence", 1.0),
        ("probability", True),
        ("probability", "0.1"),
        ("probability", float("nan")),
        ("probability", float("inf")),
        ("probability", -0.1),
        ("probability", 1.1),
        ("stage2_invoked", 1),
        ("stage2_invoked", "false"),
        ("action", "block"),
        ("private_audit", {}),
    ],
)
def test_response_rejects_invalid_or_extra_values(schema, field, value):
    values = {
        "request_id": "scan-1",
        "admission_sequence": 1,
        "action": "allow",
        "probability": 0.1,
        "stage2_invoked": False,
        field: value,
    }
    with pytest.raises(ValidationError):
        schema.ScanResponse(**values)


def test_drain_has_only_strict_nonnegative_counters(schema):
    values = {
        "admitted_requests": 3,
        "completed_requests": 2,
        "failed_requests": 1,
        "transformer_forward_attempts": 2,
        "successful_transformer_scores": 1,
    }
    assert schema.DrainResponse(**values).model_dump() == values
    for field in values:
        for value in (-1, True, "1", 1.0):
            with pytest.raises(ValidationError):
                schema.DrainResponse(**(values | {field: value}))
    with pytest.raises(ValidationError):
        schema.DrainResponse(**values, stage1_audit={})


def test_drain_request_has_unique_strict_phase_ids(schema):
    assert schema.DrainRequest().request_ids == []
    assert schema.DrainRequest(request_ids=["warmup-1", "warmup-2"]).request_ids == [
        "warmup-1",
        "warmup-2",
    ]
    for values in (
        ["duplicate", "duplicate"],
        ["bad id"],
        [1],
        "request",
        None,
        ("id",),
    ):
        with pytest.raises(ValidationError):
            schema.DrainRequest(request_ids=values)
    with pytest.raises(ValidationError):
        schema.DrainRequest(extra=True)
