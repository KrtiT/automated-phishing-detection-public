"""Freshly bound substitutions still cannot select a different original manifest."""

import json
from dataclasses import asdict

import pytest
from study_series_adoption_fixtures import digest
from study_series_input_fixtures import (
    api,
    candidates,
    descriptor,
    manifests,
    metadata,
    restore,
    series_case,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.operational_cell_inputs import bind_cell_descriptor
from automated_phishing_detection.operational_schedule import cell_for_ordinal

__all__ = ["candidates", "manifests", "series_case"]


def rebound(case, described, manifest_bytes, ordinal=73):
    content = canonical_bytes(described)
    binding = bind_cell_descriptor(content, cell_reservation_sha256="3" * 64)
    return restore(
        case,
        ordinal,
        descriptor_bytes=content,
        binding_bytes=binding,
        manifest_bytes=manifest_bytes,
        expected_binding_sha256=digest(binding),
    )


@pytest.mark.parametrize(
    "field",
    (
        "root_reservation_sha256",
        "accepted_inputs_sha256",
        "manifest_sha256",
    ),
)
def test_fully_rebound_descriptor_substitutions_reject(series_case, field):
    payloads = descriptor(series_case)
    described = json.loads(payloads.descriptor_bytes)
    described[field] = "0" * 64
    with pytest.raises(api().SeriesInputError):
        rebound(series_case, described, payloads.manifest_bytes)


def test_fully_rebound_valid_schedule_cell_cannot_replay_accepted_prefix(series_case):
    payloads = descriptor(series_case)
    described = json.loads(payloads.descriptor_bytes)
    described["cell"] = asdict(cell_for_ordinal(72))
    with pytest.raises(api().SeriesInputError):
        rebound(series_case, described, payloads.manifest_bytes)


@pytest.mark.parametrize("ordinal", (73, 121))
def test_rehashed_manifest_order_must_still_match_original_selection(
    series_case, ordinal
):
    payloads = descriptor(series_case, ordinal)
    if ordinal == 73:
        value = json.loads(payloads.manifest_bytes)
        value["records"].reverse()
        manifest = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    else:
        manifest = b"".join(reversed(payloads.manifest_bytes.splitlines(keepends=True)))
    assert manifest != payloads.manifest_bytes
    described = json.loads(payloads.descriptor_bytes)
    described["manifest_sha256"] = digest(manifest)
    with pytest.raises(api().SeriesInputError):
        rebound(series_case, described, manifest, ordinal)


@pytest.mark.parametrize("field", ("worker", "protected_evaluation_authorized"))
def test_metadata_cannot_grow_live_authority_fields(series_case, field):
    value = json.loads(metadata(series_case))
    value[field] = True
    with pytest.raises(api().SeriesInputError):
        descriptor(series_case, content=canonical_bytes(value))


def test_original_metadata_requires_exact_canonical_bytes_even_when_repinned(
    series_case,
):
    origin_bytes = b" " + series_case.origin_bytes
    with pytest.raises(api().SeriesInputError):
        api().build_series_input_metadata(
            origin_bytes,
            canonical_bytes(series_case.profile),
            series_case.internal,
            series_case.external,
            expected_origin_sha256=digest(origin_bytes),
            expected_profile_sha256=digest(series_case.profile),
            series_reservation_sha256="1" * 64,
            segment_reservation_sha256="2" * 64,
        )


def test_failure_does_not_echo_private_manifest_bytes(series_case):
    with pytest.raises(api().SeriesInputError) as caught:
        restore(series_case, manifest_bytes=b"private-invented-marker")
    assert str(caught.value) == "invalid_series_operational_inputs"
    assert caught.value.__suppress_context__ is True
