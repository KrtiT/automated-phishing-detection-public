"""A separate wire identity cannot enter legacy study admissions."""

import json
from dataclasses import FrozenInstanceError, asdict

import pytest
from study_admission_fixtures import frame as original_frame
from study_series_admission_fixtures import api, frame

from automated_phishing_detection._study_admission_frame import decode_admission_frame


def test_exact_series_frame_roundtrips_without_authority():
    value = frame()
    assert api().decode_series_admission(value.canonical_bytes) == value
    assert len(value.canonical_bytes) < 4096
    assert not hasattr(value, "authorizes_execution")
    with pytest.raises(FrozenInstanceError):
        value.role = "internal"


@pytest.mark.parametrize("role", ["internal", "external", "", None, [], 1])
def test_only_fresh_service_and_client_roles_exist(role):
    with pytest.raises(ValueError):
        frame(role=role)


@pytest.mark.parametrize(
    "name,value",
    [
        ("segment_ordinal", 1),
        ("segment_ordinal", 3),
        ("segment_ordinal", True),
        ("cell_ordinal", 1),
        ("cell_ordinal", 126),
        ("cell_ordinal", True),
        ("cell_ordinal", 73.0),
        ("parent_pid", True),
        ("parent_pid", 0),
    ],
)
def test_closed_integer_ranges(name, value):
    with pytest.raises(ValueError):
        frame(**{name: value})


@pytest.mark.parametrize("value", [None, "A" * 64, "0" * 63, True, [], 0])
def test_every_digest_has_a_nonnullable_lowercase_pin(value):
    fields = [name for name in asdict(frame()) if name.endswith("_sha256")]
    for name in fields:
        with pytest.raises(ValueError):
            frame(**{name: value})


def test_protocols_reject_one_another():
    with pytest.raises(ValueError):
        decode_admission_frame(frame().canonical_bytes)
    with pytest.raises(ValueError):
        api().decode_series_admission(original_frame().canonical_bytes)


@pytest.mark.parametrize(
    "change", ["space", "newline", "duplicate", "extra", "missing"]
)
def test_canonical_closed_wire(change):
    content = frame().canonical_bytes
    if change in ("space", "newline"):
        content += b" " if change == "space" else b"\n"
    elif change == "duplicate":
        content = b'{"role":"client",' + content[1:]
    else:
        value = json.loads(content)
        if change == "extra":
            value["authorizes_execution"] = True
        else:
            value.pop("history_index_sha256")
        content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        api().decode_series_admission(content)
