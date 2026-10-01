"""Exact required fields, byte forms and immutable facts for invented indices."""

from hashlib import sha256

import pytest
from study_series_history_index_fixtures import (
    api,
    make_case,
    refresh,
    selected,
    validate,
)

from automated_phishing_detection import _study_execution_schema as legacy
from automated_phishing_detection._checkpoint_codec import canonical_bytes

RECORDS = (
    "",
    "attempts.0",
    "attempts.3",
    "original_hold",
    "selected_root",
    "accepted_sources",
    "accepted_cells.0",
    "stopped_cell",
    "attempts.0.profile",
    "attempts.0.envelope",
    "attempts.0.inventory",
    "attempts.0.interruption_review",
    "original_hold.inventory",
    "interruption_review",
)


def required_fields():
    case = make_case()
    return tuple(
        (path, name)
        for path in RECORDS
        for name in (selected(case.index, path) if path else case.index)
    )


@pytest.mark.parametrize(("path", "name"), required_fields())
def test_every_required_field_is_required_even_with_new_pins(path, name):
    case = make_case()
    target = selected(case.index, path) if path else case.index
    target.pop(name)
    refresh(case)
    with pytest.raises(
        api().SeriesHistoryIndexError, match="^invalid_series_history_index$"
    ):
        validate(case)


@pytest.mark.parametrize("path", RECORDS)
def test_every_record_rejects_additional_keys(path):
    case = make_case()
    target = selected(case.index, path) if path else case.index
    target["unreviewed_extension"] = True
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize("value", ("A" * 64, "a" * 63, True, None, [], {}))
def test_file_ref_pins_are_exact_lowercase_hashes(value):
    case = make_case()
    case.index["attempts"][0]["inventory"]["sha256"] = value
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize(
    "encoding", ("duplicate", "nonfinite", "array", "invalid_utf8")
)
def test_malformed_bytes_reject_even_with_their_exact_digest(encoding):
    case = make_case()
    content = canonical_bytes(case.index)
    content = {
        "duplicate": b'{"schema_version":1,' + content[1:],
        "nonfinite": content.replace(b'"schema_version":1', b'"schema_version":NaN'),
        "array": b"[]\n",
        "invalid_utf8": b"\xff",
    }[encoding]
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(
            case, index_bytes=content, expected_index_sha256=sha256(content).hexdigest()
        )


@pytest.mark.parametrize(
    ("kind", "directory"),
    (
        ("internal", "scientific-checkpoints"),
        ("internal", "evidence"),
        ("external", "checkpoints"),
        ("external", "evidence"),
    ),
)
def test_every_original_and_evidence_binding_copy_matches_profile(kind, directory):
    case = make_case()
    member = case.index["accepted_sources"][kind][f"attempt/{directory}/bindings.json"]
    member["sha256"] = "0" * 64
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_new_result_is_not_a_legacy_profile_or_reusable_byte_capability():
    case = make_case()
    result = validate(case)
    with pytest.raises(legacy.StudyExecutionError):
        legacy.profile(result)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case, index_bytes=result)
    assert not hasattr(api(), "bind_study_series_execution")


def test_error_does_not_echo_private_locator_or_parser_text():
    case = make_case()
    case.index["attempts"][0]["inventory"]["path"] = "/private-marker\0"
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError) as caught:
        validate(case)
    assert str(caught.value) == "invalid_series_history_index"
    assert caught.value.__suppress_context__ is True
