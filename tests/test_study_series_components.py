"""Current/original public candidate comparisons never create live authority."""

import json
from dataclasses import FrozenInstanceError

import pytest
from study_series_components_fixtures import api, arguments, case
from study_urlnorm_fixtures import digest


def test_full_current_components_project_to_independently_original_pins():
    selected = case()
    result = api().verify_series_component_transition(**arguments(selected))
    assert result.original_external_bytes == selected.before[0]
    assert result.original_operational_bytes == selected.before[1]
    assert result.current_external_bytes == selected.after[0]
    assert result.current_operational_bytes == selected.after[1]
    assert result.profile_sha256 == digest(arguments(selected)["profile_bytes"])
    assert result.authorizes_execution is False
    assert all(
        type(content) is bytes for content in (*selected.before, *selected.after)
    )
    with pytest.raises(FrozenInstanceError):
        result.current_external_bytes = b"changed"


def test_original_audit_pin_is_saved_artifact_digest_not_report_digest():
    selected = case()
    original = selected.profile["scientific_pins"]["original_h2_audit_sha256"]
    assert original != digest(selected.audit)
    assert (
        original
        == json.loads(selected.audit)["artifact_hashes"]["validation-audit.json"]
    )
    assert api().verify_series_component_transition(**arguments(selected))
