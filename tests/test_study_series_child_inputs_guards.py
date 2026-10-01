"""Child byte restoration is neither historical reconstruction nor live authority."""

import json
from dataclasses import replace

import pytest
from study_series_child_inputs_fixtures import (
    api,
    candidates,
    child_case,
    digest,
    manifests,
    restored,
    series_case,
)
from test_study_history_external_boundaries import forbid_live

from automated_phishing_detection import (
    _operational_cell_records as records,
)
from automated_phishing_detection import (
    _operational_input_schema as schema,
)
from automated_phishing_detection import (
    _saved_external_inputs as saved,
)
from automated_phishing_detection import (
    _study_history_cell_inputs as history,
)
from automated_phishing_detection import (
    _study_series_input_cells as cells,
)
from automated_phishing_detection import (
    _study_series_input_context as context,
)
from automated_phishing_detection import (
    bound_models,
    bound_runtime,
    operational_cell_inputs,
)
from automated_phishing_detection import (
    replay_manifest_codec as replay,
)
from automated_phishing_detection import (
    study_series_inputs as parent,
)
from automated_phishing_detection.external_source_handoff import (
    ObservedExternalCompletion,
    VerifiedExternalSnapshot,
)
from automated_phishing_detection.internal_process_handoff import (
    ObservedInternalCompletion,
)
from automated_phishing_detection.internal_source_handoff import (
    VerifiedInternalSnapshot,
)
from automated_phishing_detection.operational_inputs import AcceptedOperationalInputs

__all__ = ["candidates", "child_case", "manifests", "series_case"]


def forbidden(*args, **kwargs):
    pytest.fail("byte adapter attempted parsing before pins, science or fake authority")


def test_no_io_models_history_reconstruction_or_live_wrapper(child_case, monkeypatch):
    api()
    forbid_live(monkeypatch)
    targets = (
        (history, "sources"),
        (history, "_internal_manifest"),
        (history, "_external_manifest"),
        (cells, "selection"),
        (context, "_origin"),
        (context, "authenticate"),
        (context, "build"),
        (parent, "restore_series_cell_inputs"),
        (records, "authenticate"),
        (operational_cell_inputs, "restore_cell_inputs"),
        (bound_models, "load_bound_models"),
        (bound_runtime, "open_bound_session"),
    )
    for target, name in targets:
        monkeypatch.setattr(target, name, forbidden)
    for target in (
        VerifiedInternalSnapshot,
        VerifiedExternalSnapshot,
        ObservedInternalCompletion,
        ObservedExternalCompletion,
        AcceptedOperationalInputs,
    ):
        monkeypatch.setattr(target, "__init__", forbidden)
    assert restored(child_case) == child_case.expected


@pytest.mark.parametrize(
    "field",
    (
        "metadata_bytes",
        "profile_bytes",
        "descriptor_bytes",
        "binding_bytes",
    ),
)
def test_changed_buffer_is_never_passed_to_json_parser(child_case, monkeypatch, field):
    original = schema.loads
    changed = b"private invented unpinned byte marker"

    def guarded(content, **keywords):
        if content == changed:
            pytest.fail("unpinned buffer reached JSON parsing")
        return original(content, **keywords)

    monkeypatch.setattr(schema, "loads", guarded)
    with pytest.raises(api().SeriesChildInputError) as caught:
        restored(child_case, **{field: changed})
    assert caught.value.__suppress_context__ is True
    assert str(caught.value) == "invalid_series_child_inputs"


def test_changed_manifest_is_never_passed_to_rows_parser(child_case, monkeypatch):
    monkeypatch.setattr(replay, "_decoded", forbidden)
    monkeypatch.setattr(saved, "_rows", forbidden)
    with pytest.raises(api().SeriesChildInputError):
        restored(child_case, manifest_bytes=b"private invented unpinned manifest")


def test_existing_same_parent_routes_remain_closed(child_case):
    result = restored(child_case)
    with pytest.raises(schema.OperationalInputError):
        schema.validate_metadata(json.loads(result.computational.accepted_bytes))
    with pytest.raises(schema.OperationalInputError):
        operational_cell_inputs.restore_cell_inputs(
            child_case.metadata,
            child_case.descriptor,
            child_case.binding,
            child_case.manifest,
            expected_binding_sha256=child_case.frame.cell_binding_sha256,
            expected_cell_reservation_sha256="3" * 64,
        )


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("role", "internal"),
        ("segment_ordinal", True),
        ("cell_ordinal", True),
        ("cell_ordinal", 126),
        ("parent_pid", False),
        ("envelope_sha256", "0" * 63),
    ),
)
def test_mutated_frozen_frame_is_revalidated(child_case, field, replacement):
    frame = replace(child_case.frame)
    object.__setattr__(frame, field, replacement)
    with pytest.raises(api().SeriesChildInputError):
        restored(child_case, frame=frame)


def test_canonical_pins_do_not_permit_noncanonical_metadata(child_case):
    content = b" " + child_case.metadata
    frame = replace(child_case.frame, accepted_inputs_sha256=digest(content))
    with pytest.raises(api().SeriesChildInputError):
        restored(child_case, metadata_bytes=content, frame=frame)


def test_actual_parent_authentication_is_not_claimed_by_pure_frame(child_case):
    frame = replace(child_case.frame, role="client", parent_pid=999999)
    result = restored(child_case, frame=frame)
    assert result.authorizes_execution is False
    primary, execution = result.computational.primary, result.computational.execution
    primary.clear()
    execution.clear()
    assert result == child_case.expected
    assert result.computational.primary and result.computational.execution
