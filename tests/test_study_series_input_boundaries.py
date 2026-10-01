"""Separate computational records reuse numerical contracts, not live authority."""

import builtins
from dataclasses import replace
from pathlib import Path

import pytest
from study_series_input_fixtures import (
    api,
    candidates,
    manifests,
    metadata,
    restore,
    series_case,
)

from automated_phishing_detection import (
    _operational_cell_records,
    bound_models,
    bound_runtime,
    operational_cell_inputs,
)
from automated_phishing_detection import operational_role_records as roles
from automated_phishing_detection import operational_runtime as runtime
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.external_source_handoff import (
    ObservedExternalCompletion,
)
from automated_phishing_detection.internal_process_handoff import (
    ObservedInternalCompletion,
)
from automated_phishing_detection.operational_inputs import AcceptedOperationalInputs

__all__ = ["candidates", "manifests", "series_case"]


def test_adapter_uses_no_io_observed_types_or_legacy_restoration(
    series_case, monkeypatch
):
    api()

    def forbidden(*args, **kwargs):
        pytest.fail("adapter attempted IO, observed ownership or legacy restoration")

    for target, name in (
        (builtins, "open"),
        (Path, "open"),
        (Path, "read_bytes"),
        (bound_models, "load_bound_models"),
        (bound_runtime, "open_bound_session"),
        (runtime, "open_bound_session"),
        (ObservedInternalCompletion, "__init__"),
        (ObservedExternalCompletion, "__init__"),
        (AcceptedOperationalInputs, "__init__"),
        (operational_cell_inputs, "restore_cell_inputs"),
        (_operational_cell_records, "authenticate"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    assert metadata(series_case)
    assert restore(series_case).authorizes_execution is False


@pytest.mark.parametrize("ordinal", (73, 121))
def test_unchanged_role_and_runtime_validation_accept_computational_inputs(
    series_case, ordinal
):
    inputs = restore(series_case, ordinal).computational
    context = roles.OperationalRoleContext(
        123, ("invented-service",), "http://127.0.0.1:8765"
    )
    service = roles.build_service_role(inputs, context, primary=inputs.primary)
    roles.verify_role_record(service, inputs=inputs, context=context, role="service")
    client = roles.build_client_role(inputs, context)
    roles.verify_role_record(client, inputs=inputs, context=context, role="client")
    current = replace(
        series_case.original.binding, revision=inputs.execution["revision"]
    )
    artifacts = ArtifactPaths(
        *(Path("/invented") / name for name in ArtifactPaths.__dataclass_fields__)
    )
    runtime._validate(current, artifacts, inputs, context, lambda *unused: None)


def test_missing_original_snapshot_member_cannot_be_ignored(series_case):
    changed = type(series_case)(**vars(series_case))
    changed.internal = replace(
        series_case.internal, payloads=series_case.internal.payloads[1:]
    )
    with pytest.raises(api().SeriesInputError):
        metadata(changed)


def test_fresh_views_do_not_rewrite_original_or_current_metadata(series_case):
    result = restore(series_case)
    expected = result.computational.primary
    result.computational.primary.clear()
    result.computational.execution.clear()
    assert result.computational.primary == expected
    assert result.origin_metadata_bytes == series_case.origin_bytes
