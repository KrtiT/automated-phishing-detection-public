import json

import pytest
from study_series_prefix_fixtures import (
    api,
    candidates,
    frame,
    imported,
    manifests,
    prefix_case,
    series_case,
)

__all__ = ["candidates", "manifests", "prefix_case", "series_case"]


def test_required_pure_prefix_api_exists():
    module = api()
    for name in (
        "series_identity",
        "segment_identity",
        "history_import_bytes",
        "segment_intent_bytes",
        "validate_series_child_prefix",
    ):
        assert callable(getattr(module, name))


@pytest.mark.parametrize("role", ("service", "client"))
def test_exact_four_file_prefix_joins_new_and_original_declarations(prefix_case, role):
    case = prefix_case
    assert (
        api().validate_series_child_prefix(
            case.binding, frame(case, role=role), case.payloads
        )
        is None
    )
    assert imported(case) == case.imported
    value = json.loads(case.imported)
    assert value["imported_prefix_length"] == 72
    assert value["origin_reservation_sha256"] != value["segment_reservation_sha256"]
    assert case.binding.authorizes_execution is False


def test_identity_views_are_fresh_and_do_not_contain_future_pins(prefix_case):
    case = prefix_case
    first = api().series_identity(case.binding)
    expected = api().series_identity(case.binding)
    first["execution"].clear()
    assert api().series_identity(case.binding) == expected
    assert "segment_reservation_sha256" not in expected
    segment = api().segment_identity(case.binding, case.series)
    assert "accepted_inputs_sha256" not in segment
    assert segment["series_reservation_sha256"] == case.series.reservation_sha256
