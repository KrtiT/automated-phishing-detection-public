"""Pure series reduction never substitutes missing or malformed evidence."""

from importlib import import_module
from importlib.util import find_spec

import pytest


def api():
    name = "automated_phishing_detection.study_series_reduction"
    assert find_spec(name), "missing cross-origin scientific reducer"
    return import_module(name)


def test_cross_origin_reducer_exists_without_legacy_parent_input():
    assert callable(api().reduce_series_science)


@pytest.mark.parametrize("historical,fresh", [((), ()), ([], ()), ((), [])])
def test_missing_matrix_is_symbolic_rejection(historical, fresh):
    module = api()
    with pytest.raises(module.SeriesReductionError, match="^invalid_series_reduction$"):
        module.reduce_series_science(
            historical,
            fresh,
            selected_metadata_bytes=b"{}",
            current_context_bytes=b"{}",
            profile_bytes=b"{}",
            expected_profile_sha256="0" * 64,
            expected_selected_metadata_sha256="0" * 64,
            expected_current_context_sha256="0" * 64,
            internal_snapshot=None,
            external_snapshot=None,
        )
