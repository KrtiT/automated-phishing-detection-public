"""Independent specification probes for both-buffer authentication ordering."""

import pytest
from study_series_child_inputs_fixtures import (
    api,
    candidates,
    child_case,
    manifests,
    restored,
    series_case,
)

from automated_phishing_detection import _operational_input_schema as schema

__all__ = ["candidates", "child_case", "manifests", "series_case"]


@pytest.mark.parametrize("field", ["metadata_bytes", "profile_bytes"])
def test_both_context_pins_precede_any_json_parse(child_case, monkeypatch, field):
    def forbidden(*arguments, **keywords):
        pytest.fail("JSON parsing preceded authentication of both context buffers")

    monkeypatch.setattr(schema, "loads", forbidden)
    with pytest.raises(api().SeriesChildInputError):
        restored(child_case, **{field: b"invented unpinned context"})


def test_restoration_interrupt_retains_exact_object(child_case, monkeypatch):
    interruption = KeyboardInterrupt("invented parse interruption")

    def interrupt(*arguments, **keywords):
        raise interruption

    monkeypatch.setattr(schema, "loads", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        restored(child_case)
    assert caught.value is interruption
