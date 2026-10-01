"""Temporal order witnesses never represent continuous physical telemetry."""

import importlib.util
from dataclasses import FrozenInstanceError

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_timeline_fixtures import (
    make_timeline,
    refresh_timeline,
    verify_timeline,
)
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


def test_stopped_timeline_api_exists():
    assert (
        importlib.util.find_spec("automated_phishing_detection.stopped_study_timeline")
        is not None
    )


def test_next_child_before_violation_is_order_witness(prepared, manifests):
    case = make_timeline(prepared, manifests)
    result = verify_timeline(case)
    assert result.accepted_ordinals == (1,)
    assert result.stopped_ordinal == 2
    assert result.next_service_sample_started_at == "2026-01-01T00:00:30+00:00"
    assert result.prefix_completed_before_sample_at == "2026-01-01T00:00:40+00:00"
    assert result.last_clean_sample_started_at == "2026-01-01T00:00:40+00:00"
    assert result.first_ac_absence_sample_started_at == "2026-01-01T00:00:45+00:00"
    assert result.sample_count == 4
    assert result.maximum_sample_start_gap_seconds == 28
    assert result.observation_sha256 == tuple(sorted(case.pins.items()))
    assert result.establishes_continuous_power is False
    assert result.samples_are_atomic is False
    assert result.authorizes_execution is False
    with pytest.raises(FrozenInstanceError):
        result.stopped_ordinal = 3


@pytest.mark.parametrize("replacement", ["903 1", "904 123"])
def test_next_child_wrong_parent_or_pid_rejected(prepared, manifests, replacement):
    case = make_timeline(prepared, manifests)
    for sample in case.values["conditions.jsonl"][1:3]:
        sample["processes"] = "\n".join(sample["processes"].splitlines()[:-1])
        sample["processes"] += f"\n{replacement} 0.0 0.1 python\n"
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)
