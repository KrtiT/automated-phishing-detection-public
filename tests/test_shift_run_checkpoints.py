"""Original phase checkpoints must link to, not impersonate, the completed run."""

import importlib
import importlib.util
import json

import pytest
from shift_run_codec_fixtures import shift_case

from automated_phishing_detection.http_replay import ERRORS


def module():
    name = "automated_phishing_detection.shift_run_checkpoints"
    assert importlib.util.find_spec(name), "missing shift checkpoint linkage"
    return importlib.import_module(name)


def wire(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


@pytest.fixture(scope="module")
def sample():
    return shift_case(error="timeout")


@pytest.mark.parametrize("error", [None, *sorted(ERRORS)])
def test_exact_original_checkpoint_stages_accept_completed_errors(error):
    api = module()
    run, warmup, measured = shift_case(error=error)
    assert api.verify_shift_checkpoints(warmup, measured, run=run) is None
    saved = json.loads(measured)
    assert (
        saved["after_measured"] is saved["trace"] is saved["measured_drain_ms"] is None
    )


@pytest.mark.parametrize("phase", ["warmup", "measured"])
@pytest.mark.parametrize("field", ["after_measured", "trace", "measured_drain_ms"])
def test_later_completion_data_cannot_be_inserted_into_checkpoint(sample, phase, field):
    api = module()
    run, warmup, measured = sample
    saved = json.loads(warmup if phase == "warmup" else measured)
    value = getattr(run, field)
    saved[field] = value.model_dump() if hasattr(value, "model_dump") else value
    contents = (wire(saved), measured) if phase == "warmup" else (warmup, wire(saved))
    with pytest.raises(api.ShiftRunCodecError):
        api.verify_shift_checkpoints(*contents, run=run)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "reverse",
        "duplicate",
        "position",
        "phase",
        "bool_position",
        "duration",
        "duration_bool",
        "counter",
        "failed",
        "forward",
        "warmup_changed",
    ],
)
def test_occurrence_drains_are_exact_ordered_and_physically_linked(sample, change):
    api = module()
    run, warmup, measured = sample
    saved = json.loads(measured)
    drains = saved["occurrence_drains"]
    _mutate_drains(drains, change)
    with pytest.raises(api.ShiftRunCodecError):
        api.verify_shift_checkpoints(warmup, wire(saved), run=run)


def _mutate_drains(drains, change):
    if change == "missing":
        drains.pop()
    elif change in ("extra", "duplicate"):
        drains.append(drains[-1])
    elif change == "reverse":
        drains.reverse()
    elif change in ("position", "phase", "bool_position", "duration", "duration_bool"):
        field, value = {
            "position": ("position", 1),
            "phase": ("phase", "warmup"),
            "bool_position": ("position", False),
            "duration": ("elapsed_ms", 9.0),
            "duration_bool": ("elapsed_ms", True),
        }[change]
        drains[1][field] = value
    else:
        if change == "warmup_changed":
            drains[0]["elapsed_ms"] = 7.0
        else:
            field = {
                "counter": "completed_requests",
                "failed": "failed_requests",
                "forward": "successful_transformer_scores",
            }[change]
            drains[1]["counts"][field] += 1
