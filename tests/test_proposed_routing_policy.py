from pathlib import Path
import sys

import pytest


BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

import proposed_routing_policy as routing_policy  # noqa: E402
from proposed_routing_policy import (  # noqa: E402
    FIRST_WINDOW_END,
    ROUTING_HORIZON,
    WINDOW_STRIDE,
    drift_override_requests,
)


class IntSubclass(int):
    pass


def test_imports_routing_policy_from_local_code_directory():
    assert Path(routing_policy.__file__).resolve() == (
        CODE_DIR / "proposed_routing_policy.py"
    )


def test_exposes_matrix_v1_1_window_constants():
    assert FIRST_WINDOW_END == 256
    assert WINDOW_STRIDE == 64
    assert ROUTING_HORIZON == 256


def test_routes_one_complete_future_window_after_an_alert():
    routed = drift_override_requests(512, [256])

    assert routed == tuple(range(257, 513))


def test_never_reroutes_the_request_that_produced_an_alert():
    routed = drift_override_requests(520, [256])

    assert 256 not in routed
    assert routed[0] == 257
    assert routed[-1] == 512


def test_unions_overlapping_alert_windows_in_request_order():
    routed = drift_override_requests(600, iter((320, 256)))

    assert routed == tuple(range(257, 577))


def test_truncates_a_terminal_alert_window_at_the_stream_end():
    routed = drift_override_requests(400, [384])

    assert routed == tuple(range(385, 401))


@pytest.mark.parametrize("request_count", [0, 255, 512])
def test_returns_empty_tuple_without_alerts(request_count):
    assert drift_override_requests(request_count, []) == ()


def test_allows_an_alert_on_the_final_request_without_routing_past_it():
    assert drift_override_requests(256, [256]) == ()


def test_rejects_negative_request_count():
    with pytest.raises(ValueError, match="request_count"):
        drift_override_requests(-1, [])


@pytest.mark.parametrize(
    "request_count",
    [
        pytest.param(True, id="boolean"),
        pytest.param(256.0, id="float"),
        pytest.param("256", id="string"),
        pytest.param(IntSubclass(256), id="integer-subclass"),
    ],
)
def test_rejects_request_count_that_is_not_an_exact_integer(request_count):
    with pytest.raises(TypeError, match="request_count"):
        drift_override_requests(request_count, [])


@pytest.mark.parametrize(
    "alerts",
    [
        pytest.param([True], id="boolean"),
        pytest.param([256.0], id="float"),
        pytest.param(["256"], id="string"),
        pytest.param([IntSubclass(256)], id="integer-subclass"),
    ],
)
def test_rejects_alert_that_is_not_an_exact_integer(alerts):
    with pytest.raises(TypeError, match="alert"):
        drift_override_requests(512, alerts)


def test_rejects_noniterable_alerts():
    with pytest.raises(TypeError, match="alert_window_ends"):
        drift_override_requests(512, None)


def test_rejects_duplicate_alerts():
    with pytest.raises(ValueError, match="unique"):
        drift_override_requests(512, [256, 256])


def test_rejects_alert_before_the_first_window_end():
    with pytest.raises(ValueError, match="256"):
        drift_override_requests(512, [192])


def test_rejects_alert_off_the_window_stride():
    with pytest.raises(ValueError, match="stride"):
        drift_override_requests(512, [257])


def test_rejects_alert_beyond_the_request_stream():
    with pytest.raises(ValueError, match="request_count"):
        drift_override_requests(300, [320])
