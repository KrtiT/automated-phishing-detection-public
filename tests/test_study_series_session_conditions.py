"""The continuation retains the original sampled physical-session rules."""

import pytest
from study_series_session_fixtures import api, inhibitor, observation


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"battery": "Now drawing from 'Battery Power'"}, "ac_power_absent"),
        (
            {"power_settings": "Battery Power:\n powermode 0\n"},
            "ac_energy_settings_unavailable",
        ),
        (
            {"power_settings": "AC Power:\n powermode 1\n"},
            "automatic_energy_mode_not_confirmed",
        ),
        (
            {"power_settings": "AC Power:\n powermode 0\n lowpowermode 1\n"},
            "low_power_enabled",
        ),
        ({"competing_known_workload_pids": [99]}, "observed_competing_workload"),
    ],
)
def test_original_host_gates(changes, expected):
    module = api("_study_series_session_conditions")
    assert module.violation(observation(**changes)) == expected


def test_thermal_observation_does_not_add_a_new_failure_criterion():
    module = api("_study_series_session_conditions")
    assert module.violation(observation()) is None
    assert module.inhibition(observation(), inhibitor()) is None


@pytest.mark.parametrize("selected", [inhibitor(0), inhibitor(1)])
def test_owned_inhibitor_must_remain_alive(selected):
    module = api("_study_series_session_conditions")
    assert module.inhibition(observation(), selected) == "owned_sleep_inhibitor_exited"


def test_assertions_from_another_process_do_not_qualify():
    module = api("_study_series_session_conditions")
    sample = observation(
        assertions=observation()["assertions"].replace("999992", "999990")
    )
    assert (
        module.inhibition(sample, inhibitor()) == "owned_sleep_assertions_unavailable"
    )


def test_competing_detector_excludes_only_owned_descendants(monkeypatch):
    module = api("_study_series_session_conditions")
    monkeypatch.setattr(module.os, "getpid", lambda: 100)
    monkeypatch.setattr(
        module,
        "command",
        lambda *args: "\n".join(
            [
                "100 90 series-session",
                "101 100 python root.py",
                "102 101 python train_invented.py",
                "103 99 python -m pytest",
                "104 99 python train_elsewhere.py",
                "105 99 editor",
            ]
        ),
    )
    assert module.competing() == [103, 104]
