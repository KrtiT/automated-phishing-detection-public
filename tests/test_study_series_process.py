import json

import pytest
from study_series_admission_fixtures import api, frame
from study_series_process_fixtures import admitted_options, observe
from test_operational_process import assert_reaped, inputs

from automated_phishing_detection import operational_process as original


def test_real_series_pair_retains_writers_through_observed_service_cleanup(tmp_path):
    attempt, options = inputs(tmp_path)
    admissions = {}

    def issue(role, command):
        admissions[role] = api().SeriesParentAdmission(frame(command, role))
        return admissions[role]

    observed = observe(attempt, admitted_options(options), issue)
    progress = json.loads(observed.record)
    assert progress["status"] == "observed"
    assert progress["research_accepted"] is False
    assert set(admissions) == {"service", "client"}
    for role, admission in admissions.items():
        assert admission.pid == progress[role]["pid"]
        assert admission.exit_observed is True
        assert admission.exit_code == 0
        assert admission.pipe.descriptors == set()
    assert_reaped(progress)


@pytest.mark.parametrize("role", ("service", "client"))
@pytest.mark.parametrize("event", ("launched", "observed"))
@pytest.mark.parametrize("failure", (ValueError("ledger failure"), KeyboardInterrupt()))
def test_callback_failure_retains_actual_exits_and_closes_all_writers(
    tmp_path, role, event, failure
):
    attempt, options = inputs(tmp_path)
    admissions = {}

    def fail(*unused):
        raise failure

    def issue(actual_role, command):
        callbacks = {f"on_{event}": fail} if role == actual_role else {}
        admissions[actual_role] = api().SeriesParentAdmission(
            frame(command, actual_role), **callbacks
        )
        return admissions[actual_role]

    expected = original.OperationalProcessError
    if isinstance(failure, KeyboardInterrupt):
        expected = type(failure)
    with pytest.raises(expected) as caught:
        observe(attempt, admitted_options(options), issue)
    if isinstance(failure, KeyboardInterrupt):
        assert caught.value is failure
    progress = json.loads(original.process_progress(caught.value))
    for actual_role, admission in admissions.items():
        assert admission.pid == progress[actual_role]["pid"]
        assert admission.exit_observed is True
        assert admission.pipe.descriptors == set()
    assert_reaped(progress)
