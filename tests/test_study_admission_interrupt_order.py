import json
import signal

import pytest
from study_admission_fixtures import frame, module
from test_operational_process import inputs
from test_study_admission_process import observe, options_with_admission

from automated_phishing_detection import operational_process as operational


def issue_callbacks(first, *, earlier_body=False):
    admissions = {}

    def interrupt(*unused):
        raise first

    def later(*unused):
        raise KeyboardInterrupt("later callback")

    def issue(role, command):
        callbacks = {}
        if role == "client":
            callbacks = {"on_observed": interrupt}
            if earlier_body:
                callbacks = {"on_launched": interrupt, "on_observed": later}
        admission = module().ParentAdmission(frame(command, role), **callbacks)
        admissions[role] = admission
        return admission

    return issue, admissions


def assert_retained_children(error, admissions):
    progress = json.loads(operational.process_progress(error))
    for role, admission in admissions.items():
        assert progress[role]["pid"] == admission.pid
        assert progress[role]["exit_observed"] is admission.exit_observed is True
        assert admission.pipe.descriptors == set()


@pytest.mark.parametrize("first", [KeyboardInterrupt("first"), SystemExit(17)])
def test_callback_interrupt_precedes_sigint_at_observation_boundary(
    tmp_path, monkeypatch, first
):
    attempt, options = inputs(tmp_path)
    issue, admissions = issue_callbacks(first)
    original, injected = operational.Observations.fail, []

    def fail(observations, check_id):
        original(observations, check_id)
        if check_id == "parent_interrupted" and not injected:
            injected.append(True)
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(operational.Observations, "fail", fail)
    with pytest.raises(BaseException) as caught:
        observe(attempt, options_with_admission(options), issue)
    assert injected == [True]
    assert caught.value is first
    assert_retained_children(caught.value, admissions)


@pytest.mark.parametrize("first", [KeyboardInterrupt("first"), SystemExit(17)])
def test_body_interrupt_precedes_callback_interrupt_during_cleanup(tmp_path, first):
    attempt, options = inputs(tmp_path)
    issue, admissions = issue_callbacks(first, earlier_body=True)
    with pytest.raises(BaseException) as caught:
        observe(attempt, options_with_admission(options), issue)
    assert caught.value is first
    assert_retained_children(caught.value, admissions)
