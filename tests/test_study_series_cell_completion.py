import json

import pytest
from study_series_completion_fixtures import holder, module, setup, write_working

from automated_phishing_detection._operational_cell_protocol import (
    SNAPSHOT_NAMES,
    WORKING_NAMES,
)


def test_exact_fresh_held_transition_retains_observation_and_current_expectations(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    with holder(case) as completer:
        assert completer.working is completer.candidate is None
        write_working(case)
        result = completer.complete(**case.options)
        assert result is completer.candidate
        assert set(dict(result.payloads)) == set(SNAPSHOT_NAMES)
        assert (
            tuple(name for name, unused in completer.working.payloads) == WORKING_NAMES
        )
    assert json.loads(case.public.read_bytes()) == {"status": "invented_only"}
    assert [call[0] for call in case.calls] == ["working", "published"]
    assert case.calls[0][2]["attempt"] is case.attempt
    assert case.calls[0][2]["observation"] is case.options["observation"]
    assert case.calls[0][2]["profile_bytes"] == case.options["profile_bytes"]


def test_no_retry_after_completion_or_holder_exit(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with holder(case) as completer:
        write_working(case)
        completer.complete(**case.options)
        with pytest.raises(module().OperationalCellCompletionError):
            completer.complete(**case.options)
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)
    assert len(case.calls) == 2


def test_normal_exit_without_completion_never_publishes(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case):
            pass
    assert not case.public.exists()
