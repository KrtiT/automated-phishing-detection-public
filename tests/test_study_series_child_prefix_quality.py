"""Independent real-signal cleanup probes using explicitly opaque invented IO."""

import pytest
from operational_input_signal_fixtures import (
    assert_closed,
    interrupt_fdopen,
    watch_open,
)
from study_series_child_prefix_fixtures import api, hold, setup


@pytest.mark.parametrize(
    "name", ["reservation.json", "segment-intent.json", "history-import.json"]
)
def test_real_sigint_before_selected_file_close_closes_all_owned_handles(
    tmp_path, monkeypatch, name
):
    case = setup(tmp_path, monkeypatch)
    with monkeypatch.context() as guard:
        descriptors = watch_open(guard, name, before_close=True)
        with pytest.raises(KeyboardInterrupt):
            with hold(case) as payloads:
                assert dict(payloads) == case.values
    assert descriptors
    assert_closed(descriptors)


def test_real_sigint_after_fdopen_keeps_owned_descriptor_registered(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    with monkeypatch.context() as guard:
        descriptors = interrupt_fdopen(guard)
        with pytest.raises(KeyboardInterrupt):
            with hold(case):
                pytest.fail("interrupted read yielded a prefix")
    assert descriptors
    assert_closed(descriptors)
    assert not case.calls


def test_original_system_exit_survives_actual_later_close_sigint(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    original = SystemExit(7)
    with monkeypatch.context() as guard:
        descriptors = watch_open(guard, "history-import.json", before_close=True)
        with pytest.raises(SystemExit) as caught:
            with hold(case):
                raise original
    assert caught.value is original
    assert_closed(descriptors)


def test_pure_prefix_rejection_closes_each_already_opened_file(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)

    def reject(*arguments):
        raise ValueError("invented declaration rejection")

    monkeypatch.setattr(api(), "_validate", reject)
    with monkeypatch.context() as guard:
        descriptors = watch_open(guard, "reservation.json")
        with pytest.raises(api().SeriesChildPrefixError):
            with hold(case):
                pytest.fail("invalid pure prefix yielded")
    assert len(descriptors) == 2
    assert_closed(descriptors)
