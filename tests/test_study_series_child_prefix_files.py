import errno
import os
import signal

import pytest
from study_series_child_prefix_fixtures import api, hold, setup

NAMES = (
    "series/reservation.json",
    "segment/reservation.json",
    "segment/segment-intent.json",
    "segment/history-import.json",
)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "operation", ("mode", "symlink", "hardlink", "missing", "changed")
)
def test_invalid_selected_prefix_record_rejects_before_pure_validation(
    tmp_path, monkeypatch, name, operation
):
    case = setup(tmp_path, monkeypatch)
    path = tmp_path / name
    if operation == "mode":
        path.chmod(0o644)
    elif operation == "missing":
        path.unlink()
    elif operation == "changed":
        path.write_bytes(b"changed")
    else:
        original = tmp_path / "other"
        path.rename(original)
        if operation == "symlink":
            path.symlink_to(original)
        else:
            path.hardlink_to(original)
    with pytest.raises(api().SeriesChildPrefixError):
        with hold(case):
            pytest.fail("invalid prefix yielded")
    assert case.calls == []


@pytest.mark.parametrize("kind", ("series", "segment"))
def test_root_directory_mode_is_held_even_when_inventory_can_grow(
    tmp_path, monkeypatch, kind
):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(api().SeriesChildPrefixError):
        with hold(case):
            case.paths[kind].chmod(0o755)


@pytest.mark.parametrize("position", (1, 2, 3, 4))
def test_real_sigint_during_each_prefix_file_open_closes_registered_handles(
    tmp_path, monkeypatch, position
):
    case = setup(tmp_path, monkeypatch)
    original_open, descriptors = os.open, []

    def interrupted(name, flags, *args, **kwargs):
        descriptor = original_open(name, flags, *args, **kwargs)
        if name in ("reservation.json", "segment-intent.json", "history-import.json"):
            descriptors.append(descriptor)
            if len(descriptors) == position:
                signal.raise_signal(signal.SIGINT)
        return descriptor

    monkeypatch.setattr(os, "open", interrupted)
    with pytest.raises(KeyboardInterrupt):
        with hold(case):
            pytest.fail("interrupted acquisition yielded")
    assert len(descriptors) == position
    for descriptor in descriptors:
        with pytest.raises(OSError) as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


def test_arbitrary_yielded_exception_remains_original(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    original = RuntimeError("invented caller failure")
    with pytest.raises(RuntimeError) as caught:
        with hold(case):
            raise original
    assert caught.value is original
