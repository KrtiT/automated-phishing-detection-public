from hashlib import sha256

import pytest
from study_series_child_inputs_fixtures import (
    candidates,
    child_case,
    manifests,
    series_case,
)
from study_series_child_transport_fixtures import api, hold, written

__all__ = ["candidates", "manifests", "series_case", "child_case"]
NAMES = ("accepted-inputs.json", "descriptor.json", "binding.json", "manifest")


def selected(paths, name):
    return paths[0 if name == "accepted-inputs.json" else 1] / name


def mutate(path, operation):
    if operation == "mode":
        path.chmod(0o644)
    elif operation == "missing":
        path.unlink()
    elif operation == "directory":
        path.unlink()
        path.mkdir(mode=0o700)
    elif operation == "bytes":
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    else:
        original = path.parent.parent / "other-original"
        path.rename(original)
        if operation == "symlink":
            path.symlink_to(original)
        else:
            path.hardlink_to(original)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "operation", ("mode", "missing", "directory", "bytes", "symlink", "hardlink")
)
def test_each_unsafe_original_input_is_rejected(tmp_path, child_case, name, operation):
    paths = written(tmp_path, child_case)
    mutate(selected(paths, name), operation)
    with pytest.raises(api().OperationalInputTransportError):
        with hold(paths, child_case):
            pytest.fail("unsafe input yielded")


@pytest.mark.parametrize("name", NAMES)
def test_same_length_mutation_after_yield_rejects_success(tmp_path, child_case, name):
    paths = written(tmp_path, child_case)
    with pytest.raises(api().OperationalInputTransportError):
        with hold(paths, child_case):
            mutate(selected(paths, name), "bytes")


@pytest.mark.parametrize("name", NAMES)
def test_interrupt_identity_survives_mutation_cleanup_failure(
    tmp_path, child_case, name
):
    paths = written(tmp_path, child_case)
    interrupted = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with hold(paths, child_case):
            mutate(selected(paths, name), "bytes")
            raise interrupted
    assert caught.value is interrupted


def test_binding_first_and_each_buffer_read_once(tmp_path, child_case, monkeypatch):
    paths = written(tmp_path, child_case)
    calls = []
    original = api().storage.read_file

    def observed(directory, name, state):
        content = original(directory, name, state)
        calls.append((name, sha256(content).hexdigest()))
        return content

    monkeypatch.setattr(api().storage, "read_file", observed)
    with hold(paths, child_case):
        pass
    assert [name for name, unused in calls] == [
        "binding.json",
        "descriptor.json",
        "accepted-inputs.json",
        "manifest",
    ]
    assert calls[0][1] == child_case.frame.cell_binding_sha256


def test_wrong_binding_stops_before_other_input_reads(
    tmp_path, child_case, monkeypatch
):
    paths = written(tmp_path, child_case)
    mutate(selected(paths, "binding.json"), "bytes")
    original = api().storage.read_file
    calls = []

    def observed(directory, name, state):
        calls.append(name)
        return original(directory, name, state)

    monkeypatch.setattr(api().storage, "read_file", observed)
    with pytest.raises(api().OperationalInputTransportError):
        with hold(paths, child_case):
            pytest.fail("bad binding yielded")
    assert calls == ["binding.json"]


@pytest.mark.parametrize("index", (0, 1))
def test_extra_leaf_member_rejects_but_other_siblings_are_allowed(
    tmp_path, child_case, index
):
    paths = written(tmp_path, child_case)
    extra = paths[index] / "extra"
    extra.write_bytes(b"invented")
    with pytest.raises(api().OperationalInputTransportError):
        with hold(paths, child_case):
            pytest.fail("expanded leaf yielded")
    extra.unlink()
    (tmp_path / "growing-attempt").mkdir(mode=0o700)
    with hold(paths, child_case):
        pass
