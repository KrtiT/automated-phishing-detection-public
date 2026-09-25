from hashlib import sha256

import pytest
from operational_input_transport_fixtures import (
    CONTENTS,
    expectations,
    fake_restore,
    module,
    written,
)


def record_reads(monkeypatch):
    transport = module()
    original, calls = transport.storage.read_file, []

    def read(directory, name, initial):
        calls.append(name)
        return original(directory, name, initial)

    monkeypatch.setattr(transport.storage, "read_file", read)
    return calls


def test_all_four_states_captured_before_any_once_only_read(tmp_path, monkeypatch):
    paths = written(tmp_path)
    transport = module()
    captured, original = [], transport.files.capture
    fake_restore(monkeypatch)

    def capture(directory, name):
        captured.append(name)
        return original(directory, name)

    def read(directory, name, initial):
        assert set(captured) == set(CONTENTS)
        return original_read(directory, name, initial)

    monkeypatch.setattr(transport.files, "capture", capture)
    reads = record_reads(monkeypatch)
    original_read = transport.storage.read_file
    monkeypatch.setattr(transport.storage, "read_file", read)
    with transport.hold_operational_inputs(*paths, **expectations()):
        pass
    assert reads == [
        "binding.json",
        "descriptor.json",
        "accepted-inputs.json",
        "manifest",
    ]


def test_wrong_expected_binding_reads_no_other_content_or_parser(tmp_path, monkeypatch):
    paths = written(tmp_path)
    calls, _ = fake_restore(monkeypatch)
    reads = record_reads(monkeypatch)
    expected = {**expectations(), "expected_binding_sha256": "9" * 64}
    with pytest.raises(module().OperationalInputTransportError):
        with module().hold_operational_inputs(*paths, **expected):
            pytest.fail("wrong independent binding yielded")
    assert reads == ["binding.json"]
    assert not calls


def test_no_json_parsing_in_transport_before_pure_verifier(tmp_path, monkeypatch):
    root, cell = written(tmp_path)
    invalid = b"not-json-or-a-manifest\xff"
    for directory in (root, cell):
        for path in directory.iterdir():
            path.write_bytes(invalid)
    calls, _ = fake_restore(monkeypatch)
    expected = {
        **expectations(),
        "expected_binding_sha256": sha256(invalid).hexdigest(),
    }
    with module().hold_operational_inputs(root, cell, **expected):
        pass
    assert calls == [({name: invalid for name in CONTENTS}, expected)]


def test_cross_file_change_after_capture_rejected_before_restore(tmp_path, monkeypatch):
    root, cell = written(tmp_path)
    transport = module()
    calls, _ = fake_restore(monkeypatch)
    original = transport.storage.read_file

    def read(directory, name, initial):
        result = original(directory, name, initial)
        if name == "binding.json":
            (root / "accepted-inputs.json").write_bytes(b"changed")
        return result

    monkeypatch.setattr(transport.storage, "read_file", read)
    with pytest.raises(transport.OperationalInputTransportError):
        with transport.hold_operational_inputs(root, cell, **expectations()):
            pytest.fail("cross-file mutation yielded")
    assert not calls


def test_change_inside_pure_restore_rejected_before_yield(tmp_path, monkeypatch):
    root, cell = written(tmp_path)

    def restore(contents, expected):
        (cell / "manifest").write_bytes(b"changed during pure call")
        return object()

    monkeypatch.setattr(module(), "_restore", restore)
    with pytest.raises(module().OperationalInputTransportError):
        with module().hold_operational_inputs(root, cell, **expectations()):
            pytest.fail("mutated restored context yielded")
