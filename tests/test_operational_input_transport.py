import os
import stat

import pytest
from operational_input_transport_fixtures import (
    CONTENTS,
    expectations,
    fake_restore,
    module,
    retained,
)


def test_fixed_writers_preserve_exact_private_inventories(tmp_path):
    with retained(tmp_path) as (root, cell):
        assert set(os.listdir(root)) == {"accepted-inputs.json"}
        assert set(os.listdir(cell)) == set(CONTENTS) - {"accepted-inputs.json"}
        for directory in (root, cell):
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
            for path in directory.iterdir():
                assert path.read_bytes() == CONTENTS[path.name]
                assert stat.S_IMODE(path.stat().st_mode) == 0o600
                assert path.stat().st_nlink == 1
    assert all(path.exists() for path in (root, cell))


def test_reader_returns_pure_result_and_exact_four_buffers(tmp_path, monkeypatch):
    calls, restored = fake_restore(monkeypatch)
    with retained(tmp_path) as (root, cell):
        with module().hold_operational_inputs(root, cell, **expectations()) as result:
            assert result is restored
    assert calls == [(CONTENTS, expectations())]


def test_existing_leaf_is_never_reused_or_deleted(tmp_path):
    root = tmp_path.resolve() / "accepted-inputs"
    root.mkdir(mode=0o700)
    marker = root / "partial"
    marker.write_bytes(b"unconfirmed")
    with pytest.raises(module().OperationalInputTransportError):
        with module().retain_operational_root_inputs(root, accepted_inputs=b"new"):
            pytest.fail("existing directory yielded")
    assert marker.read_bytes() == b"unconfirmed"


def test_mutable_siblings_do_not_change_held_leaf_inventories(tmp_path, monkeypatch):
    fake_restore(monkeypatch)
    with retained(tmp_path) as (root, cell):
        with module().hold_operational_inputs(root, cell, **expectations()):
            attempt = tmp_path / "cells"
            attempt.mkdir(mode=0o700)
            (attempt / "process-pair-intent.json").write_bytes(b"invented")


@pytest.mark.parametrize(
    "content", [None, "bytes", bytearray(b"bytes"), memoryview(b"b")]
)
def test_writer_rejects_nonexact_bytes_before_creation(tmp_path, content):
    root = tmp_path.resolve() / "accepted-inputs"
    with pytest.raises(module().OperationalInputTransportError):
        with module().retain_operational_root_inputs(root, accepted_inputs=content):
            pytest.fail("invalid content yielded")
    assert not root.exists()


@pytest.mark.parametrize("field", list(expectations()))
@pytest.mark.parametrize("invalid", [None, 1, True, "A" * 64, "a" * 63, "z" * 64])
def test_expectations_rejected_before_any_path_access(tmp_path, field, invalid):
    expected = {**expectations(), field: invalid}
    with pytest.raises(module().OperationalInputTransportError):
        with module().hold_operational_inputs(
            tmp_path / "missing", tmp_path / "other", **expected
        ):
            pytest.fail("invalid expectation yielded")
