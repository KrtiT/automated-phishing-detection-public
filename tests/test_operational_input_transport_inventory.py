import os

import pytest
from operational_input_transport_fixtures import (
    CONTENTS,
    expectations,
    fake_restore,
    module,
    written,
)


def damage_file(path, operation):
    if operation == "extra":
        path.with_name("extra").write_bytes(b"extra")
    elif operation.startswith("mode"):
        path.chmod(int(operation[-3:], 8))
    else:
        path.unlink()
        if operation == "directory":
            path.mkdir(mode=0o700)
        elif operation == "fifo":
            os.mkfifo(path, 0o600)
        elif operation in ("symlink", "hardlink"):
            target = path.with_name("target")
            target.write_bytes(CONTENTS[path.name])
            target.chmod(0o600)
            if operation == "symlink":
                path.symlink_to(target)
            else:
                os.link(target, path)


@pytest.mark.parametrize("name", tuple(CONTENTS))
@pytest.mark.parametrize(
    "operation",
    [
        "missing",
        "extra",
        "directory",
        "fifo",
        "symlink",
        "hardlink",
        "mode644",
        "mode400",
    ],
)
def test_reader_rejects_nonclosed_file_inventory(
    tmp_path, monkeypatch, name, operation
):
    root, cell = written(tmp_path)
    calls, _ = fake_restore(monkeypatch)
    damage_file((root if name == "accepted-inputs.json" else cell) / name, operation)
    with pytest.raises(module().OperationalInputTransportError):
        with module().hold_operational_inputs(root, cell, **expectations()):
            pytest.fail("invalid file inventory yielded")
    assert not calls


@pytest.mark.parametrize("selected", [0, 1])
@pytest.mark.parametrize("mode", [0o755, 0o750, 0o500])
def test_reader_rejects_nonprivate_leaf(tmp_path, monkeypatch, selected, mode):
    paths = written(tmp_path)
    fake_restore(monkeypatch)
    paths[selected].chmod(mode)
    with pytest.raises(module().OperationalInputTransportError):
        with module().hold_operational_inputs(*paths, **expectations()):
            pytest.fail("invalid directory mode yielded")


def test_reader_accepts_shared_outer_parent(tmp_path, monkeypatch):
    paths = written(tmp_path)
    fake_restore(monkeypatch)
    tmp_path.chmod(0o755)
    with module().hold_operational_inputs(*paths, **expectations()):
        pass


@pytest.mark.parametrize(
    "operation", ["same", "nested", "symlink", "parent_symlink", "traversal", "string"]
)
def test_reader_rejects_path_aliases(tmp_path, monkeypatch, operation):
    root, cell = written(tmp_path)
    fake_restore(monkeypatch)
    if operation == "same":
        cell = root
    elif operation == "nested":
        cell = root / "nested"
    elif operation == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(cell, target_is_directory=True)
        cell = alias
    elif operation == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        cell = alias / cell.name
    elif operation == "traversal":
        cell = cell / ".." / cell.name
    else:
        cell = str(cell)
    with pytest.raises(module().OperationalInputTransportError):
        with module().hold_operational_inputs(root, cell, **expectations()):
            pytest.fail("aliased path yielded")


@pytest.mark.parametrize("selected", [0, 1])
@pytest.mark.parametrize(
    "operation", ["replace", "extra", "chmod", "same_size", "swap_directory"]
)
def test_post_yield_mutation_rejects_acceptance(
    tmp_path, monkeypatch, selected, operation
):
    paths = written(tmp_path)
    fake_restore(monkeypatch)
    with pytest.raises(module().OperationalInputTransportError):
        with module().hold_operational_inputs(*paths, **expectations()):
            directory = paths[selected]
            name = "accepted-inputs.json" if selected == 0 else "manifest"
            path = directory / name
            if operation == "swap_directory":
                directory.rename(directory.with_name(directory.name + "-old"))
                directory.mkdir(mode=0o700)
            elif operation == "replace":
                path.unlink()
                path.write_bytes(CONTENTS[name])
                path.chmod(0o600)
            elif operation == "same_size":
                path.write_bytes(b"x" * len(CONTENTS[name]))
            elif operation == "chmod":
                directory.chmod(0o755)
            else:
                (directory / "extra").write_bytes(b"partial")


@pytest.mark.parametrize("operation", ["content", "mode", "extra"])
def test_parent_writer_final_check_preserves_rejected_files(tmp_path, operation):
    root = tmp_path.resolve() / "accepted-inputs"
    with pytest.raises(module().OperationalInputTransportError):
        with module().retain_operational_root_inputs(root, accepted_inputs=b"original"):
            if operation == "content":
                (root / "accepted-inputs.json").write_bytes(b"modified")
            elif operation == "mode":
                root.chmod(0o755)
            else:
                (root / "extra").write_bytes(b"unconfirmed")
    assert root.exists()
    assert (root / "accepted-inputs.json").exists()


def test_failed_write_preserves_partial_files_without_retry(tmp_path, monkeypatch):
    transport = module()
    original, calls = transport.files.write_file, []

    def fail(directory, name, content, mode):
        calls.append(name)
        original(directory, name, content[:3], mode)
        raise OSError("invented private diagnostic")

    monkeypatch.setattr(transport.files, "write_file", fail)
    root = tmp_path.resolve() / "accepted-inputs"
    with pytest.raises(transport.OperationalInputTransportError) as caught:
        with transport.retain_operational_root_inputs(
            root, accepted_inputs=b"retained"
        ):
            pytest.fail("failed write yielded")
    assert str(caught.value) == "invalid_operational_input_transport"
    assert calls == ["accepted-inputs.json"]
    assert (root / "accepted-inputs.json").read_bytes() == b"ret"
