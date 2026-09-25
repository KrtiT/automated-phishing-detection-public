import signal

import pytest
from operational_input_signal_fixtures import (
    assert_closed,
    interrupt_fdopen,
    watch_open,
)
from operational_input_transport_fixtures import (
    expectations,
    fake_restore,
    module,
    written,
)


@pytest.mark.parametrize(
    "selected",
    [
        "accepted-inputs",
        "cell-001",
        "binding.json",
        "descriptor.json",
        "accepted-inputs.json",
        "manifest",
    ],
)
@pytest.mark.parametrize("boundary", ["open", "close"])
def test_real_sigint_reader_acquisition_and_cleanup(
    tmp_path, monkeypatch, selected, boundary
):
    paths = written(tmp_path)
    fake_restore(monkeypatch)
    with monkeypatch.context() as guard:
        descriptors = watch_open(
            guard,
            selected,
            interrupt=boundary == "open",
            before_close=boundary == "close",
        )
        with pytest.raises(KeyboardInterrupt):
            with module().hold_operational_inputs(*paths, **expectations()):
                pass
    assert descriptors
    assert_closed(descriptors)
    assert all(path.exists() for path in paths)


@pytest.mark.parametrize("selected", ["accepted-inputs", "accepted-inputs.json"])
@pytest.mark.parametrize("boundary", ["open", "close"])
def test_real_sigint_writer_preserves_partial_state_and_closes(
    tmp_path, monkeypatch, selected, boundary
):
    path = tmp_path.resolve() / "accepted-inputs"
    with monkeypatch.context() as guard:
        descriptors = watch_open(
            guard,
            selected,
            interrupt=boundary == "open",
            before_close=boundary == "close",
        )
        with pytest.raises(KeyboardInterrupt):
            with module().retain_operational_root_inputs(
                path, accepted_inputs=b"invented"
            ):
                pass
    assert descriptors
    assert_closed(descriptors)
    assert path.exists()
    if selected == "accepted-inputs.json":
        assert (path / selected).exists()


def test_real_sigint_after_reader_fdopen_closes_owned_descriptor(tmp_path, monkeypatch):
    paths = written(tmp_path)
    fake_restore(monkeypatch)
    with monkeypatch.context() as guard:
        descriptors = interrupt_fdopen(guard)
        with pytest.raises(KeyboardInterrupt):
            with module().hold_operational_inputs(*paths, **expectations()):
                pytest.fail("interrupted file open yielded")
    assert descriptors
    assert_closed(descriptors)


def test_real_sigint_after_writer_fdopen_preserves_file(tmp_path, monkeypatch):
    path = tmp_path.resolve() / "accepted-inputs"
    with monkeypatch.context() as guard:
        descriptors = interrupt_fdopen(guard)
        with pytest.raises(KeyboardInterrupt):
            with module().retain_operational_root_inputs(
                path, accepted_inputs=b"invented"
            ):
                pytest.fail("interrupted file open yielded")
    assert descriptors
    assert_closed(descriptors)
    assert (path / "accepted-inputs.json").exists()


def test_pure_restore_is_not_interrupt_deferred(tmp_path, monkeypatch):
    paths = written(tmp_path)
    reached = []

    def restore(contents, expected):
        signal.raise_signal(signal.SIGINT)
        reached.append("after signal")

    monkeypatch.setattr(module(), "_restore", restore)
    with monkeypatch.context() as guard:
        descriptors = watch_open(guard, "accepted-inputs")
        with pytest.raises(KeyboardInterrupt):
            with module().hold_operational_inputs(*paths, **expectations()):
                pytest.fail("interrupted pure restore yielded")
    assert not reached
    assert_closed(descriptors)


@pytest.mark.parametrize("operation", ["reader", "writer"])
def test_yielded_body_is_not_interrupt_deferred(tmp_path, monkeypatch, operation):
    reached = []
    if operation == "reader":
        paths = written(tmp_path)
        fake_restore(monkeypatch)
        manager = module().hold_operational_inputs(*paths, **expectations())
    else:
        manager = module().retain_operational_root_inputs(
            tmp_path.resolve() / "accepted-inputs", accepted_inputs=b"invented"
        )
    with pytest.raises(KeyboardInterrupt):
        with manager:
            signal.raise_signal(signal.SIGINT)
            reached.append("after signal")
    assert not reached
