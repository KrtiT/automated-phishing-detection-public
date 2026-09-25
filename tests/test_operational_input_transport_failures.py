import pytest
from operational_input_transport_fixtures import (
    expectations,
    fake_restore,
    module,
    written,
)


@pytest.mark.parametrize(
    "original", [ValueError("private"), KeyboardInterrupt("private"), SystemExit(0)]
)
@pytest.mark.parametrize("operation", ["reader", "writer"])
def test_original_body_exception_preserved(tmp_path, monkeypatch, original, operation):
    if operation == "reader":
        paths = written(tmp_path)
        fake_restore(monkeypatch)
        manager = module().hold_operational_inputs(*paths, **expectations())
    else:
        manager = module().retain_operational_root_inputs(
            tmp_path.resolve() / "inputs", accepted_inputs=b"invented"
        )
    with pytest.raises(BaseException) as caught:
        with manager:
            raise original
    assert caught.value is original


@pytest.mark.parametrize("original", [KeyboardInterrupt("original"), SystemExit(0)])
@pytest.mark.parametrize(
    "cleanup", [ValueError("cleanup"), KeyboardInterrupt("cleanup"), SystemExit(17)]
)
def test_original_interruption_survives_cleanup_failure(
    tmp_path, monkeypatch, original, cleanup
):
    paths = written(tmp_path)
    fake_restore(monkeypatch)

    def fail(*arguments):
        raise cleanup

    with pytest.raises(BaseException) as caught:
        with module().hold_operational_inputs(*paths, **expectations()):
            monkeypatch.setattr(module().storage, "check_all", fail)
            raise original
    assert caught.value is original


@pytest.mark.parametrize(
    "cleanup", [ValueError("cleanup"), KeyboardInterrupt("cleanup"), SystemExit(0)]
)
def test_selected_cleanup_error_keeps_original_failure_associations(
    tmp_path, monkeypatch, cleanup
):
    paths = written(tmp_path)
    fake_restore(monkeypatch)
    original = ValueError("body")
    original.progress, original.worker_failure = b"private progress", object()

    def fail(*arguments):
        raise cleanup

    with pytest.raises(BaseException) as caught:
        with module().hold_operational_inputs(*paths, **expectations()):
            monkeypatch.setattr(module().storage, "check_all", fail)
            raise original
    assert caught.value.progress == original.progress
    assert caught.value.worker_failure is original.worker_failure
    if isinstance(cleanup, Exception):
        assert str(caught.value) == "invalid_operational_input_transport"
    else:
        assert caught.value is cleanup


def test_later_specific_failure_association_not_overwritten(tmp_path, monkeypatch):
    paths = written(tmp_path)
    fake_restore(monkeypatch)
    original, cleanup = ValueError("body"), KeyboardInterrupt("cleanup")
    original.progress, cleanup.progress = b"earlier", b"later"

    def fail(*arguments):
        raise cleanup

    with pytest.raises(KeyboardInterrupt) as caught:
        with module().hold_operational_inputs(*paths, **expectations()):
            monkeypatch.setattr(module().storage, "check_all", fail)
            raise original
    assert caught.value is cleanup
    assert caught.value.progress == b"later"


def test_failed_entry_creates_no_fabricated_associations(tmp_path):
    with pytest.raises(module().OperationalInputTransportError) as caught:
        with module().hold_operational_inputs(
            tmp_path / "missing", tmp_path / "missing2", **expectations()
        ):
            pytest.fail("failed entry yielded")
    assert not vars(caught.value)


def test_restore_error_context_survives_later_directory_rejection(
    tmp_path, monkeypatch
):
    paths = written(tmp_path)
    original, cleanup = ValueError("pure rejection"), KeyboardInterrupt("cleanup")
    original.progress = b"existing private diagnostic"

    def fail(*arguments):
        raise cleanup

    def restore(contents, expected):
        monkeypatch.setattr(module().storage, "check_all", fail)
        raise original

    monkeypatch.setattr(module(), "_restore", restore)
    with pytest.raises(KeyboardInterrupt) as caught:
        with module().hold_operational_inputs(*paths, **expectations()):
            pytest.fail("rejected restore yielded")
    assert caught.value is cleanup
    assert caught.value.progress == original.progress
