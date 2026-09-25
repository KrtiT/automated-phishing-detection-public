"""Persistence and held-input rejection never hide the original partial evidence."""

from types import SimpleNamespace

import pytest
from operational_child_fixtures import child_module


@pytest.mark.parametrize(
    "first", [KeyboardInterrupt("first"), SystemExit(7), ValueError("body")]
)
@pytest.mark.parametrize("later", [OSError("write"), KeyboardInterrupt("write")])
def test_progress_survives_failure_writer_error(first, later):
    api = child_module("client")
    first.progress = b'{"partial":"known"}\n'

    def retain(name, content):
        assert name == "client-failure.json" and content == first.progress
        raise later

    expected = later if isinstance(first, Exception) else first
    with pytest.raises(type(expected)) as caught:
        api._failed(SimpleNamespace(retain=retain), first)
    assert caught.value is expected and expected.progress == first.progress


def test_ordinary_failure_without_progress_invents_no_snapshot():
    api = child_module("client")
    error = ValueError("before replay")

    def forbidden(*args):
        pytest.fail("unknown progress was fabricated")

    with pytest.raises(ValueError) as caught:
        api._failed(SimpleNamespace(retain=forbidden), error)
    assert caught.value is error
