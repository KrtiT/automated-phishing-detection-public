import os

import pytest
from study_preparation_transport_fixtures import (
    inputs,
    module,
    preparation_api,
    preparation_case,
    retained_case,
    runner,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "retained_case", "runner"]


def test_reader_retains_real_preparation_without_rewriting(retained_case):
    retained, api = retained_case, module()
    before = {entry.name: entry.stat() for entry in retained.directory.iterdir()}
    with api.hold_study_preparation(
        retained.directory, **retained.expected
    ) as restored:
        assert restored.payloads == retained.snapshot.payloads
        assert restored.reservation_sha256 == retained.snapshot.reservation_sha256
        assert (
            restored.completion_sha256
            == retained.expected["expected_completion_sha256"]
        )
    after = {entry.name: entry.stat() for entry in retained.directory.iterdir()}
    assert before == after


def test_reader_opens_each_retained_file_once(retained_case, monkeypatch):
    retained, api = retained_case, module()
    names = ("reservation.json", *(name for name, _ in retained.snapshot.payloads))
    original, opened = os.open, []

    def observe(name, flags, *arguments, **keywords):
        if name in names:
            opened.append(name)
            assert not flags & (os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_RDWR)
        return original(name, flags, *arguments, **keywords)

    monkeypatch.setattr(os, "open", observe)
    with api.hold_study_preparation(retained.directory, **retained.expected):
        pass
    assert tuple(opened) == names


@pytest.mark.parametrize(
    "failure", [KeyboardInterrupt(), SystemExit(7), RuntimeError("invented")]
)
def test_reader_preserves_body_failure_and_files(retained_case, failure):
    retained, api = retained_case, module()
    names = set(entry.name for entry in retained.directory.iterdir())
    with pytest.raises(type(failure)) as captured:
        with api.hold_study_preparation(retained.directory, **retained.expected):
            raise failure
    assert captured.value is failure
    assert set(entry.name for entry in retained.directory.iterdir()) == names
