"""Parent and child writers share direct create-only fixed-name retention."""

import importlib
import importlib.util

import pytest

from automated_phishing_detection import execution_receipt as receipt


def module():
    name = "automated_phishing_detection._operational_attempt_io"
    assert importlib.util.find_spec(name), "missing shared fixed-name attempt writer"
    return importlib.import_module(name)


def test_parent_and_child_additions_are_not_staging_directories(tmp_path):
    api = module()
    attempt = receipt.reserve_attempt(
        tmp_path / "attempt", identity={"kind": "fixture"}
    )
    with api.held_attempt_writer(attempt, names=("service-started.json",)) as parent:
        with api.held_attempt_writer(attempt, names=("service-role.json",)) as child:
            parent.retain("service-started.json", b"parent")
            child.retain("service-role.json", b"child")
            parent.check()
            child.check()
    assert {path.name for path in attempt.directory.iterdir()} == {
        "reservation.json",
        "service-started.json",
        "service-role.json",
    }


@pytest.mark.parametrize(
    "names",
    [
        (),
        [],
        ("reservation.json",),
        ("other",),
        ("../name",),
        ("service-role.json", "service-role.json"),
        ("client-failure.json",),
    ],
)
def test_invalid_parent_names_reject_without_attempt_access(names):
    with pytest.raises(ValueError):
        with module().held_attempt_writer(object(), names=names):
            pytest.fail("invalid private writer names yielded")
