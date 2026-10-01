import pytest
from study_series_child_inputs_fixtures import (
    candidates,
    child_case,
    manifests,
    series_case,
)
from study_series_child_transport_fixtures import api, hold, written

__all__ = ["candidates", "manifests", "series_case", "child_case"]


def test_actual_held_child_inputs_equal_stronger_parent(tmp_path, child_case):
    paths = written(tmp_path, child_case)
    with hold(paths, child_case) as result:
        assert result == child_case.expected
        assert result.authorizes_execution is False
    assert all(path.exists() for path in paths)


@pytest.mark.parametrize(
    "field,value",
    (
        ("profile_bytes", b"changed"),
        ("frame", None),
        ("expected_cell_reservation_sha256", True),
    ),
)
def test_invalid_expectations_do_not_open_input_paths(
    tmp_path, child_case, monkeypatch, field, value
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid expectation opened protected inputs")

    monkeypatch.setattr(api().storage, "hold", forbidden)
    with pytest.raises(api().OperationalInputTransportError):
        with hold(
            (tmp_path / "missing", tmp_path / "other"), child_case, **{field: value}
        ):
            pytest.fail("invalid inputs yielded")


def test_yielded_body_interrupt_preserved_without_deleting_inputs(tmp_path, child_case):
    paths = written(tmp_path, child_case)
    original = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with hold(paths, child_case):
            raise original
    assert caught.value is original
    assert all(path.exists() for path in paths)
