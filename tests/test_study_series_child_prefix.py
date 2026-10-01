import pytest
from study_series_child_prefix_fixtures import api, hold, setup


def test_four_pinned_immutable_buffers_retained_while_root_directories_grow(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    with hold(case) as payloads:
        assert type(payloads) is tuple and dict(payloads) == case.values
        (case.paths["series"] / "new-accounting.json").write_bytes(
            b"invented later record"
        )
        (case.paths["segment"] / "evidence").mkdir(mode=0o700)
    assert case.calls == [(case.binding, case.frame, case.values)]


@pytest.mark.parametrize(
    "name",
    (
        "series/reservation.json",
        "segment/reservation.json",
        "segment/segment-intent.json",
        "segment/history-import.json",
    ),
)
def test_held_prefix_byte_mutation_rejects_normal_exit(tmp_path, monkeypatch, name):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(api().SeriesChildPrefixError):
        with hold(case):
            (tmp_path / name).write_bytes(b"changed")


def test_yielded_interrupt_survives_prefix_cleanup_failure(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    original = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with hold(case):
            (case.paths["series"] / "reservation.json").write_bytes(b"changed")
            raise original
    assert caught.value is original
