"""Invented opaque root records for isolated IO lifecycle tests, not authority."""

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

from study_series_admission_fixtures import frame as admission_frame

from automated_phishing_detection._study_history_snapshot_records import digest


def api():
    name = "automated_phishing_detection._study_series_child_prefix"
    assert find_spec(name), "missing held four-file series prefix"
    return import_module(name)


def setup(tmp_path, monkeypatch):
    paths = {name: tmp_path.resolve() / name for name in ("series", "segment")}
    values = {
        "series/reservation.json": b"invented series reservation",
        "segment/reservation.json": b"invented segment reservation",
        "segment/segment-intent.json": b"invented intent",
        "segment/history-import.json": b"invented history import",
    }
    for directory in paths.values():
        directory.mkdir(mode=0o700)
    for name, content in values.items():
        path = tmp_path.resolve() / name
        path.write_bytes(content)
        path.chmod(0o600)
    frame = admission_frame(
        series_reservation_sha256=digest(values["series/reservation.json"]),
        segment_reservation_sha256=digest(values["segment/reservation.json"]),
        intent_sha256=digest(values["segment/segment-intent.json"]),
        predecessor_sha256=digest(values["segment/history-import.json"]),
    )
    case = SimpleNamespace(
        paths=paths, values=values, frame=frame, binding=object(), calls=[]
    )
    monkeypatch.setattr(api(), "_context", lambda *args: paths)
    monkeypatch.setattr(api(), "_validate", lambda *args: case.calls.append(args))
    return case


def hold(case):
    return api().hold_series_child_prefix(case.binding, case.frame)
