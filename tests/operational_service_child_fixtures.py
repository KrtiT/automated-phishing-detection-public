"""Inherited socket/pipe fixtures exercise the real child service lifecycle."""

import os
import socket
from contextlib import contextmanager

from operational_owner_fixtures import models
from test_selective_service import SyntheticScorer

from automated_phishing_detection import bound_runtime


@contextmanager
def inherited(monkeypatch):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener_fd = os.dup(listener.fileno())
    stop_read, stop_write = os.pipe()
    ready_read, ready_write = os.pipe()
    owned = (listener_fd, stop_read, ready_write)
    for name, value in zip(("APD_LISTENER_FD", "APD_STOP_FD", "APD_READY_FD"), owned):
        monkeypatch.setenv(name, str(value))
    base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    monkeypatch.setenv("APD_BASE_URL", base_url)
    try:
        yield base_url, stop_write, ready_read, owned
    finally:
        listener.close()
        for descriptor in (*owned, stop_write, ready_read):
            try:
                os.close(descriptor)
            except OSError:
                pass


def synthetic_primary(monkeypatch, primary):
    bound, scorer = models(), SyntheticScorer()
    points = primary["thresholds"]
    bound.artifact_hashes = tuple(sorted(primary["artifact_hashes"].items()))
    bound.length_only.validation_threshold_record["threshold"] = points["length_only"]
    bound.cascade.stage1_threshold = points["logistic_l1"]
    bound.cascade.transformer_threshold = points["transformer"]
    bound.cascade.half_width = points["half_width"]
    bound.monitor_boundary = points["monitor_boundary"]
    monkeypatch.setattr(bound_runtime, "recheck_binding", lambda binding: None)
    monkeypatch.setattr(bound_runtime, "load_bound_models", lambda *args: bound)
    monkeypatch.setattr(bound_runtime, "SelectiveCascade", lambda model: scorer)
    return scorer
