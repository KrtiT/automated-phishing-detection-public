"""Synthetic admission pipes exercise ownership without real execution authority."""

import os
from contextlib import contextmanager

from automated_phishing_detection._study_admission_io import AdmissionPipe
from automated_phishing_detection._study_series_admission import (
    consume_series_admission,
)


@contextmanager
def live_pipe(case, monkeypatch):
    channel = AdmissionPipe()
    reader, writer = channel.open(case.child.frame.canonical_bytes)
    descriptor = os.dup(reader)
    monkeypatch.setenv(case.module.LOCATOR, str(descriptor))
    monkeypatch.setattr(
        case.module, "consume_series_admission", consume_series_admission
    )
    try:
        yield descriptor, writer, channel
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        channel.close()
