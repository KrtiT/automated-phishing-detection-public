"""Only the two concrete checkpoint writers enter failure snapshots."""

import pytest
from test_external_source_checkpoints import writer

from automated_phishing_detection._external_source_checkpoints import (
    ExternalCheckpointWriter,
)
from automated_phishing_detection._prepared_external_io import (
    PreparedExternalCheckpointWriter,
)
from automated_phishing_detection.external_source_failure import (
    ExternalSourceFailureError,
    ExternalSourceFailureState,
)

__all__ = ["writer"]


@pytest.mark.parametrize(
    "base", [ExternalCheckpointWriter, PreparedExternalCheckpointWriter]
)
def test_unknown_writer_subclasses_cannot_enter_failure_snapshot(writer, base):
    class UnknownWriter(base):
        pass

    state = ExternalSourceFailureState()
    state.writer = UnknownWriter(writer.attempt, identity=writer.identity)
    with pytest.raises(ExternalSourceFailureError):
        state.snapshot(writer.attempt, writer.identity, "scoring", ValueError())
