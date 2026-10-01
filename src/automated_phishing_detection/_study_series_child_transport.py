"""Hold four parent-pinned series input files without live access authority."""

from contextlib import contextmanager

from . import _operational_input_files as storage
from . import _operational_input_schema as schema
from . import _study_preparation_files as files
from . import _study_series_input_context as context
from . import operational_input_transport as original
from ._study_series_admission import validate_series_frame
from .study_series_child_inputs import restore_series_child_inputs

OperationalInputTransportError = original.OperationalInputTransportError


def _preflight(profile_bytes, frame, reservation):
    validate_series_frame(frame)
    schema.digest(reservation)
    profile = context._profile(profile_bytes, frame.profile_sha256)
    schema.require(frame.segment_ordinal == profile["segment"]["ordinal"])
    schema.require(
        profile["segment"]["start_ordinal"]
        <= frame.cell_ordinal
        <= profile["segment"]["end_ordinal"]
    )
    schema.require(
        frame.origin_reservation_sha256 == profile["origin"]["root_reservation_sha256"]
    )
    schema.require(frame.history_index_sha256 == profile["history"]["index_sha256"])


def _restore_held(held, profile_bytes, frame, reservation):
    contents = original._contents(held, frame.cell_binding_sha256)
    restored = restore_series_child_inputs(
        contents["accepted-inputs.json"],
        profile_bytes,
        contents["descriptor.json"],
        contents["binding.json"],
        contents["manifest"],
        frame=frame,
        expected_cell_reservation_sha256=reservation,
    )
    files.deferred(storage.check_all, held)
    return restored


@contextmanager
def hold_series_child_inputs(
    accepted_inputs_directory,
    cell_input_directory,
    *,
    profile_bytes,
    frame,
    expected_cell_reservation_sha256,
):
    """Yield a computational carrier only while all original input identities hold."""
    failure, yielded = None, False
    try:
        _preflight(profile_bytes, frame, expected_cell_reservation_sha256)
        paths = original._paths(accepted_inputs_directory, cell_input_directory)
        with storage.hold(*paths) as held:
            try:
                restored = _restore_held(
                    held, profile_bytes, frame, expected_cell_reservation_sha256
                )
                yielded = True
                yield restored
            except BaseException as error:
                failure = error
                raise
    except BaseException as error:
        original._reject(error, failure, yielded=yielded)
