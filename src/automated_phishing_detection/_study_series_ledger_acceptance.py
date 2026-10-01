"""Join actual callback and observation facts before one atomic fresh acceptance."""

from . import _operational_cell_process_records as process
from . import _study_history_cell_records as publication
from . import _study_series_ledger_context as context
from . import _study_series_ledger_issuance as issuance
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_protocol import SNAPSHOT_NAMES
from ._operational_cell_publication_records import inventory
from ._study_execution_policy import DEADLINES
from ._study_series_ledger_context import digest, schema
from .operational_cell_inputs import RestoredOperationalCell
from .study_series_cell import SeriesCellScience
from .study_series_inputs import SeriesOperationalCell


def _inputs(ledger, candidate, current):
    schema.require(type(candidate) is SeriesCellScience)
    schema.require(type(candidate.inputs) is SeriesOperationalCell)
    inputs = candidate.inputs.computational
    schema.require(type(inputs) is RestoredOperationalCell)
    schema.require(inputs.accepted_bytes == ledger._metadata)
    schema.require(
        candidate.inputs.origin_metadata_bytes
        == canonical_bytes(schema.loads(ledger._metadata)["origin"])
    )
    schema.require(inputs.cell == current["cell"])
    schema.require(inputs.descriptor_bytes == current["descriptor_bytes"])
    schema.require(inputs.binding_bytes == current["binding_bytes"])
    schema.require(
        digest(inputs.manifest_bytes)
        == schema.loads(inputs.descriptor_bytes)["manifest_sha256"]
    )
    schema.require(
        candidate.reservation_sha256 == current["attempt"].reservation_sha256
    )
    context.cell_context(
        ledger,
        inputs.cell,
        current["attempt"],
        inputs.descriptor_bytes,
        inputs.binding_bytes,
    )
    return inputs


def evidence(ledger, candidate, current):
    inputs = _inputs(ledger, candidate, current)
    values = inventory(candidate.payloads, SNAPSHOT_NAMES)
    unused, public = publication.publication(
        values, inputs, candidate.reservation_sha256, str(current["attempt"].directory)
    )
    schema.require(type(candidate.summary_bytes) is bytes)
    publication.verify_summary(public, candidate.summary_bytes)
    return values


def _entries(ledger, observation):
    from ._study_series_ledger_records import encoded

    schema.require(len(ledger._entries) == 2 * (len(ledger._accepted) + 1))
    observed = schema.loads(observation.record, canonical=False)
    for offset, role in enumerate(("service", "client")):
        index = 2 * len(ledger._accepted) + offset
        entry, command = ledger._entries[index], ledger._commands[index]
        frame = issuance._frame(ledger, role, command)
        expected = issuance._entry(frame)
        expected.update(
            launched_pid=observed[role]["pid"],
            observation_recorded=True,
            exit_observed=True,
            exit_code=0,
        )
        schema.same(entry, expected)
        schema.require(entry["frame_bytes"] == encoded(frame.canonical_bytes))


def _observed(ledger, candidate, observation, pair, values):
    schema.require(
        type(pair) is bytes and pair == values["attempt/process-pair-intent.json"]
    )
    private = {
        name.removeprefix("attempt/"): value
        for name, value in values.items()
        if name.startswith("attempt/")
    }
    process.verify_process_records(
        private,
        inputs=candidate.inputs.computational,
        observation=observation,
        reservation=candidate.reservation_sha256,
        service_command=ledger._commands[-2],
        client_command=ledger._commands[-1],
        expected_deadlines=dict(DEADLINES),
    )
    _entries(ledger, observation)


def accept(ledger, candidate, observation, pair):
    from ._study_series_ledger_records import completed

    schema.require(not ledger._closed and ledger._current is not None)
    schema.require(len(ledger._entries) == 2 * (len(ledger._accepted) + 1))
    values = evidence(ledger, candidate, ledger._current)
    _observed(ledger, candidate, observation, pair, values)
    slot = completed(ledger._current, candidate, observation, pair)
    entries = ledger._entries[:-2] + [
        entry | {"accepted": True} for entry in ledger._entries[-2:]
    ]
    ledger.__dict__.update(
        _entries=entries, _accepted=[*ledger._accepted, slot], _current=None
    )
