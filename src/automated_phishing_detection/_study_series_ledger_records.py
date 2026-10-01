"""Immutable failure facts and complete canonical fresh-suffix accounting."""

import base64
from dataclasses import dataclass, field

from . import _study_series_ledger_context as context
from ._checkpoint_codec import canonical_bytes
from ._operational_process_records import ProcessObservation, _bytes
from ._study_series_ledger_context import digest, schema
from .study_series_cell import SeriesCellScience

STAGES = frozenset(
    (
        "validation",
        "selection",
        "reservation",
        "input_retention",
        "observation",
        "completion",
        "finalization",
    )
)
FACTS = (
    "stage",
    "reservation_sha256",
    "descriptor_bytes",
    "binding_bytes",
    "pair_intent_bytes",
    "observation_bytes",
    "progress_bytes",
    "snapshot_sha256",
    "publishing",
    "holders_closed",
)


@dataclass(frozen=True)
class SeriesCellStop:
    """Actual caller state only; construction is not a process or holder attestation."""

    stage: str
    attempt: object = field(default=None, repr=False)
    descriptor_bytes: bytes | None = field(default=None, repr=False)
    binding_bytes: bytes | None = field(default=None, repr=False)
    pair_intent_bytes: bytes | None = field(default=None, repr=False)
    observation: ProcessObservation | None = field(default=None, repr=False)
    progress_bytes: bytes | None = field(default=None, repr=False)
    candidate: SeriesCellScience | None = field(default=None, repr=False)
    publishing: bool = False


def encoded(content):
    schema.require(type(content) is bytes)
    return base64.b64encode(content).decode("ascii")


def _optional(content):
    return None if content is None else encoded(content)


def empty(ordinal):
    return dict(ordinal=ordinal, status="unattempted", **dict.fromkeys(FACTS))


def _hashes(candidate):
    return {name: digest(content) for name, content in candidate.payloads}


def completed(current, candidate, observation, pair):
    return empty(current["cell"].ordinal) | dict(
        status="accepted",
        reservation_sha256=current["attempt"].reservation_sha256,
        descriptor_bytes=encoded(current["descriptor_bytes"]),
        binding_bytes=encoded(current["binding_bytes"]),
        pair_intent_bytes=encoded(pair),
        observation_bytes=encoded(observation.record),
        snapshot_sha256=_hashes(candidate),
        publishing=True,
        holders_closed=True,
    )


def _current(ledger, stopped):
    current = ledger._current or dict(cell=context.next_cell(ledger))
    result = current.copy()
    for name in ("attempt", "descriptor_bytes", "binding_bytes"):
        supplied, retained = getattr(stopped, name), current.get(name)
        schema.require(supplied is None or retained is None or supplied == retained)
        result[name] = retained if supplied is None else supplied
    return result


def _process(content, reservation):
    schema.require(type(content) is bytes)
    value = schema.loads(content, canonical=False)
    schema.require(content == _bytes(value))
    schema.require(
        reservation is not None and value["reservation_sha256"] == reservation
    )


def _candidate(ledger, stopped, current):
    from ._study_series_ledger_acceptance import evidence

    if stopped.candidate is None:
        return None
    schema.require(stopped.publishing)
    values = evidence(ledger, stopped.candidate, current)
    for name, content in (
        (
            "process-pair.json",
            None if stopped.observation is None else stopped.observation.record,
        ),
        ("process-pair-intent.json", stopped.pair_intent_bytes),
    ):
        schema.require(content is None or content == values[f"attempt/{name}"])
    return _hashes(stopped.candidate)


def _stopped_slot(ledger, stopped, current):
    reservation = context.partial(ledger, current)
    observation = stopped.observation
    schema.require(observation is None or type(observation) is ProcessObservation)
    content = None if observation is None else observation.record
    for value in (content, stopped.progress_bytes, stopped.pair_intent_bytes):
        if value is not None:
            _process(value, reservation)
    return empty(current["cell"].ordinal) | dict(
        status="stopped",
        stage=stopped.stage,
        reservation_sha256=reservation,
        descriptor_bytes=_optional(current["descriptor_bytes"]),
        binding_bytes=_optional(current["binding_bytes"]),
        pair_intent_bytes=_optional(stopped.pair_intent_bytes),
        observation_bytes=_optional(content),
        progress_bytes=_optional(stopped.progress_bytes),
        snapshot_sha256=_candidate(ledger, stopped, current),
        publishing=stopped.publishing,
        holders_closed=False,
    )


def stop(ledger, stopped):
    ledger._closed = True
    schema.require(ledger._stopped is None and type(stopped) is SeriesCellStop)
    schema.require(type(stopped.stage) is str and stopped.stage in STAGES)
    schema.require(type(stopped.publishing) is bool)
    current = _current(ledger, stopped)
    slot = _stopped_slot(ledger, stopped, current)
    ledger.__dict__.update(_current=current, _stopped=slot)


def snapshot(ledger):
    schema.require(ledger._current is None or ledger._stopped is not None)
    cells = list(ledger._accepted)
    if ledger._stopped is not None:
        cells.append(ledger._stopped)
    first = ledger._profile["segment"]["start_ordinal"]
    cells.extend(empty(ordinal) for ordinal in range(first + len(cells), 126))
    return canonical_bytes(
        dict(
            schema_version=1,
            protocol="study-series-ledger-v1",
            parent_pid=ledger._parent_pid,
            series_reservation_sha256=ledger._series.reservation_sha256,
            segment_reservation_sha256=ledger._segment.reservation_sha256,
            intent_sha256=digest(ledger._intent),
            history_import_sha256=digest(ledger._import),
            accepted_inputs_sha256=digest(ledger._metadata),
            segment_ordinal=2,
            start_ordinal=first,
            end_ordinal=125,
            admissions=ledger._entries,
            cells=cells,
        )
    )
