"""Join full historical cell receipts to authenticated acceptance metadata."""

from . import _adopted_study_records as adopted
from . import _operational_cell_process_records as process
from . import _study_history_snapshot_records as records
from . import _study_run_schema as schema
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_protocol import PRIVATE_NAMES, SNAPSHOT_NAMES
from ._stopped_study_process import historical_process

PUBLIC_NAMES = {
    "schema_version",
    "protocol",
    "status",
    "execution",
    "cell",
    "summary",
    "private_sha256",
}


def _evidence(values, acceptance):
    decoded = {
        name: adopted.decoded(acceptance[name])
        for name in (
            "descriptor_bytes",
            "binding_bytes",
            "observation_bytes",
            "pair_intent_bytes",
        )
    }
    for name, key in (
        ("process-pair.json", "observation_bytes"),
        ("process-pair-intent.json", "pair_intent_bytes"),
    ):
        schema.require(values[f"attempt/{name}"] == decoded[key])
    return decoded


def _execution(root, descriptor, reservation):
    return {
        "kind": "operational_cell",
        "protocol": "operational-cell-v1",
        **{name: root[name] for name in schema.EXECUTION},
        "operational_profile_sha256": root["operational_profile_sha256"],
        "root_reservation_sha256": root["reservation_sha256"],
        "descriptor_sha256": records.digest(descriptor),
        "reservation_sha256": reservation,
    }


def _process_records(values, evidence, descriptor, metadata, reservation):
    observation, intent = historical_process(evidence, reservation)
    working = {
        name.removeprefix("attempt/"): content for name, content in values.items()
    }
    endpoint = process._lifecycle(working, observation, descriptor["cell"]["workload"])
    for role in ("service", "client"):
        observed = observation[role]
        command = intent[f"{role}_command_sha256"]
        for name, expected in (
            ("intent", {"command_sha256": command}),
            ("started", {"pid": observed["pid"]}),
            ("process", observed),
        ):
            schema.require(working[f"{role}-{name}.json"] == process._bytes(expected))
        expected = dict(
            schema_version=1,
            protocol="operational-role-v1",
            role=role,
            binding_sha256=records.digest(evidence["binding_bytes"]),
            pid=observed["pid"],
            command_sha256=command,
            base_url=endpoint,
            workload=descriptor["cell"]["workload"],
        )
        if role == "service":
            expected.update(metadata["primary"])
        schema.require(working[f"{role}-role.json"] == canonical_bytes(expected))


def cell_snapshot(snapshot, expected, acceptance, projection, authorization):
    schema.require(type(snapshot) is tuple and len(snapshot) == 3)
    ordinal, reservation, payloads = snapshot
    schema.require(type(ordinal) is int and ordinal == projection["cell"]["ordinal"])
    schema.require(reservation == projection["reservation_sha256"])
    values = records.authenticate(
        payloads, SNAPSHOT_NAMES, expected, projection["snapshot_sha256"]
    )
    evidence = _evidence(values, acceptance)
    descriptor = schema.load(evidence["descriptor_bytes"])
    execution = _execution(
        schema.load(authorization.execution_bytes),
        evidence["descriptor_bytes"],
        reservation,
    )
    profile = schema.load(authorization.profile_bytes)
    path = profile["paths"]["cells-dir"] + f"/cell-{ordinal:03d}-attempt"
    records.reservation(values, path, execution)
    private = records.outputs(values, PRIVATE_NAMES, "")
    public = records.public_record(
        values, execution, private, PUBLIC_NAMES, "operational_evidence_published", 1
    )
    schema.require(public["protocol"] == "operational-cell-v1")
    schema.same(public["cell"], projection["cell"])
    metadata = schema.load(authorization.source_results_bytes)["accepted_inputs"]
    _process_records(values, evidence, descriptor, metadata, reservation)
    records.finalization(values, execution, private)
    return ordinal, tuple(sorted(records.hashes(values).items()))


def accepted_cells(snapshots, expected, authorization):
    schema.require(type(snapshots) is tuple and type(expected) is tuple)
    ordinals = authorization.accepted_ordinals
    schema.require(len(snapshots) == len(expected) == len(ordinals))
    accounting = schema.load(authorization.accounting_bytes)
    scientific = schema.load(adopted.decoded(accounting["scientific_accounting_bytes"]))
    acceptances = accounting["authorization_ledger"]["cell_acceptances"]
    results = []
    for ordinal, snapshot, pin, acceptance, projection in zip(
        ordinals,
        snapshots,
        expected,
        acceptances,
        scientific["cells"][: len(ordinals)],
        strict=True,
    ):
        schema.require(type(pin) is tuple and len(pin) == 2)
        schema.require(type(pin[0]) is int and pin[0] == ordinal)
        results.append(
            cell_snapshot(snapshot, pin[1], acceptance, projection, authorization)
        )
    return tuple(results)
