"""Rehash invented complete evidence; no source execution or approval is proved."""

import json
from dataclasses import replace
from hashlib import sha256

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection import _study_run_schema as schema
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_admission import decode_admission_frame


def digest(content):
    return sha256(content).hexdigest()


def sources(content, execution, external_pin):
    source = json.loads(content)
    source["execution"] = execution
    accepted = source["accepted_inputs"]
    accepted["root_reservation_sha256"] = execution["reservation_sha256"]
    accepted["operational_profile_sha256"] = execution["operational_profile_sha256"]
    accepted["execution"] = {name: execution[name] for name in schema.EXECUTION}
    for role in ("internal", "external"):
        accepted[role]["execution"].update(accepted["execution"])
    handoff = digest(canonical_bytes(accepted["internal"]))
    external = accepted["external"]
    external["execution"].update(
        internal_handoff_sha256=handoff, source_profile_sha256=external_pin
    )
    for directory in ("checkpoints", "evidence"):
        external["snapshot_sha256"][
            f"attempt/{directory}/internal-source-handoff.json"
        ] = handoff
    source["accepted_inputs_sha256"] = digest(canonical_bytes(accepted))
    return canonical_bytes(source)


def _bindings(ledger, execution):
    bindings = {}
    for value in ledger["cell_acceptances"]:
        descriptor = json.loads(records.decoded(value["descriptor_bytes"]))
        descriptor.update(
            root_reservation_sha256=execution["reservation_sha256"],
            accepted_inputs_sha256=ledger["accepted_inputs_sha256"],
        )
        content = canonical_bytes(descriptor)
        bound = json.loads(records.decoded(value["binding_bytes"]))
        binding = canonical_bytes(bound | {"descriptor_sha256": digest(content)})
        value.update(
            descriptor_bytes=records.encoded(content),
            binding_bytes=records.encoded(binding),
        )
        bindings[value["cell_ordinal"]] = digest(binding)
    return bindings


def _frames(ledger, execution, bindings):
    for entry in ledger["admissions"]:
        changes = {
            "profile_sha256": execution["study_profile_sha256"],
            "envelope_sha256": execution["adoption_envelope_sha256"],
            "root_reservation_sha256": execution["reservation_sha256"],
            "intent_sha256": ledger["intent_sha256"],
            "barrier_sha256": ledger["barrier_sha256"],
        }
        if entry["role"] == "external":
            changes["predecessor_sha256"] = ledger["handoff_sha256"]
        elif entry["role"] in ("service", "client"):
            changes.update(
                predecessor_sha256=ledger["source_results_sha256"],
                accepted_inputs_sha256=ledger["accepted_inputs_sha256"],
                cell_binding_sha256=bindings[entry["cell_ordinal"]],
            )
        frame = replace(
            decode_admission_frame(records.decoded(entry["frame_bytes"])), **changes
        )
        entry.update(
            frame_bytes=records.encoded(frame.canonical_bytes),
            frame_sha256=frame.sha256,
        )


def accounting(case, execution, contents):
    source = json.loads(contents["source-results.json"])
    ledger = case.ledger
    ledger.update(
        intent_sha256=digest(contents["study-intent.json"]),
        barrier_sha256=digest(contents["prediction-barrier.json"]),
        handoff_sha256=source["accepted_inputs"]["external"]["execution"][
            "internal_handoff_sha256"
        ],
        source_results_sha256=digest(contents["source-results.json"]),
        accepted_inputs_sha256=source["accepted_inputs_sha256"],
    )
    _frames(ledger, execution, _bindings(ledger, execution))
    wrapped = json.loads(case.contents["study-accounting.json"])
    scientific = json.loads(records.decoded(wrapped["scientific_accounting_bytes"]))
    scientific["execution"] = records.scientific_execution(execution)
    return canonical_bytes(
        records.envelope("matrix_accepted", execution)
        | {
            "scientific_accounting_bytes": records.encoded(canonical_bytes(scientific)),
            "authorization_ledger": ledger,
        }
    )
