"""One parent issues fixed continuations only after actual predecessor acceptance."""

from hashlib import sha256

from . import _adopted_study_records as records
from . import _study_run_schema as schema
from ._checkpoint_codec import canonical_bytes
from .internal_external_handoff import build_internal_handoff
from .operational_schedule import validate_cell


class AdmissionLedger:
    def __init__(self, authorization, attempt, intent):
        self.authorization, self.attempt = authorization, attempt
        self.intent_sha256 = sha256(intent).hexdigest()
        self.preparation = self.barrier_sha256 = None
        self.handoff_sha256 = self.source_results_sha256 = None
        self.accepted_inputs_sha256 = None
        self.entries, self.completed_cells = [], 0
        self.cell_acceptances = []
        self.current_cell = self.cell_binding_sha256 = None

    def barrier_retained(self, preparation, content):
        execution = records.original.study_identity(
            self.authorization.base, self.authorization.operational
        )
        execution["reservation_sha256"] = self.attempt.reservation_sha256
        expected, held = records.original.prediction_barrier(
            preparation, execution=execution
        )
        schema.require(not held and expected == content)
        value = schema.load(content)
        schema.require(self.barrier_sha256 is None and not self.entries)
        schema.require(value["status"] == "necessary_capacity_present")
        schema.require(value["predictions_started"] is False)
        schema.feasibility(value["feasibility"])
        schema.require(not value["feasibility"]["shortages"])
        schema.require(
            value["study_preparation_reservation_sha256"]
            == preparation.reservation_sha256
        )
        schema.require(
            value["study_preparation_complete_sha256"] == preparation.completion_sha256
        )
        self.preparation, self.barrier_sha256 = preparation, sha256(content).hexdigest()

    def _accepted_worker(self, role, observed):
        entry = next((entry for entry in self.entries if entry["role"] == role), None)
        schema.require(entry is not None and not entry["accepted"])
        worker = observed.worker
        schema.require(worker.command_sha256 == entry["command_sha256"])
        schema.require(worker.exit.pid == entry["launched_pid"])
        schema.require(worker.exit.exit_observed is True and worker.exit.exit_code == 0)
        schema.require(entry["exit_observed"] is True and entry["exit_code"] == 0)
        return entry

    def internal_accepted(self, observed):
        entry = self._accepted_worker("internal", observed)
        handoff = build_internal_handoff(observed)
        self.handoff_sha256 = sha256(handoff.handoff_bytes).hexdigest()
        entry["accepted"] = True

    def external_accepted(self, observed):
        from .external_source_handoff import ObservedExternalCompletion

        schema.require(type(observed) is ObservedExternalCompletion)
        self._accepted_worker("external", observed)["accepted"] = True

    def inputs_retained(self, accepted, source_results_bytes):
        schema.require(self.accepted_inputs_sha256 is None)
        schema.require(
            len(self.entries) == 2 and all(entry["accepted"] for entry in self.entries)
        )
        source = schema.load(source_results_bytes)
        schema.same(source["accepted_inputs"], schema.load(accepted.metadata_bytes))
        self.accepted_inputs_sha256 = sha256(accepted.metadata_bytes).hexdigest()
        schema.require(source["accepted_inputs_sha256"] == self.accepted_inputs_sha256)
        self.source_results_sha256 = sha256(source_results_bytes).hexdigest()

    def cell_started(self, cell):
        validate_cell(cell)
        schema.require(
            self.accepted_inputs_sha256 is not None and self.current_cell is None
        )
        schema.require(cell.ordinal == self.completed_cells + 1)
        self.current_cell, self.cell_binding_sha256 = cell, None

    def cell_accepted(self, cell, retained, *, pair_intent_bytes):
        from ._adopted_study_cell_evidence import evidence, validate
        from ._study_operational_projection import project
        from .study_operational_records import OperationalCellSlot

        schema.require(cell == self.current_cell)
        entries = self.entries[-2:]
        schema.require(
            tuple(entry["role"] for entry in entries) == ("service", "client")
        )
        schema.require(all(entry["cell_ordinal"] == cell.ordinal for entry in entries))
        schema.require(
            all(entry["exit_observed"] and entry["exit_code"] == 0 for entry in entries)
        )
        value = evidence(retained, pair_intent_bytes)
        validate(
            value,
            entries,
            {"reservation_sha256": self.attempt.reservation_sha256},
            self.accepted_inputs_sha256,
            project(OperationalCellSlot(cell, accepted=retained)),
        )
        self.cell_acceptances.append(value)
        for entry in entries:
            entry["accepted"] = True
        self.completed_cells += 1
        self.current_cell = self.cell_binding_sha256 = None

    def _gate(self, role, predecessor, accepted, cell_binding):
        schema.require(self.barrier_sha256 is not None)
        if role == "internal":
            schema.require(
                not self.entries
                and (predecessor, accepted, cell_binding) == (None, None, None)
            )
        elif role == "external":
            schema.require(len(self.entries) == 1 and self.entries[0]["accepted"])
            schema.require(
                predecessor == self.handoff_sha256 and accepted is cell_binding is None
            )
        else:
            self._cell_gate(role, predecessor, accepted, cell_binding)

    def _cell_gate(self, role, predecessor, accepted, cell_binding):
        schema.require(self.current_cell is not None and role in ("service", "client"))
        schema.require(predecessor == self.source_results_sha256)
        schema.require(accepted == self.accepted_inputs_sha256)
        schema.operational.digest(cell_binding)
        expected = 2 + 2 * self.completed_cells + (role == "client")
        schema.require(len(self.entries) == expected)
        if role == "client":
            schema.require(self.entries[-1]["launched_pid"] is not None)
            schema.require(cell_binding == self.cell_binding_sha256)
        else:
            self.cell_binding_sha256 = cell_binding

    def issue(
        self,
        role,
        command,
        *,
        predecessor_sha256=None,
        accepted_inputs_sha256=None,
        cell_binding_sha256=None,
    ):
        from ._adopted_study_issuance import issue_admission

        self._gate(
            role, predecessor_sha256, accepted_inputs_sha256, cell_binding_sha256
        )
        return issue_admission(
            self,
            role,
            command,
            predecessor_sha256,
            accepted_inputs_sha256,
            cell_binding_sha256,
        )

    def snapshot(self):
        return canonical_bytes(
            {
                "schema_version": 1,
                "protocol": "study-authorization-ledger-v1",
                "intent_sha256": self.intent_sha256,
                "barrier_sha256": self.barrier_sha256,
                "handoff_sha256": self.handoff_sha256,
                "source_results_sha256": self.source_results_sha256,
                "accepted_inputs_sha256": self.accepted_inputs_sha256,
                "cell_acceptances": self.cell_acceptances,
                "admissions": self.entries,
            }
        )
