"""Join accepted primary identity inside one lazy inference owner session.

This composition does not authorize access or establish actual process exits.
The caller retains role bytes; the observing parent authenticates the cell.
"""

from contextlib import contextmanager

from ._checkpoint_codec import canonical_bytes
from ._exception_cleanup import CleanupStack
from ._external_source_profile import _execution
from .bound_models import ArtifactPaths
from .bound_runtime import open_bound_session
from .evaluation_producer import _thresholds
from .execution_preflight import ExecutionBinding
from .live_monitor import LiveMonitor
from .operational_role_records import build_client_role, build_service_role
from .selective_service import create_app
from .shift_schema import ShiftPlan
from .shift_service import create_shift_app


class OperationalRuntimeError(ValueError):
    """The loaded owner or supplied execution context does not match the cell."""


def _require(condition):
    if not condition:
        raise OperationalRuntimeError("invalid_operational_runtime")


def _validate(binding, paths, inputs, role_context, retain):
    try:
        _require(type(binding) is ExecutionBinding)
        _require(type(paths) is ArtifactPaths and callable(retain))
        build_client_role(inputs, role_context)
        execution = _execution(binding, dict(binding.source_hashes))
        _require(canonical_bytes(inputs.execution) == canonical_bytes(execution))
    except Exception:
        raise OperationalRuntimeError("invalid_operational_runtime") from None


def _primary(session):
    pairs = session.models.artifact_hashes
    _require(type(pairs) is tuple)
    hashes = {}
    for pair in pairs:
        _require(type(pair) is tuple and len(pair) == 2)
        name, digest = pair
        _require(type(name) is str and name not in hashes)
        hashes[name] = digest
    return {"artifact_hashes": hashes, "thresholds": _thresholds(session)}


def _scorer(session, workload):
    if workload != "shift_period":
        return session.scorer
    return LiveMonitor(
        session.scorer,
        stage1_model=session.models.cascade.stage1_model,
        gmm=session.models.gmm,
        boundary=session.models.monitor_boundary,
    )


@contextmanager
def _owner(binding, paths, inputs, role_context, retain):
    with CleanupStack() as cleanup:
        context = open_bound_session(binding, paths)
        cleanup.push(context.__exit__)
        session = context.__enter__()
        content = build_service_role(inputs, role_context, primary=_primary(session))
        scorer = _scorer(session, inputs.cell.workload)
        retain("service-role.json", content)
        yield scorer


def create_operational_app(binding, paths, inputs, *, role_context, retain):
    """Create a lazy service whose owner joins loaded identity before readiness."""
    _validate(binding, paths, inputs, role_context, retain)

    def factory():
        return _owner(binding, paths, inputs, role_context, retain)

    if inputs.cell.workload == "shift_period":
        plan = ShiftPlan(inputs.manifest_sha256, inputs.cell.run_index, inputs.requests)
        return create_shift_app(factory, plan)
    return create_app(factory, workload=inputs.cell.workload)
