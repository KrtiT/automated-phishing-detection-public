"""One fixed operational client; public execution remains prospectively closed."""

from . import _operational_child_context as context
from ._prepared_failure_context import carry_failure_context
from .http_replay import replay_run
from .http_run_codec import encode_http_run
from .operational_role_records import build_client_role
from .shift_replay import replay_shift_run
from .shift_run_codec import encode_shift_run
from .shift_schema import ShiftPlan


async def _replay(held):
    inputs, cell = held.inputs, held.inputs.cell
    held.retain("client-role.json", build_client_role(inputs, held.role_context))
    if cell.workload == "shift_period":
        plan = ShiftPlan(inputs.manifest_sha256, cell.run_index, inputs.requests)
        run = await replay_shift_run(
            held.role_context.base_url, plan, retain=held.retain
        )
        return encode_shift_run(run)
    run = await replay_run(
        held.role_context.base_url,
        inputs.requests,
        manifest_sha256=inputs.manifest_sha256,
        prevalence_basis_points=cell.prevalence_basis_points,
        concurrency=cell.concurrency,
        run_index=cell.run_index,
        warmup_count=1000,
        workload=cell.workload,
        retain=held.retain,
    )
    return encode_http_run(run)


def _failed(held, error):
    progress = BaseException.__dict__["__dict__"].__get__(error).get("progress")
    try:
        if type(progress) is bytes:
            held.retain("client-failure.json", progress)
    except BaseException as later:
        carry_failure_context(later, error)
        if not isinstance(error, Exception):
            raise error from None
        raise
    raise error from None


async def _run_bound_client(
    binding,
    profile,
    *,
    accepted_inputs_directory,
    cell_input_directory,
    expected_binding_sha256,
):
    with context.held_child(
        binding,
        profile,
        "client",
        accepted_inputs_directory=accepted_inputs_directory,
        cell_input_directory=cell_input_directory,
        expected_binding_sha256=expected_binding_sha256,
    ) as held:
        try:
            content = await _replay(held)
            held.retain("run.json", content)
        except BaseException as error:
            _failed(held, error)


async def run_operational_client(
    root,
    *,
    expected_revision,
    expected_contract_sha256,
    expected_operational_profile_sha256,
    accepted_inputs_directory,
    cell_input_directory,
    expected_binding_sha256,
):
    binding, profile = context.public_context(
        root,
        expected_revision,
        expected_contract_sha256,
        expected_operational_profile_sha256,
    )
    await _run_bound_client(
        binding,
        profile,
        accepted_inputs_directory=accepted_inputs_directory,
        cell_input_directory=cell_input_directory,
        expected_binding_sha256=expected_binding_sha256,
    )
