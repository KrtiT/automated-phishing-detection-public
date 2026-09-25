"""One inherited-loopback operational service; no executable profile is adopted."""

import os

from . import _operational_child_context as context
from . import _study_preparation_files as files
from .operational_runtime import create_operational_app
from .operational_service import serve_service


def _notify_ready(descriptor):
    context.require(os.write(descriptor, b"ready\n") == 6)


async def _run_bound_service(
    binding,
    profile,
    *,
    accepted_inputs_directory,
    cell_input_directory,
    expected_binding_sha256,
    artifacts,
):
    with context.held_child(
        binding,
        profile,
        "service",
        accepted_inputs_directory=accepted_inputs_directory,
        cell_input_directory=cell_input_directory,
        expected_binding_sha256=expected_binding_sha256,
    ) as held:
        await _serve(held, binding, artifacts)


async def _serve(held, binding, artifacts):
    listener, stop_fd, ready_fd = held.handles

    def retain(name, content):
        held.retain(name, content)
        if name == "service-ready.json":
            files.deferred(_notify_ready, ready_fd)

    app = create_operational_app(
        binding, artifacts, held.inputs, role_context=held.role_context, retain=retain
    )
    await serve_service(app, listener, stop_fd, retain=retain)


async def run_operational_service(
    root,
    *,
    expected_revision,
    expected_contract_sha256,
    expected_operational_profile_sha256,
    accepted_inputs_directory,
    cell_input_directory,
    expected_binding_sha256,
    artifacts,
):
    binding, profile = context.public_context(
        root,
        expected_revision,
        expected_contract_sha256,
        expected_operational_profile_sha256,
    )
    await _run_bound_service(
        binding,
        profile,
        accepted_inputs_directory=accepted_inputs_directory,
        cell_input_directory=cell_input_directory,
        expected_binding_sha256=expected_binding_sha256,
        artifacts=artifacts,
    )
