import importlib
import os
import sys

from automated_phishing_detection._process_support import command_hash


def module():
    name = "automated_phishing_detection._study_admission"
    assert importlib.util.find_spec(name) is not None, "missing study admission"
    return importlib.import_module(name)


def command(program="pass"):
    return sys.executable, "-c", program


def frame(arguments=None, role="internal", **changes):
    values = dict(
        role=role,
        profile_sha256="1" * 64,
        envelope_sha256="2" * 64,
        parent_pid=os.getpid(),
        command_sha256=command_hash(arguments or command()),
        root_reservation_sha256="3" * 64,
        intent_sha256="4" * 64,
        barrier_sha256="5" * 64,
        preparation_reservation_sha256="6" * 64,
        preparation_completion_sha256="7" * 64,
        predecessor_sha256=None if role == "internal" else "8" * 64,
        accepted_inputs_sha256="9" * 64 if role in ("service", "client") else None,
        cell_binding_sha256="a" * 64 if role in ("service", "client") else None,
    )
    values.update(changes)
    return module().AdmissionFrame(**values)


def child_program(role="internal", suffix=""):
    return (
        "import os, sys\n"
        "from automated_phishing_detection._study_admission import "
        "consume_child_admission\n"
        "arguments=(sys.executable, '-c', sys.argv[1], sys.argv[1])\n"
        f"admission=consume_child_admission({role!r}, arguments)\n"
        "admission.check()\n" + suffix + "\nadmission.close()\n"
    )


def child_command(role="internal", suffix=""):
    program = child_program(role, suffix)
    return (*command(program), program)
