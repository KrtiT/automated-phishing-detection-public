"""Invented series frames and no-op children, never study execution authority."""

import importlib
import importlib.util
import os
import sys

from automated_phishing_detection._process_support import command_hash


def api():
    name = "automated_phishing_detection._study_series_admission"
    assert importlib.util.find_spec(name), "missing series admission transport"
    return importlib.import_module(name)


def command():
    return sys.executable, "-c", "pass"


def frame(arguments=None, role="service", **changes):
    values = dict(
        role=role,
        profile_sha256="1" * 64,
        envelope_sha256="2" * 64,
        parent_pid=os.getpid(),
        command_sha256=command_hash(arguments or command()),
        series_reservation_sha256="3" * 64,
        segment_reservation_sha256="4" * 64,
        origin_reservation_sha256="5" * 64,
        history_index_sha256="6" * 64,
        intent_sha256="7" * 64,
        predecessor_sha256="8" * 64,
        accepted_inputs_sha256="9" * 64,
        cell_binding_sha256="a" * 64,
        segment_ordinal=2,
        cell_ordinal=73,
    )
    return api().SeriesAdmissionFrame(**(values | changes))


def child_command(role="service", suffix=""):
    program = (
        "import os, sys\n"
        "from automated_phishing_detection._study_series_admission import "
        "consume_series_admission\n"
        "arguments=(sys.executable, '-c', sys.argv[1], sys.argv[1])\n"
        f"admission=consume_series_admission({role!r}, arguments)\n"
        "admission.check()\n" + suffix + "\nadmission.close()\n"
    )
    return sys.executable, "-c", program, program
