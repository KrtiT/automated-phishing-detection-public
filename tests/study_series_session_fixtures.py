"""Invented host conditions and processes, never actual research inputs."""

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace


def api(suffix):
    name = "automated_phishing_detection." + suffix
    assert find_spec(name), "missing fixed series session composition"
    return import_module(name)


def observation(**changes):
    return {
        "observed_at": "2026-10-01T00:00:00+00:00",
        "battery": "Now drawing from 'AC Power'",
        "power_settings": "AC Power:\n powermode 0\n lowpowermode 0\n",
        "thermal": "ThermalWarningLevel=1",
        "assertions": "\n".join(
            "pid 999992(caffeinate): " + kind
            for kind in (
                "PreventSystemSleep",
                "PreventUserIdleSystemSleep",
                "PreventUserIdleDisplaySleep",
            )
        ),
        "competing_known_workload_pids": [],
    } | changes


def inhibitor(returncode=None):
    return SimpleNamespace(pid=999992, poll=lambda: returncode)


class Process:
    def __init__(self, returncode=None):
        self.pid, self.returncode = 999991, returncode
        self.signals, self.waits = [], []

    def poll(self):
        return self.returncode

    def send_signal(self, selected):
        self.signals.append(selected)

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.returncode = 130 if self.signals else 0
        return self.returncode
