"""Unchanged sampled host conditions for one operator-reserved continuation."""

import os
import re
import subprocess
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).isoformat()


def command(*arguments):
    return subprocess.run(
        arguments, check=True, text=True, capture_output=True, timeout=15
    ).stdout


def competing():
    rows = command("/bin/ps", "-axo", "pid=,ppid=,args=").splitlines()
    processes = [line.strip().split(None, 2) for line in rows]
    processes = [row for row in processes if len(row) == 3]
    owned = {os.getpid()}
    while True:
        expanded = owned | {
            int(pid) for pid, parent, unused in processes if int(parent) in owned
        }
        if expanded == owned:
            break
        owned = expanded
    pattern = re.compile(
        r"\b(pytest|torchrun|benchmark|train_[\w.-]+|fit_[\w.-]+)\b|\b(uv|python\S*|make)\s+(build|test|train|fit)\b"
    )
    return [
        int(pid)
        for pid, unused, arguments in processes
        if int(pid) not in owned and pattern.search(arguments)
    ]


def capture():
    return {
        "observed_at": stamp(),
        "battery": command("/usr/bin/pmset", "-g", "batt"),
        "power_settings": command("/usr/bin/pmset", "-g", "custom"),
        "thermal": command("/usr/bin/pmset", "-g", "therm"),
        "assertions": command("/usr/bin/pmset", "-g", "assertions"),
        "processes": command("/bin/ps", "-axo", "pid,ppid,pcpu,pmem,comm"),
        "competing_known_workload_pids": competing(),
        "interference_scope": "Known-command detection plus retained process observations; operator reservation remains required.",
    }


def ac_settings(value):
    lines = value.splitlines()
    if lines.count("AC Power:") != 1:
        return None
    selected = []
    for line in lines[lines.index("AC Power:") + 1 :]:
        if line and not line[0].isspace():
            break
        selected.append(line)
    return "\n".join(selected)


def violation(observation):
    if "Now drawing from 'AC Power'" not in observation["battery"]:
        return "ac_power_absent"
    settings = ac_settings(observation["power_settings"])
    if settings is None:
        return "ac_energy_settings_unavailable"
    if re.findall(r"^\s*powermode\s+([0-9]+)\s*$", settings, re.MULTILINE) != ["0"]:
        return "automatic_energy_mode_not_confirmed"
    if re.search(r"^\s*lowpowermode\s+[1-9]", settings, re.MULTILINE):
        return "low_power_enabled"
    if observation["competing_known_workload_pids"]:
        return "observed_competing_workload"
    return None


def inhibition(observation, inhibitor):
    if inhibitor.poll() is not None:
        return "owned_sleep_inhibitor_exited"
    for kind in (
        "PreventSystemSleep",
        "PreventUserIdleSystemSleep",
        "PreventUserIdleDisplaySleep",
    ):
        pattern = rf"pid\s+{inhibitor.pid}\(caffeinate\):[^\n]*\b{kind}\b"
        if not re.search(pattern, observation["assertions"]):
            return "owned_sleep_assertions_unavailable"
    return None
