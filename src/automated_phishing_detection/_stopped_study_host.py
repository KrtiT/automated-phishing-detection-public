"""Pure checks of retained host samples, not continuous telemetry or live state."""

import math
import re
from datetime import datetime, timedelta

from . import _operational_input_schema as schema

FIELDS = frozenset(
    {
        "observed_at",
        "battery",
        "power_settings",
        "thermal",
        "assertions",
        "processes",
        "competing_known_workload_pids",
        "interference_scope",
    }
)
ASSERTIONS = (
    "PreventSystemSleep",
    "PreventUserIdleSystemSleep",
    "PreventUserIdleDisplaySleep",
)
SCOPE = "Known-command detection plus retained process observations; operator reservation remains required."


def timestamp(value):
    schema.require(type(value) is str)
    selected = datetime.fromisoformat(value)
    schema.require(selected.tzinfo is not None and selected.utcoffset() == timedelta(0))
    return selected


def pid(value):
    schema.require(type(value) is int and value > 0)
    return value


def processes(content):
    schema.require(type(content) is str)
    lines = content.splitlines()
    schema.require(
        lines and lines[0].split() == ["PID", "PPID", "%CPU", "%MEM", "COMM"]
    )
    parents, images = {}, {}
    for line in lines[1:]:
        fields = line.split(None, 4)
        schema.require(len(fields) == 5)
        schema.require(all(re.fullmatch(r"[0-9]+", item) for item in fields[:2]))
        process_pid, parent = map(int, fields[:2])
        schema.require(process_pid not in parents)
        schema.require(
            all(math.isfinite(float(item)) and float(item) >= 0 for item in fields[2:4])
        )
        parents[process_pid], images[process_pid] = parent, fields[4]
    return parents, images


def sample(value):
    schema.keys(value, FIELDS)
    timestamp(value["observed_at"])
    for key in FIELDS - {"competing_known_workload_pids"}:
        schema.require(type(value[key]) is str)
    competitors = value["competing_known_workload_pids"]
    schema.require(
        type(competitors) is list and len(set(competitors)) == len(competitors)
    )
    for process_pid in competitors:
        pid(process_pid)
    schema.require(value["interference_scope"] == SCOPE)
    processes(value["processes"])


def ac_power(value):
    selected = re.findall(
        r"^Now drawing from '([^']+)'\s*$", value["battery"], re.MULTILINE
    )
    schema.require(len(selected) == 1 and selected[0] in ("AC Power", "Battery Power"))
    return selected[0] == "AC Power"


def automatic(value):
    lines = value["power_settings"].splitlines()
    schema.require(lines.count("AC Power:") == 1)
    selected = []
    for line in lines[lines.index("AC Power:") + 1 :]:
        if line and not line[0].isspace():
            break
        selected.append(line)
    settings = "\n".join(selected)
    schema.require(
        re.findall(r"^\s*powermode\s+([0-9]+)\s*$", settings, re.MULTILINE) == ["0"]
    )
    low_power = re.findall(r"^\s*lowpowermode\s+(\S+)\s*$", settings, re.MULTILINE)
    schema.require(low_power in ([], ["0"]))


def owned_assertions(value, inhibitor):
    pid(inhibitor)
    for kind in ASSERTIONS:
        pattern = rf"pid\s+{inhibitor}\(caffeinate\):[^\n]*\b{kind}\b"
        schema.require(re.search(pattern, value["assertions"]) is not None)


def clean(value, inhibitor=None):
    sample(value)
    schema.require(ac_power(value))
    automatic(value)
    schema.require(not value["competing_known_workload_pids"])
    if inhibitor is not None:
        owned_assertions(value, inhibitor)


def owned_processes(value, launch, *, root=True):
    observed, images = processes(value["processes"])
    supervisor = launch["supervisor_pid"]
    schema.require(supervisor in observed)
    schema.require(observed.get(launch["caffeinate_pid"]) == supervisor)
    if root:
        schema.require(observed.get(launch["root_pid"]) == supervisor)
    seen = {launch["root_pid"], launch["caffeinate_pid"]}
    ancestor = supervisor
    while ancestor in observed and ancestor != 0:
        schema.require(ancestor not in seen)
        seen.add(ancestor)
        ancestor = observed[ancestor]
    return observed, images
