"""Authenticate sampled temporal facts, never physical eligibility or authority.

The next-child witness is conditional on trusted original monotonic admission
and sequential supervision. The witness image must match the root's image;
this relies on the pinned runner admitting every Python child, unlike probes.
Captures are non-atomic; their stamps precede power
and process observations. A subsequent clean capture bounds prefix acceptance,
not its exact time. Continuous AC, causal attribution and science remain separate.
"""

from dataclasses import dataclass

from . import _operational_input_schema as schema
from . import _stopped_study_host as host
from . import _stopped_study_timeline_records as records
from .stopped_study_authorization import verify_stopped_study_authorization


@dataclass(frozen=True)
class StoppedStudyTimeline:
    reservation_sha256: str
    accepted_ordinals: tuple[int, ...]
    stopped_ordinal: int
    observation_sha256: tuple[tuple[str, str], ...]
    next_service_sample_started_at: str
    prefix_completed_before_sample_at: str
    last_clean_sample_started_at: str
    first_ac_absence_sample_started_at: str
    root_exit_recorded_at: str
    sample_count: int
    maximum_sample_start_gap_seconds: float
    establishes_continuous_power: bool = False
    samples_are_atomic: bool = False
    authorizes_execution: bool = False


def _next_service(authority):
    ledger = schema.loads(authority.accounting_bytes)["authorization_ledger"]
    index = 2 + 2 * len(authority.accepted_ordinals)
    schema.require(len(ledger["admissions"]) > index)
    entry = ledger["admissions"][index]
    schema.require(entry["role"] == "service" and entry["accepted"] is False)
    process_pid = host.pid(entry["launched_pid"])
    schema.require(process_pid != authority.parent_pid)
    schema.require(
        all(
            previous["launched_pid"] != process_pid
            for previous in ledger["admissions"][:index]
        )
    )
    return process_pid


def _samples(samples, launch, next_pid):
    previous = host.timestamp(launch["launched_at"])
    witness = upper_bound = last_clean = first_absence = None
    for sample in samples:
        host.sample(sample)
        observed_at = host.timestamp(sample["observed_at"])
        schema.require(observed_at > previous)
        previous = observed_at
        ac_present = host.ac_power(sample)
        if first_absence is None and not ac_present:
            first_absence = sample["observed_at"]
        if first_absence is None:
            host.clean(sample, launch["caffeinate_pid"])
            seen, images = host.owned_processes(sample, launch)
            last_clean = sample["observed_at"]
            if witness is not None and upper_bound is None:
                upper_bound = sample["observed_at"]
            if (
                seen.get(next_pid) == launch["root_pid"]
                and witness is None
                and images[next_pid] == images[launch["root_pid"]]
            ):
                witness = sample["observed_at"]
    schema.require(upper_bound is not None and first_absence is not None)
    return witness, upper_bound, last_clean, first_absence


def _maximum_gap(values):
    samples = (
        values["pre.json"]["initial"],
        values["pre.json"],
        *values["conditions.jsonl"],
        values["post.json"],
    )
    times = [host.timestamp(value["observed_at"]) for value in samples]
    return max(
        (after - before).total_seconds() for before, after in zip(times, times[1:])
    )


def _verify(authority, payloads, expected, supervisor):
    values, hashes = records.authenticate(payloads, expected)
    launch = values["launch.json"]
    records.launch_record(launch, authority, supervisor)
    records.pre_record(values["pre.json"], launch, authority, supervisor)
    samples = values["conditions.jsonl"]
    next_pid = _next_service(authority)
    schema.require(next_pid not in {launch["supervisor_pid"], launch["caffeinate_pid"]})
    witness, bound, clean, absence = _samples(samples, launch, next_pid)
    records.post_record(values["post.json"], launch, samples[-1])
    records.cleanup_record(values["sleep-cleanup.json"], launch, values["post.json"])
    return StoppedStudyTimeline(
        authority.reservation_sha256,
        authority.accepted_ordinals,
        authority.stopped_ordinal,
        hashes,
        witness,
        bound,
        clean,
        absence,
        values["post.json"]["ended_at"],
        len(samples),
        _maximum_gap(values),
    )


def verify_stopped_study_timeline(
    snapshot,
    observation_payloads,
    *,
    expected_profile_sha256,
    expected_envelope_sha256,
    expected_snapshot_sha256,
    expected_observation_sha256,
    expected_supervisor_sha256,
):
    """Reauthenticate original authority, then join independently pinned samples."""
    try:
        authority = verify_stopped_study_authorization(
            snapshot,
            expected_profile_sha256=expected_profile_sha256,
            expected_envelope_sha256=expected_envelope_sha256,
            expected_snapshot_sha256=expected_snapshot_sha256,
        )
        return _verify(
            authority,
            observation_payloads,
            expected_observation_sha256,
            expected_supervisor_sha256,
        )
    except Exception:
        raise ValueError("invalid_stopped_study_timeline") from None
