"""Fixed operational records and commands; no scientific execution authority."""

PROTOCOL = "operational-cell-v1"
SERVICE_SCRIPT = "scripts/run_operational_service.py"
CLIENT_SCRIPT = "scripts/run_operational_client.py"
COMMON_ARGUMENTS = (
    "--repo-root",
    "--expected-revision",
    "--expected-contract-sha256",
    "--expected-operational-profile-sha256",
    "--accepted-inputs-dir",
    "--cell-input-dir",
    "--expected-binding-sha256",
)
SERVICE_ARGUMENTS = COMMON_ARGUMENTS + (
    "--length-only",
    "--logistic-l1",
    "--transformer-bundle",
    "--gmm",
)
WORKING_NAMES = (
    "reservation.json",
    "process-pair-intent.json",
    "service-intent.json",
    "service-started.json",
    "service-process.json",
    "client-intent.json",
    "client-started.json",
    "client-process.json",
    "service-role.json",
    "client-role.json",
    "service-ready.json",
    "service-stop.json",
    "service-cleanup.json",
    "warmup.json",
    "measured.json",
    "run.json",
    "process-pair.json",
)
PRIVATE_NAMES = WORKING_NAMES[1:]
SNAPSHOT_NAMES = (
    *(f"attempt/{name}" for name in WORKING_NAMES),
    "attempt/finalize.claim",
    "attempt/outcome.json",
    *(f"attempt/evidence/{name}" for name in PRIVATE_NAMES),
    "public-summary.json",
)
