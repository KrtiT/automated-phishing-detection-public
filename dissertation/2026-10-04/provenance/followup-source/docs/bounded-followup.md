# Bounded follow-up comparison

This extension follows the completed initial evaluation. It does not replace
the original hypotheses, evidence or standalone-runner authorization rules.
Only two changes are evaluated: scheme-neutral structural features and
worker-owned HTTP connections. No transformer retraining, GMM retuning,
test-time threshold selection or candidate search is part of this extension.

The external working evidence directory contains the immutable scientific
specification, dated implementation details, population admission, tests and
execution manifest. Do not commit research rows, model artifacts or private
predictions to this repository. The approved working context is passed explicitly:

```sh
PYTHONPATH="$PWD/src" python scripts/run_followup_detection.py \
  --context-root /absolute/path/to/working-context \
  --manifest /absolute/path/to/execution-manifest.json --phase fit
PYTHONPATH="$PWD/src" python scripts/run_followup_detection.py \
  --context-root /absolute/path/to/working-context \
  --manifest /absolute/path/to/execution-manifest.json --phase evaluate
PYTHONPATH="$PWD/src" python scripts/run_followup_service.py \
  --context-root /absolute/path/to/working-context \
  --manifest /absolute/path/to/execution-manifest.json
```

Use the exact interpreter recorded by the manifest, not an arbitrary `python`.
Execution requires the exact clean local commit, runtime, scientific specification,
dated clarification hashes and admitted population. The fit writes its frozen
models before new-benchmark scoring. Existing attempts are never overwritten.
Any stop requires examination of the preserved failure, not an automatic retry.

The service schedule has 80 arms: two workloads, two concurrency levels, ten paired
repetitions and two client topologies. Every arm uses a fresh child service,
1,000 warmup requests and 10,000 measured requests. The no-model control and
unchanged structural scorer use reserved synthetic URL strings, not a labeled
accuracy or prevalence workload. Final measurements require observed AC power,
normal reported thermal/performance state, owned sleep inhibition and no observed
competing test/build/training/benchmark job. Partial work is retained after a stop.

Service records use a distinct outer protocol. Nested legacy wire fields do not
turn these synthetic comparisons into original cascade or H3 evidence. The
reported exact-response requirement includes admission sequence; prediction-only
agreement is separately descriptive. All-request and failure-only latencies remain
visible alongside the primary success-only p95 and explicit error denominator.

Synthetic verification:

```sh
PYTHONPATH="$PWD/src:$PWD/tests" python -m pytest -q \
  tests/test_followup_detection_script.py tests/test_followup_metrics.py \
  tests/test_followup_population.py tests/test_followup_service.py \
  tests/test_transport_neutral_features.py tests/test_transport_neutral_model.py \
  tests/test_worker_connection_replay.py tests/test_http_replay.py \
  tests/test_http_run_codec.py tests/test_selective_service.py
```

Tests and development probes are not research measurements. Retain every
unfavorable comparison and original decision; neither a favorable result nor
institutional acceptance is guaranteed by this implementation.
