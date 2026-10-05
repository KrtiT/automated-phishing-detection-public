# Frozen-source regression verification

Recorded October 1, 2026 after the broad regression command completed. Frozen
source remains the clean detached commit
`ef8ba5f0b357cf3dd60c4d663e6297d13334460c`. No measurement code, models, thresholds,
execution manifest or scientific specification changed during this verification.

## Actual outcomes

- Broad run: **13,016 passed, 3 failed, 2 skipped**, exit 1, 3,874.59 seconds.
  This is not a green full-suite run.
- Targeted reproduction of the unchanged interruption test: **3 passed, 3 failed**,
  exit 1. All failures are the HTTP role of
  `test_actual_sigint_retains_exact_replay_progress`, for no cleanup exception,
  cleanup OSError, and a later cleanup KeyboardInterrupt.
- Each failure occurs before the intended interruption phase: the test's local
  `Client` double lacks `aclose`, which the frozen HTTP client context registers
  with AsyncExitStack. Real `httpx.AsyncClient` provides this method.
- A separately retained copy of that test adds only a no-op async `aclose`
  method to the double. Its parameterization, real SIGINT injection and exception
  identity/progress assertions are unchanged. All **six cases pass**, exit 0.
  Three already-passing shift cases are included; these counts must not be added
  to the broad-suite count as six new distinct tests.
- The first supplemental command selected the enclosing unrelated application's pytest config
  and failed before collection because that config referenced absent `urllib3`.
  The retained successful command explicitly selects the academic checkout's
  `pyproject.toml`. This configuration correction is not a study retry.

## Reproduction commands

Run from `.context/gwu_working/study-followup-development-20261001`:

```sh
PYTHONPATH="$PWD/src:$PWD/tests" ../http-integration-env/bin/python -m pytest -q tests/test_operational_child_interruptions.py
PYTHONPATH="$PWD/src:$PWD/tests" ../http-integration-env/bin/python -m pytest -c pyproject.toml -q ../../dissertation/followup-20261001/verification-tests/test_operational_child_interruptions.py
```

The original frozen test remains unchanged. The supplemental file is a fixture
compatibility correction, not replacement evidence for a scientific outcome.
No protected model was fitted or used for predictions by these tests. No service
measurement attempt was created. Independent external code review remains
unavailable and is not claimed.

## Retained SHA-256 identities

| Artifact | SHA-256 |
|---|---|
| `followup-full-suite.txt` | `9e4b8b533ab6efea55d82ef5077e5a54c824174e7fe0c78c1c677325baba91d6` |
| `interrupt-fixture-reproduction.txt` | `42e0bf8630b6c36d3e33c9b81017ab483b4e10b6668dca2a3e7aa69a99cc181d` |
| `interrupt-fixture-correction.txt` (configuration error) | `cbe761051efcde1e11cbca2850d908856827431838f11a33a7b3ccec2382289c` |
| `interrupt-fixture-correction-2.txt` | `83d38d292255b116bfa795aeeaa5612c941d418c80d43572e384ffaf8da6167f` |
| Original frozen test | `dc7f7464290e967275537407df60fb23597fa90b4f3c662802aff70580553709` |
| `verification-tests/test_operational_child_interruptions.py` | `decef6970aa47fcc0d327827f6425849f6fc52fd52227f94f6619b17dd128ec2` |

The frozen-source full-suite failure record is retained alongside the supplemental
verification rather than overwritten, suppressed or described as a complete rerun.
