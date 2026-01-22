from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import burst_test  # noqa: E402


def test_expand_payloads_truncates_when_total_less():
    payloads = [{"features": [1], "context": None}, {"features": [2], "context": None}]
    result = burst_test.expand_payloads(payloads, 1)
    assert len(result) == 1
    assert result[0]["features"] == [1]


def test_expand_payloads_repeats_to_meet_total():
    payloads = [{"features": [1], "context": None}]
    result = burst_test.expand_payloads(payloads, 3)
    assert len(result) == 3
    assert all(entry == payloads[0] for entry in result)
