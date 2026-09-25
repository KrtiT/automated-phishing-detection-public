import signal

import pytest
from operational_input_signal_fixtures import (
    assert_closed,
    interrupt_fdopen,
    watch_open,
)
from study_root_retention_fixtures import api, append_all, complete, manager, root_case

from automated_phishing_detection import _study_root_records as records


@pytest.mark.parametrize(
    "selected",
    ["root", "reservation.json", "study-intent.json", "evidence", "public.json"],
)
@pytest.mark.parametrize("boundary", ["open", "close"])
def test_actual_sigint_owns_and_closes_every_acquired_descriptor(
    tmp_path, monkeypatch, selected, boundary
):
    module, case = api(), root_case(tmp_path)
    with monkeypatch.context() as guard:
        descriptors = watch_open(
            guard,
            selected,
            interrupt=boundary == "open",
            before_close=boundary == "close",
        )
        with pytest.raises(KeyboardInterrupt):
            with manager(module, case) as writer:
                append_all(writer, case)
                complete(writer, case)
    assert descriptors
    assert_closed(descriptors)
    assert (case.attempt.directory / "reservation.json").exists()


@pytest.mark.parametrize("phase", ["entry", "append", "publication"])
def test_actual_fdopen_signal_does_not_leak(tmp_path, monkeypatch, phase):
    module, case = api(), root_case(tmp_path)
    if phase == "entry":
        with monkeypatch.context() as guard:
            descriptors = interrupt_fdopen(guard)
            with pytest.raises(KeyboardInterrupt):
                with manager(module, case):
                    pytest.fail("signal must precede body")
    else:
        with pytest.raises(KeyboardInterrupt):
            with manager(module, case) as writer:
                if phase == "publication":
                    append_all(writer, case)
                with monkeypatch.context() as guard:
                    descriptors = interrupt_fdopen(guard)
                    if phase == "append":
                        writer.append(
                            "study-intent.json", case.contents["study-intent.json"]
                        )
                    else:
                        complete(writer, case)
    assert descriptors
    assert_closed(descriptors)


@pytest.mark.parametrize("phase", ["body", "append", "complete"])
def test_body_and_pure_validation_deliver_signal_immediately(
    tmp_path, monkeypatch, phase
):
    module, case = api(), root_case(tmp_path)
    reached = []

    def interrupt(*arguments, **keywords):
        signal.raise_signal(signal.SIGINT)
        reached.append("deferred")

    with pytest.raises(KeyboardInterrupt):
        with manager(module, case) as writer:
            if phase == "complete":
                append_all(writer, case)
                monkeypatch.setattr(records, "public_bytes", interrupt)
                complete(writer, case)
            elif phase == "append":
                monkeypatch.setattr(records, "decode", interrupt)
                writer.append("study-intent.json", case.contents["study-intent.json"])
            else:
                interrupt()
    assert reached == []
    assert not writer.publishing
