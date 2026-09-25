"""Carry existing private failure associations without probing or inventing state."""

_NAMES = (
    "worker_failure",
    "external_failure",
    "source_internal",
    "progress",
    "preparation_progress",
    "operational_failure",
    "study_failure",
)


def carry_failure_context(error, original):
    """Best-effort missing-key copy never replaces the selected exception."""
    if original is None or error is original:
        return
    try:
        descriptor = BaseException.__dict__["__dict__"]
        source, target = descriptor.__get__(original), descriptor.__get__(error)
        for name in _NAMES:
            if dict.__contains__(source, name) and not dict.__contains__(target, name):
                dict.__setitem__(target, name, dict.__getitem__(source, name))
    except BaseException:
        pass
