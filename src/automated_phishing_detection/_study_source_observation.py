"""Issue one fixed study source admission inside the original owned lifecycle."""


def observe_admitted(command, admissions, role, **extra):
    from .owned_worker import _observe_study_worker

    admission = admissions.issue(role, command, **extra)
    return _observe_study_worker(command, admission)
