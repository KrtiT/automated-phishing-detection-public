"""Attach only new series admissions to unchanged owned process lifecycle."""

from ._operational_process_children import OwnedChildren
from ._process_support import _defer_interrupt
from ._study_series_admission_parent import validate_series_launch


class SeriesOwnedChildren(OwnedChildren):
    def _launch_options(self, role, command):
        descriptors = (
            (self.listener.fileno(), self.stop_read, self.ready_write)
            if role == "service"
            else ()
        )
        environment = self._environment(role)
        with _defer_interrupt():
            admission = self.study_admissions(role, command)
            validate_series_launch(admission, role, command)
            self.admissions[role] = self.stack.enter_context(admission)
        descriptors += (admission.read_fd,)
        environment.update(admission.environment)
        return descriptors, environment
