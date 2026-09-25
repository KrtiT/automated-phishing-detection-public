"""Attempt cleanup without replacing the first interruption object."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager


@contextmanager
def preserve_cleanup(cleanup: Callable[[], object]) -> Iterator[None]:
    original = None
    try:
        yield
    except BaseException as error:
        original = error
        raise
    finally:
        try:
            cleanup()
        except BaseException:
            if original is not None and not isinstance(original, Exception):
                raise original from None
            raise


def _preserving_exit(callback):
    def leave(exception_type, error, traceback):
        try:
            suppressed = callback(exception_type, error, traceback)
            if error is not None and not isinstance(error, Exception):
                return False
            return bool(suppressed)
        except BaseException:
            if error is not None and not isinstance(error, Exception):
                raise error from None
            raise

    return leave


class CleanupStack(ExitStack):
    """Keep standard LIFO cleanup, preserving interruption precedence per exit."""

    def _push_exit_callback(self, callback, is_sync=True):
        super()._push_exit_callback(_preserving_exit(callback), is_sync)
