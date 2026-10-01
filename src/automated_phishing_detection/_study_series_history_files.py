"""Bounded no-follow retained reads with hash and lifetime identity checks."""

import os
import stat
from hashlib import sha256
from pathlib import Path

from . import _study_execution_schema as schema
from . import execution_receipt as receipt
from ._exception_cleanup import preserve_cleanup
from ._study_preparation_files import state


def _capture(directory, name):
    metadata = receipt._entry(directory, name)
    schema.require(metadata is not None)
    schema.require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1)
    return state(metadata), receipt._identity(os.fstat(directory.descriptor))


def _stream(descriptor, expected, retain):
    hasher, chunks = sha256(), []
    with os.fdopen(descriptor, "rb", closefd=False) as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
            if retain:
                chunks.append(chunk)
    schema.require(hasher.hexdigest() == expected)
    return b"".join(chunks) if retain else None


def _read(path, expected, retain):
    with receipt._directory(path.parent) as directory:
        initial = _capture(directory, path.name)
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory.descriptor,
        )
        with preserve_cleanup(lambda: os.close(descriptor)):
            schema.require(state(os.fstat(descriptor)) == initial[0])
            content = _stream(descriptor, expected, retain)
            schema.require(state(os.fstat(descriptor)) == initial[0])
            schema.require(_capture(directory, path.name) == initial)
            directory.check()
    return content, initial


def _tree(path):
    result = set()
    with receipt._directory(path) as directory:
        names = set(os.listdir(directory.descriptor))
        for name in names:
            metadata = receipt._entry(directory, name)
            schema.require(metadata is not None)
            selected = path / name
            if stat.S_ISDIR(metadata.st_mode):
                result.update(_tree(selected))
            else:
                schema.require(stat.S_ISREG(metadata.st_mode))
                result.add(str(selected))
        schema.require(set(os.listdir(directory.descriptor)) == names)
        directory.check()
    return frozenset(result)


class HistoryFiles:
    """Captured facts only; this object never grants access or execution rights."""

    def __init__(self):
        self.files, self.trees = {}, {}

    def read(self, path, expected, *, retain=True):
        try:
            path = schema.lexical_path(str(path))
            schema.digest(expected)
            schema.require(type(retain) is bool)
            content, captured = _read(path, expected, retain)
            prior = self.files.get(path)
            schema.require(prior is None or prior == (expected, captured))
            self.files[path] = expected, captured
            return content
        except Exception:
            raise ValueError("invalid_series_historical_file") from None

    def tree(self, root, paths):
        root = schema.lexical_path(str(root))
        expected = frozenset(paths)
        schema.require(expected and len(expected) == len(paths))
        schema.require(all(Path(path).is_relative_to(root) for path in expected))
        schema.require(_tree(root) == expected)
        schema.require(root not in self.trees or self.trees[root] == expected)
        self.trees[root] = expected

    def check(self):
        try:
            for path, (unused, initial) in self.files.items():
                with receipt._directory(path.parent) as directory:
                    schema.require(_capture(directory, path.name) == initial)
                    directory.check()
            for root, expected in self.trees.items():
                schema.require(_tree(root) == expected)
        except Exception:
            raise ValueError("changed_series_historical_file") from None
