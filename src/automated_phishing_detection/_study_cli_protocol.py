"""Dependency-free whole-study command declaration; no execution authority."""

SCRIPT = "scripts/run_study.py"
IDENTITY_ARGUMENTS = (
    "expected-revision",
    "expected-contract-sha256",
    "expected-operational-profile-sha256",
)
PATH_ARGUMENTS = (
    "repo-root",
    "source-csv",
    "suffix-rules",
    "archive",
    "preparation-attempt",
    "internal-attempt",
    "internal-public-summary",
    "external-attempt",
    "external-public-summary",
    "attempt",
    "public-summary",
    "accepted-inputs-dir",
    "cells-dir",
    "length-only",
    "logistic-l1",
    "transformer-bundle",
    "gmm",
    "formatting",
    "permutation-42",
    "permutation-43",
    "permutation-44",
    "permutation-45",
    "permutation-46",
    "random-forest",
    "seed-43-weights",
    "seed-44-weights",
    "seed-45-weights",
    "seed-46-weights",
    "training-reference",
    "validation-audit",
)
ARGUMENTS = tuple(
    f"--{name}"
    for name in (PATH_ARGUMENTS[0], *IDENTITY_ARGUMENTS, *PATH_ARGUMENTS[1:])
)
