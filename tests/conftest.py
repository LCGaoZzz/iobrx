"""Shared pytest configuration.

The official parity gates (``tests/test_parity_official.py``) carry the
``full`` marker and need the official data (``$IOBRX_TESTDATA`` or a mirror
download); ``pyproject.toml`` therefore deselects them by default so plain
``pytest -q`` is green on a fresh clone. This conftest registers the
``--run-full`` flag documented in that file as an explicit opt-in alias::

    pytest -q -m full          # marker form
    pytest -q --run-full       # flag form (this conftest)
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-full",
        action="store_true",
        default=False,
        help="run the official-data parity gates (marker 'full'; needs "
             "IOBRX_TESTDATA or a reachable mirror)",
    )


def pytest_configure(config):
    if config.getoption("--run-full"):
        # overrides the default addopts "-m 'not full'" deselect
        config.option.markexpr = "full"
