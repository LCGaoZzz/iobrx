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


def pytest_collection_modifyitems(config, items):
    """Skip comparison modules when the original IOBRpy is not installed.

    iobrx 0.4.0 no longer requires IOBRpy: the bundled kernels run on the
    reference data shipped inside the wheel. The parity suites compare the
    accelerated paths against the ORIGINAL IOBRpy implementation and cannot
    run without it; skip them with an explicit reason instead of failing.
    CI installs the ``python`` extra, so nothing is skipped there.
    """
    import importlib.util
    if importlib.util.find_spec("iobrpy") is not None:
        return
    reason = "iobrpy (the original implementation) is not installed; parity comparison unavailable"
    for item in items:
        name = item.path.parts[-1]
        if name.startswith("test_parity") or name in {"test_extract_hla_read.py", "test_portability.py"}:
            item.add_marker(pytest.mark.skip(reason=reason))
