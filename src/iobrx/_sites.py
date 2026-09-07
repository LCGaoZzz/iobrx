"""Locate the *bundled* OpenBLAS shared libraries shipped inside the installed
numpy / scipy wheels.

The bit-exactness contract of ``iobrx`` requires that the Rust kernels call
the very same BLAS/LAPACK entry points (``scipy_ddot_``, ``scipy_dgesdd_`` in
scipy's LP64 OpenBLAS; ``scipy_cblas_dgemv64_`` in numpy's ILP64 OpenBLAS)
that the installed numpy / scikit-learn themselves use. Wheels from PyPI ship
those libraries as private ``*.libs`` directories next to the package, so we
resolve them relative to the installed ``numpy`` / ``scipy`` packages rather
than hard-coding any environment path.

If numpy/scipy were NOT installed from wheels (e.g. conda, where OpenBLAS
lives in the environment's ``lib`` directory and the symbols are un-prefixed),
resolution fails and :func:`find_bundled_openblas` returns empty lists; the
callers raise a clear ``RuntimeError`` explaining that pip-installed
numpy/scipy wheels are required.
"""
from __future__ import annotations

import glob
import os

__all__ = ["site_packages", "find_bundled_openblas", "scipy_openblas", "numpy_openblas64"]


def site_packages() -> str:
    """Directory holding the installed ``numpy`` / ``scipy`` (site-packages)."""
    import numpy

    # numpy.__file__ = <site-packages>/numpy/__init__.py
    return os.path.dirname(os.path.dirname(os.path.abspath(numpy.__file__)))


def find_bundled_openblas() -> tuple[list[str], list[str]]:
    """Return ``(scipy_lp64_hits, numpy_ilp64_hits)`` bundled-openblas paths.

    ``scipy_lp64_hits``: scipy's private LP64 OpenBLAS (exports
    ``scipy_ddot_`` / ``scipy_dgesdd_``). ``numpy_ilp64_hits``: numpy's
    private ILP64 OpenBLAS (exports ``scipy_cblas_dgemv64_``).
    """
    site = site_packages()
    scipy_hits = sorted(
        glob.glob(os.path.join(site, "scipy.libs", "libscipy_openblas-*.so*"))
    )
    numpy_hits = sorted(
        glob.glob(os.path.join(site, "numpy.libs", "libscipy_openblas64*.so*"))
    )
    return scipy_hits, numpy_hits


def scipy_openblas() -> str:
    """Path to scipy's bundled LP64 OpenBLAS, or raise ``RuntimeError``."""
    hits, _ = find_bundled_openblas()
    if not hits:
        raise RuntimeError(
            "could not locate scipy's bundled OpenBLAS (scipy.libs); "
            "pip-installed scipy wheels are required for bit-exact kernels"
        )
    return hits[0]


def numpy_openblas64() -> str:
    """Path to numpy's bundled ILP64 OpenBLAS, or raise ``RuntimeError``."""
    _, hits = find_bundled_openblas()
    if not hits:
        raise RuntimeError(
            "could not locate numpy's bundled OpenBLAS (numpy.libs); "
            "pip-installed numpy wheels are required for bit-exact kernels"
        )
    return hits[0]
