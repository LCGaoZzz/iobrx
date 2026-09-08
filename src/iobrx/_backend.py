"""Optional native acceleration and explicit backend selection."""
from __future__ import annotations

import importlib
import os
import sys

_native = None
_native_error = None
if os.environ.get("IOBRX_DISABLE_RUST", "").lower() in {"1", "true", "yes"}:
    _native_error = "disabled by IOBRX_DISABLE_RUST"
else:
    try:
        _native = importlib.import_module("iobrx._rust")
    except (ImportError, OSError) as exc:
        _native_error = f"native extension unavailable ({type(exc).__name__})"

if _native is not None:
    sys.modules.setdefault("iobrx_rust", _native)


def select_backend(backend: str) -> bool:
    """Return whether native kernels may be used, without hiding runtime errors."""
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust" and _native is None:
        raise RuntimeError(f"Rust backend requested but {_native_error}")
    return backend != "python" and _native is not None


def backend_info() -> dict:
    """Report acceleration availability; no AVX-512 hardware is required.

    Native CIBERSORT additionally needs the bundled NumPy/SciPy OpenBLAS
    symbols. Other BLAS layouts use the original IOBRpy solver automatically.
    """
    from iobrx._sites import find_bundled_openblas
    scipy_blas, numpy_blas = find_bundled_openblas()
    return {
        "native_available": _native is not None,
        "native_reason": _native_error,
        "sorting": "numpy CPU dispatch",
        "requires_avx512": False,
        "bundled_openblas_found": bool(scipy_blas and numpy_blas),
    }
