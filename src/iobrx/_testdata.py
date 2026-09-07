"""Official-data resolution (README "Quickstart", official-data block).

Resolves the official IOBR validation frames (``imvigor210_eset``,
``eset_stad``, ``anno_grch38``, ...) used by the quickstart, the examples and
the ``full``-marker tests:

  1. the ``data_dir`` argument or ``$IOBRX_TESTDATA``: ``<name>.parquet``,
     then ``<name>.rda`` (first object, via pyreadr);
  2. a local cache under ``$TMPDIR/iobrx_official_data`` (``<name>.parquet``);
  3. download ``<name>.rda`` from the IOBR ``data-v1.0`` release through a
     mirror list — github.com direct first, then the ``gh-proxy.com`` and
     ``ghproxy.net`` prefix mirrors of the same release URL — and cache the
     converted parquet locally.

When every mirror is unreachable, :func:`load_official` raises
:class:`OfficialDataUnavailable` with the remediation spelled out instead of
a raw urllib traceback. The mirror list can be overridden for offline
testing with ``$IOBRX_TESTDATA_MIRRORS`` (comma-separated base URLs).
"""
from __future__ import annotations

import os
import tempfile

#: Mirror bases tried in order for the IOBR ``data-v1.0`` release assets.
MIRRORS: tuple[str, ...] = (
    "https://github.com/IOBR/IOBR/releases/download/data-v1.0",
    "https://gh-proxy.com/https://github.com/IOBR/IOBR/releases/download/data-v1.0",
    "https://ghproxy.net/https://github.com/IOBR/IOBR/releases/download/data-v1.0",
)

_TIMEOUT_S = 20.0
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "iobrx_official_data")


class OfficialDataUnavailable(RuntimeError):
    """The official frame could not be resolved locally nor downloaded."""


def _read_rda(path: str):
    import pyreadr

    return next(iter(pyreadr.read_r(path).values()))


def load_official(name: str, data_dir: str | None = None, verbose: bool = True):
    """Load an official IOBR validation frame by name.

    Parameters
    ----------
    name : str
        Frame name without extension, e.g. ``"imvigor210_eset"``. Resolution
        order: ``data_dir``/``$IOBRX_TESTDATA`` (``.parquet`` then ``.rda``),
        the local cache, then a mirror download (see module docstring).
    data_dir : str or None, optional
        Directory holding ``<name>.parquet`` / ``<name>.rda``; defaults to
        ``$IOBRX_TESTDATA``.
    verbose : bool, default True
        Print which file was used / which mirror is being tried.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    OfficialDataUnavailable
        No local copy exists and every mirror failed (offline).
    """
    import pandas as pd

    base = data_dir or os.environ.get("IOBRX_TESTDATA")
    if base:
        pq = os.path.join(base, f"{name}.parquet")
        if os.path.exists(pq):
            if verbose:
                print(f"[data] {pq}")
            return pd.read_parquet(pq)
        rda = os.path.join(base, f"{name}.rda")
        if os.path.exists(rda):
            if verbose:
                print(f"[data] {rda}")
            return _read_rda(rda)

    pq = os.path.join(_CACHE_DIR, f"{name}.parquet")
    if os.path.exists(pq):
        if verbose:
            print(f"[data] {pq} (cached)")
        return pd.read_parquet(pq)

    # 3. download through the mirror list
    import urllib.request

    override = os.environ.get("IOBRX_TESTDATA_MIRRORS")
    bases = (tuple(m.strip() for m in override.split(",") if m.strip())
             if override else MIRRORS)
    failures: list[str] = []
    tmp = os.path.join(_CACHE_DIR, f"{name}.rda.part")
    for url_base in bases:
        url = f"{url_base.rstrip('/')}/{name}.rda"
        try:
            if verbose:
                print(f"[data] downloading {url} ...")
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
                with open(tmp, "wb") as fh:
                    fh.write(resp.read())
            rda = os.path.join(_CACHE_DIR, f"{name}.rda")
            os.replace(tmp, rda)
            df = _read_rda(rda)
            df.to_parquet(pq)
            if verbose:
                print(f"[data] cached at {pq}")
            return df
        except Exception as exc:  # noqa: BLE001 - collect every mirror, then raise
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            try:
                os.remove(tmp)
            except OSError:
                pass
    raise OfficialDataUnavailable(
        f"could not resolve the official frame '{name}' locally and the "
        f"download failed on every mirror:\n  " + "\n  ".join(failures)
        + f"\nSet IOBRX_TESTDATA=/path/to/testdata (a directory holding "
        f"{name}.parquet or {name}.rda), or run the synthetic quickstart "
        "(README 'Quickstart', first block), which needs no data."
    )
