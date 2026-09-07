"""iobrx quickstart: end-to-end taste in under a minute.

Loads the official imvigor210 cohort, then runs CIBERSORT (LM22), signature
scoring (PCA + ssGSEA over ``signature_collection``) and MCP-counter, printing
per-step timings.

Data resolution for ``imvigor210_eset``:
  1. ``$IOBRX_TESTDATA/imvigor210_eset.parquet``
  2. ``$IOBRX_TESTDATA/imvigor210_eset.rda``        (pyreadr)
  3. download from the IOBR ``data-v1.0`` GitHub release and cache locally

Run::

    python examples/quickstart.py
    IOBRX_TESTDATA=/path/to/testdata python examples/quickstart.py
"""
from __future__ import annotations

import os
import tempfile
import time
from time import perf_counter

import pandas as pd

import iobrx

RELEASE = "https://github.com/IOBR/IOBR/releases/download/data-v1.0"


def load_imvigor() -> pd.DataFrame:
    env = os.environ.get("IOBRX_TESTDATA")
    if env:
        pq = os.path.join(env, "imvigor210_eset.parquet")
        if os.path.exists(pq):
            print(f"[data] {pq}")
            return pd.read_parquet(pq)
        rda = os.path.join(env, "imvigor210_eset.rda")
        if os.path.exists(rda):
            print(f"[data] {rda}")
            import pyreadr

            return next(iter(pyreadr.read_r(rda).values()))
    cache = os.path.join(tempfile.gettempdir(), "iobrx_quickstart")
    pq = os.path.join(cache, "imvigor210_eset.parquet")
    if os.path.exists(pq):
        print(f"[data] {pq} (cached)")
        return pd.read_parquet(pq)
    import urllib.request

    import pyreadr

    os.makedirs(cache, exist_ok=True)
    rda = os.path.join(cache, "imvigor210_eset.rda")
    print(f"[data] downloading {RELEASE}/imvigor210_eset.rda ...")
    urllib.request.urlretrieve(f"{RELEASE}/imvigor210_eset.rda", rda)
    df = next(iter(pyreadr.read_r(rda).values()))
    df.to_parquet(pq)
    print(f"[data] cached at {pq}")
    return df


def main() -> None:
    print(f"iobrx {iobrx.__version__}  (default n_threads={iobrx.get_threads()})\n")
    eset = load_imvigor()
    print(f"eset: {eset.shape[0]} genes x {eset.shape[1]} samples\n")

    steps = []

    t = perf_counter()
    cib = iobrx.cibersort(eset.copy(), perm=100, QN=True)
    steps.append(("cibersort (LM22, perm=100)", perf_counter() - t, cib))

    t = perf_counter()
    pca = iobrx.calculate_sig_score(eset.copy(), "signature_collection", method="pca")
    steps.append(("sig score: PCA", perf_counter() - t, pca))

    t = perf_counter()
    ssg = iobrx.calculate_sig_score(eset.copy(), "signature_collection", method="ssgsea")
    steps.append(("sig score: ssGSEA", perf_counter() - t, ssg))

    t = perf_counter()
    mcp = iobrx.mcpcounter(eset.copy(), features_type="HUGO_symbols")
    steps.append(("MCP-counter", perf_counter() - t, mcp))

    print("step                                       time      output")
    print("-" * 74)
    for name, dt, out in steps:
        print(f"{name:<42s} {dt:7.3f}s   {out.shape[0]} x {out.shape[1]}")

    print("\nCIBERSORT head (first 4 cells x 3 samples):")
    print(cib.iloc[:3, :4].to_string())
    print("\nssGSEA TMEscore head:")
    cols = [c for c in ("ID", "TMEscore_CIR", "TMEscore_plus") if c in ssg.columns]
    print(ssg[cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
