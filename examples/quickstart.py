"""iobrx quickstart: end-to-end taste in under a minute.

Loads the official imvigor210 cohort, then runs CIBERSORT (LM22), signature
scoring (PCA + ssGSEA over ``signature_collection``) and MCP-counter, printing
per-step timings.

Data resolution for ``imvigor210_eset`` (via :func:`iobrx.load_official`):
  1. ``$IOBRX_TESTDATA/imvigor210_eset.parquet`` / ``.rda``  (or a local cache)
  2. download from the IOBR ``data-v1.0`` release through a mirror list
     (github.com direct, gh-proxy.com, ghproxy.net — override with
     ``$IOBRX_TESTDATA_MIRRORS``); offline this raises
     ``iobrx.OfficialDataUnavailable`` with the remediation spelled out

No official data and no network? Run the synthetic quickstart in README.md
instead — it needs neither.

Run::

    python examples/quickstart.py
    IOBRX_TESTDATA=/path/to/testdata python examples/quickstart.py
"""
from __future__ import annotations

from time import perf_counter

import pandas as pd

import iobrx


def load_imvigor() -> pd.DataFrame:
    return iobrx.load_official("imvigor210_eset")


def main() -> None:
    print(f"iobrx {iobrx.__version__}  (default n_threads={iobrx.get_threads()})\n")
    try:
        eset = load_imvigor()
    except iobrx.OfficialDataUnavailable as exc:
        raise SystemExit(f"cannot load the official imvigor210 cohort:\n{exc}")
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
