"""The official 11-stage IOBR workflow on the official data, via iobrx.

Stages (official order, official parameters):
  sig_pca / sig_zscore / sig_ssgsea / sig_integration  on imvigor210_eset
  anno_eset (Ensembl -> symbols)                       on eset_stad
  cibersort (LM22, perm=100, QN)                       on the symbol matrix
  epic (TRef) / quantiseq (TIL10, lsei) / mcpcounter (HUGO) / estimate (affy)
  count2tpm (Ensembl counts -> TPM)                    on eset_stad

Prints a per-stage timing table. ``--verify-parity`` additionally runs the
ORIGINAL iobrpy implementations side by side and asserts bit-equality
(CIBERSORT's unseeded P-value column excepted by design).

Data resolution: ``--testdata DIR`` or ``$IOBRX_TESTDATA`` (parquet preferred,
then .rda), falling back to the IOBR ``data-v1.0`` GitHub release.

Run::

    python examples/full_workflow.py                       # timings only
    python examples/full_workflow.py --verify-parity       # + bit-parity gates
    python examples/full_workflow.py --threads 64
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from time import perf_counter

import numpy as np
import pandas as pd

RELEASE = "https://github.com/IOBR/IOBR/releases/download/data-v1.0"
_CACHE = os.path.join(tempfile.gettempdir(), "iobrx_official_data")


def load(name: str, data_dir: str | None) -> pd.DataFrame:
    d = data_dir or os.environ.get("IOBRX_TESTDATA")
    if d:
        pq = os.path.join(d, f"{name}.parquet")
        if os.path.exists(pq):
            print(f"[data] {pq}")
            return pd.read_parquet(pq)
        rda = os.path.join(d, f"{name}.rda")
        if os.path.exists(rda):
            print(f"[data] {rda}")
            import pyreadr

            return next(iter(pyreadr.read_r(rda).values()))
    pq = os.path.join(_CACHE, f"{name}.parquet")
    if os.path.exists(pq):
        print(f"[data] {pq} (cached)")
        return pd.read_parquet(pq)
    import urllib.request

    import pyreadr

    os.makedirs(_CACHE, exist_ok=True)
    rda = os.path.join(_CACHE, f"{name}.rda")
    print(f"[data] downloading {RELEASE}/{name}.rda ...")
    urllib.request.urlretrieve(f"{RELEASE}/{name}.rda", rda)
    df = next(iter(pyreadr.read_r(rda).values()))
    df.to_parquet(pq)
    print(f"[data] cached at {pq}")
    return df


def num_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    num = b.select_dtypes(include=[np.number]).columns
    d = a[num].to_numpy(dtype=np.float64) - b[num].to_numpy(dtype=np.float64)
    return {
        "index_equal": list(a.index) == list(b.index),
        "columns_equal": list(a.columns) == list(b.columns),
        "numeric_cells": int(d.size),
        "cells_bit_identical": int((d == 0).sum()),
        "max_abs_diff": float(np.abs(d).max()) if d.size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--testdata", default=None, help="dir with official parquet/rda files")
    ap.add_argument("--threads", type=int, default=None,
                    help="cibersort n_threads (default: iobrx default, see set_threads)")
    ap.add_argument("--verify-parity", action="store_true",
                    help="also run ORIGINAL iobrpy and assert bit-equality")
    args = ap.parse_args()

    import iobrx

    n_threads = args.threads if args.threads is not None else iobrx.get_threads()
    print(f"iobrx {iobrx.__version__} | cibersort n_threads={n_threads}\n")

    imvigor = load("imvigor210_eset", args.testdata)
    stad_raw = load("eset_stad", args.testdata)
    anno = load("anno_grch38", args.testdata)

    t0 = perf_counter()
    stad_symbol = iobrx.anno_eset(stad_raw.copy(), anno.copy(), symbol="symbol",
                                  probe="id", method="mean")
    print(f"[data] derived symbol matrix {stad_symbol.shape} in "
          f"{perf_counter() - t0:.2f}s\n")

    from importlib.resources import files

    qs_data = pd.read_pickle(str(files("iobrpy.resources").joinpath("quantiseq_data.pkl")))
    epic_tref = pd.read_pickle(
        str(files("iobrpy.resources").joinpath("epic_TRef_BRef.pkl")))["TRef"]

    times: dict[str, float] = {}
    outs: dict[str, object] = {}

    def timed(name, fn):
        t = perf_counter()
        outs[name] = fn()
        times[name] = perf_counter() - t

    for m in ("pca", "zscore", "ssgsea", "integration"):
        timed(f"sig_{m}", lambda m=m: iobrx.calculate_sig_score(
            imvigor.copy(), "signature_collection", m, 3, True, n_threads))
    timed("anno_eset", lambda: iobrx.anno_eset(
        stad_raw.copy(), anno.copy(), symbol="symbol", probe="id", method="mean"))
    timed("cibersort", lambda: iobrx.cibersort(
        stad_symbol.copy(), perm=100, QN=True, n_threads=n_threads))
    timed("epic", lambda: iobrx.epic(stad_symbol.copy(), reference=epic_tref))
    timed("quantiseq", lambda: iobrx.quantiseq(
        stad_symbol.copy(), data=qs_data, arrays=False, tumor=False, mRNAscale=True,
        method="lsei", rmgenes="default"))
    timed("mcpcounter", lambda: iobrx.mcpcounter(
        stad_symbol.copy(), features_type="HUGO_symbols"))
    timed("estimate", lambda: iobrx.estimate_score(
        stad_symbol.copy(), platform="affy"))
    timed("count2tpm", lambda: iobrx.count2tpm(
        stad_raw.copy(), anno_grch38=anno.copy(), anno_gc_vm32=None,
        idType="Ensembl", org="hsa", source="local", check_data=True,
        remove_version=True))

    stages = list(times)
    print(f"\n{'stage':<18s} {'time':>9s}")
    print("-" * 30)
    for s in stages:
        print(f"{s:<18s} {times[s]:8.3f}s")
    print("-" * 30)
    print(f"{'TOTAL':<18s} {sum(times.values()):8.3f}s")

    if not args.verify_parity:
        return

    # ---------------- original-on-the-fly parity ----------------
    print("\n--verify-parity: running ORIGINAL iobrpy side by side ...")
    from iobrpy.workflow import cibersort as orig_cibersort
    from iobrpy.workflow.anno_eset import anno_eset as orig_anno
    from iobrpy.workflow.calculate_sig_score import calculate_sig_score as orig_sig
    from iobrpy.workflow.count2tpm import count2tpm as orig_count2tpm
    from iobrpy.workflow.epic import EPIC as orig_epic
    from iobrpy.workflow.estimate import estimate_score as orig_estimate
    from iobrpy.workflow.mcpcounter import MCPcounter_estimate as orig_mcp
    from iobrpy.workflow.quantiseq import deconvolute_quantiseq_default as orig_qs

    fails = []
    print(f"\n{'stage':<18s} {'max_abs_diff':>13s}  verdict")
    print("-" * 46)
    for m in ("pca", "zscore", "ssgsea", "integration"):
        slow = orig_sig(imvigor.copy(), ["signature_collection"], m, 3, True, 1)
        r = num_diff(outs[f"sig_{m}"], slow)
        ok = r["max_abs_diff"] == 0.0 and r["index_equal"] and r["columns_equal"]
        print(f"{f'sig_{m}':<18s} {r['max_abs_diff']:13.1e}  {'BIT-EXACT' if ok else 'DIFF'}")
        if not ok:
            fails.append(f"sig_{m}")
    for name, slow in [
        ("anno_eset", orig_anno(stad_raw.copy(), anno.copy(), symbol="symbol",
                                probe="id", method="mean")),
        ("quantiseq", orig_qs(mix=stad_symbol.copy(), data=qs_data, arrays=False,
                              tumor=False, mRNAscale=True, method="lsei",
                              rmgenes="default")),
        ("mcpcounter", orig_mcp(expression=stad_symbol.copy(),
                                features_type="HUGO_symbols")),
        ("estimate", orig_estimate(input_df=stad_symbol.copy(), platform="affy")),
        ("count2tpm", orig_count2tpm(stad_raw.copy(), anno.copy(), None, "Ensembl",
                                     "hsa", "local", None, "id", "symbol",
                                     "eff_length", True, True)),
    ]:
        r = num_diff(outs[name], slow)
        ok = r["max_abs_diff"] == 0.0 and r["index_equal"] and r["columns_equal"]
        print(f"{name:<18s} {r['max_abs_diff']:13.1e}  {'BIT-EXACT' if ok else 'DIFF'}")
        if not ok:
            fails.append(name)
    slow_epic = orig_epic(bulk=stad_symbol.copy(), reference=epic_tref)
    r = None
    ok = True
    for key in ("mRNAProportions", "cellFractions", "fit_gof"):
        rk = num_diff(outs["epic"][key], slow_epic[key])
        ok &= rk["max_abs_diff"] == 0.0 and rk["index_equal"] and rk["columns_equal"]
    print(f"{'epic':<18s} {'0.0e+00' if ok else 'DIFF':>13s}  {'BIT-EXACT' if ok else 'DIFF'}")
    if not ok:
        fails.append("epic")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mix.csv")
        stad_symbol.to_csv(path)
        slow_cib = orig_cibersort.cibersort(path, perm=100, QN=True)
    nonp = [c for c in slow_cib.columns if c != "P-value"]
    r = num_diff(outs["cibersort"][nonp], slow_cib[nonp])
    ok = r["max_abs_diff"] == 0.0 and r["index_equal"] and r["columns_equal"]
    print(f"{'cibersort*':<18s} {r['max_abs_diff']:13.1e}  {'BIT-EXACT' if ok else 'DIFF'}")
    if not ok:
        fails.append("cibersort")
    print("\n* excluding the P-value column: the ORIGINAL draws OS entropy for "
          "its permutations,\n  so its p-values are not reproducible run-to-run "
          "by design (formula identical).")

    if fails:
        raise SystemExit(f"PARITY FAILED for: {', '.join(fails)}")
    print("\nAll parity gates passed (bit-exact).")


if __name__ == "__main__":
    main()
