"""ips_fast: accelerated drop-in for the ``iobrpy.workflow.IPS`` CLI stage
(``python -m iobrpy.main IPS --input ... --output ...``).

Same behaviour as upstream (IPS.py, IOBRpy 0.2.0): read a genes (HGNC
symbol index) x samples expression matrix (``.tsv``/``.txt`` -> tab, any
other extension -> comma), apply the ``max > 100 -> log2(x+1)`` heuristic,
intersect the 162-row ``IPS_genes.txt`` resource (160 unique symbols, 26
NAME groups) with the index, then per sample

    z = (expr.reindex(IPS_genes.GENE) - expr.mean()) / expr.std(ddof=1)
    wg = groupby('NAME', sort=False).mean(z) * mean(WEIGHT)
    MHC = nanmean(wg[0:10]); CP = nanmean(wg[10:20])
    EC  = nanmean(wg[20:24]); SC = nanmean(wg[24:26])
    AZ  = MHC + CP + EC + SC;  IPS = int(round(AZ * 10/3, 0))

and write ``ID, MHC_IPS, EC_IPS, SC_IPS, CP_IPS, AZ_IPS, IPS_IPS`` (note
the upstream EC/SC-before-CP column reorder; float columns round(6),
``IPS_IPS`` int) with the separator chosen by the output extension.

Bit-exactness strategy (frozen official gate: stad symbol-TPM 54,658x10
from count2tpm -> 10x6 CSV, sha256 ``793b4718...`` byte-identical to the
recorded iobrpy 0.2.0 CLI baseline):

  1. EVERY numeric statement is executed verbatim — the pandas
     ``reindex`` / ``Series.mean`` / ``Series.std(ddof=1)`` /
     ``groupby.agg`` chain (pandas' group means use Kahan-compensated
     accumulation whose exact summation order naive numpy reductions do
     NOT reproduce), ``np.nanmean`` position slices and Python's
     banker's-rounding ``int(round(az * 10.0 / 3.0, 0))`` included.
     Vectorizing the per-sample loop was measured pointless: the whole
     loop costs ~1.9 ms/sample (~19 ms on the 10-sample gate), while the
     original's wall clock is >95% the ~1.42 s ``iobrpy.main`` full-import
     cold-start floor (module_spec §2.6).
  2. The acceleration is the import floor: pandas/numpy and the
     ``IPS_genes.txt`` resource are imported/loaded lazily (resource frame
     memoized per process), and the ``iobrpy.main`` chain, argparse, tqdm
     bar, dead-code dicts (``groups`` / ``w_means``, computed but never
     used upstream) and the decorative banner are dropped from the API
     path (banner/info lines reproduce under ``verbose=True``).

BUG-COMPATIBILITY (upstream defects deliberately preserved, module_spec
§8-2):
  * the MHC/CP/EC/SC blocks are POSITIONAL slices of the group-order-
    dependent ``wg`` array — missing groups silently misalign the slices;
  * a wholly missing EC or SC block makes ``np.nanmean`` of an empty
    slice NaN -> ``int(round(NaN...))`` raises
    ``ValueError: cannot convert float NaN to integer`` (the upstream
    crash on partial panels, e.g. imvigor210);
  * missing single genes (group still present) only print a warning
    upstream and are skipped by the group mean (skipna) — preserved;
  * z-scores use the WHOLE input matrix's per-sample mean/std, so the
    result depends on the full gene set/scale of the input (upstream
    semantics, not a bug, but contract-relevant).
"""
from __future__ import annotations

import os
from iobrx._resources import resource_path as _resource_path

__all__ = ["ips"]

_IPS_GENES = None


def _ips_genes():
    """Memoized IPS_genes.txt resource frame (immutable packaged file)."""
    global _IPS_GENES
    if _IPS_GENES is None:
        import pandas as pd
        from importlib.resources import files

        resource_path = _resource_path("IPS_genes.txt")
        _IPS_GENES = pd.read_csv(str(resource_path), sep="\t")
    return _IPS_GENES


def _print_iobrpy_banner():
    """Upstream banner (verbose console protocol only)."""
    print("   ")
    try:
        from iobrpy.utils.print_colorful_message import print_colorful_message
    except Exception:  # pragma: no cover
        def print_colorful_message(text, color=None, **kw):
            print(text)
    print_colorful_message("#########################################################", "blue")
    print_colorful_message(" IOBRpy: Immuno-Oncology Biological Research using Python ", "cyan")
    print_colorful_message(" If you encounter any issues, please report them at ", "cyan")
    print_colorful_message(" https://github.com/IOBR/IOBRpy/issues ", "cyan")
    print_colorful_message("#########################################################", "blue")
    print(" Author: Haonan Huang, Dongqiang Zeng")
    print(" Email: interlaken@smu.edu.cn ")
    print_colorful_message("#########################################################", "blue")
    print("   ")


def _run_original_cli(input_path: str, output_path: str):
    """engine='original': the UNTOUCHED upstream ``IPS.main()`` via an
    argv bridge (upstream exposes the stage only as CLI main())."""
    import importlib
    import sys

    mod = importlib.import_module("iobrpy.workflow.IPS")
    saved = sys.argv
    try:
        sys.argv = ["IPS", "--input", str(input_path), "--output", str(output_path)]
        mod.main()
    finally:
        sys.argv = saved


def ips(eset, output_file=None, verbose: bool = False, engine: str = "fast"):
    """Immunophenoscore (Charoentong 2017 four-block design), accelerated.

    Parameters
    ----------
    eset : str, path-like or pandas.DataFrame
        Expression matrix, genes (index, HGNC symbols) x samples; or a path
        to it (``.tsv``/``.txt`` -> tab, anything else comma — the ORIGINAL
        extension rule). The ``max > 100 -> log2(x+1)`` heuristic applies as
        upstream.
    output_file : str, optional
        Output CSV/TSV path (``.tsv``/``.txt`` -> tab, else comma, as
        upstream). When None, nothing is written.
    verbose : bool, default False
        Reproduce the ORIGINAL's missing-gene warning, ``Results saved to:``
        line and IOBRpy banner (the tqdm bar is dropped, per iobrx
        convention).
    engine : {'fast', 'original'}, default 'fast'
        'original' runs the untouched upstream ``IPS.main()`` through an
        argv bridge (file input AND output_file required; returns None).

    Returns
    -------
    pandas.DataFrame or None
        ``ID, MHC_IPS, EC_IPS, SC_IPS, CP_IPS, AZ_IPS, IPS_IPS`` — the same
        frame the ORIGINAL writes (the ORIGINAL itself returns None;
        engine='original' also returns None).
    """
    if engine == "original":
        if not isinstance(eset, (str, os.PathLike)) or output_file is None:
            raise ValueError(
                "engine='original' (backend='python') needs a file path input "
                "and an output_file — it runs the untouched upstream CLI main()"
            )
        _run_original_cli(str(eset), str(output_file))
        return None

    import numpy as np
    import pandas as pd

    # ---- read input (ORIGINAL extension-driven separator rule) ----
    if isinstance(eset, (str, os.PathLike)):
        input_path = str(eset)
        _, in_ext = os.path.splitext(input_path)
        in_ext = in_ext.lower()
        if in_ext == ".csv":
            sep_in = ","
        elif in_ext in [".tsv", ".txt"]:
            sep_in = "\t"
        else:
            sep_in = ","
        gene_expression = pd.read_csv(input_path, sep=sep_in, index_col=0)
    else:
        gene_expression = eset

    # ---- log2 heuristic (ORIGINAL statement) ----
    if gene_expression.values.max() > 100:
        gene_expression = np.log2(gene_expression + 1)

    IPS_genes = _ips_genes()

    # ---- missing-gene warning (ORIGINAL text; upstream prints always) ----
    missing_genes = IPS_genes[~IPS_genes["GENE"].isin(gene_expression.index)]
    if not missing_genes.empty:
        if verbose:
            print("\nWarning: The following genes are missing or mismatched in the expression matrix:")
            print(missing_genes[["GENE", "NAME"]].to_string(index=False))

    IPS_genes = IPS_genes[IPS_genes["GENE"].isin(gene_expression.index)].reset_index(drop=True)

    # (upstream computes `sig_names`, `groups` and `w_means` here; they are
    #  dead code — computed and never used — so the fast path drops them.)

    # ---- per-sample IPS (EVERY statement verbatim; see module docstring) ----
    results = []
    gene_col = IPS_genes["GENE"]
    for sample in gene_expression.columns:
        expr = gene_expression[sample]
        z = (expr.reindex(gene_col) - expr.mean()) / expr.std(ddof=1)

        df_z = pd.DataFrame({
            "GENE": IPS_genes["GENE"],
            "NAME": IPS_genes["NAME"],
            "z": z.values,
            "WEIGHT": IPS_genes["WEIGHT"],
        })

        agg = df_z.groupby("NAME", sort=False).agg({"z": "mean", "WEIGHT": "mean"}).reset_index()
        mig = agg["z"].values
        weights = agg["WEIGHT"].values
        wg = mig * weights

        # MHC, CP, EC, SC, AZ — POSITIONAL slices (bug-compatible, docstring)
        mhc = np.nanmean(wg[0:10])
        cp = np.nanmean(wg[10:20])
        ec = np.nanmean(wg[20:24])
        sc = np.nanmean(wg[24:26])
        az = mhc + cp + ec + sc

        ips = int(round(az * 10.0 / 3.0, 0))

        results.append({
            "Sample": sample,
            "MHC": mhc,
            "CP": cp,
            "EC": ec,
            "SC": sc,
            "AZ": az,
            "IPS": ips,
        })

    # ---- result frame (ORIGINAL column reorder / rounding statements) ----
    result_df = pd.DataFrame(results)
    result_df = result_df[["Sample", "MHC", "EC", "SC", "CP", "AZ", "IPS"]]
    result_df = result_df.rename(columns={"Sample": "ID"})
    new_cols = ["ID"] + [f"{c}_IPS" for c in result_df.columns.tolist()[1:]]
    result_df.columns = new_cols
    for col in ["MHC_IPS", "CP_IPS", "EC_IPS", "SC_IPS", "AZ_IPS"]:
        result_df[col] = result_df[col].round(6)
    result_df["IPS_IPS"] = result_df["IPS_IPS"].astype(int)

    if output_file is not None:
        output_path = str(output_file)
        # Determine separator based on output extension (ORIGINAL rule)
        _, ext = os.path.splitext(output_path)
        ext = ext.lower()
        if ext == ".csv":
            sep = ","
        elif ext in [".tsv", ".txt"]:
            sep = "\t"
        else:
            # Default to comma for unknown extensions
            sep = ","
        result_df.to_csv(output_path, sep=sep, index=False)
        if verbose:
            print(f"\nResults saved to: {output_path} ")
            _print_iobrpy_banner()

    return result_df
