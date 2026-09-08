"""prepare_salmon_fast: accelerated drop-in for
``iobrpy.workflow.prepare_salmon.prepare_salmon_tpm`` (CLI
``python -m iobrpy.main prepare_salmon -i ... -o ... [--return_feature
{ENST,ENSG,symbol}] [--remove_version]``).

Same behaviour as upstream (prepare_salmon.py, IOBRpy 0.2.0): read the
merge_salmon TPM matrix (tab-separated, ``.gz`` transparent via
``compression='infer'``), require a ``Name`` column of GENCODE-style
8-field pipe IDs (``ENST.ver|ENSG.ver|OTTHUMG.ver|OTTHUMT.ver|tx_name|
gene_symbol|length|biotype``), parse the annotation IN PLACE from those
pipe fields (upstream reads no external annotation file at all), replace
``Name`` with the selected feature, optionally strip ``.version``
suffixes, then ``remove_duplicate_genes``: ``groupby('Name')`` (sorted ->
alphabetical row order), per-numeric-column group MEAN, non-numeric
columns take the group's first value via a left merge; finally
``df.to_csv(output_matrix, index=False)`` — always comma-separated,
regardless of the output extension (upstream quirk, preserved).

Bit-exactness strategy (frozen official gate: fixture8_salmon_tpm.tsv.gz
60,000 transcripts x 8 samples -> 30,000 x 9 CSV, sha256 ``d5546a04...``
byte-identical to the recorded iobrpy 0.2.0 CLI baseline):

  1. EVERY statement is executed verbatim — the pipe-split annotation
     parse (including the ``Length mismatch`` ValueError when a Name has
     < 8 fields), ``str.split('.').str[0]``, the per-column
     ``group[col].mean()`` loop (pandas groupby means use Kahan-
     compensated accumulation; folding the columns into one multi-column
     call was rejected — ~5% of net runtime at gate scale, not worth any
     summation-order risk) and the ``to_csv`` writer.
  2. The acceleration is the import floor: upstream's 1.742 s median wall
     clock is ~1.5 s ``iobrpy.main`` full-import cold start (module_spec
     §3.10); this module lazily imports pandas only on call, drops the
     tqdm bars and routes the ``>>>`` progress lines / green ``Done!`` /
     IOBRpy banner through ``verbose=True``.

BUG-COMPATIBILITY (upstream defects deliberately preserved, module_spec
§5.3-3/4):
  * the WHOLE flow sits in a try/except: any error only prints a red
    ``Error occurred: ...`` line, does NOT raise and does NOT write the
    output file (upstream returns rc=0 in the CLI);
  * the output is ALWAYS a comma-separated CSV even for ``.tsv`` output
    paths (``to_csv`` without ``sep``);
  * ``return_feature`` is matched case-insensitively inside the function
    (``.lower()``) although the CLI's argparse choices are
    case-sensitive — an arbitrary-cased 'symbol'/'SYMBOL' works here as
    in the function-level API.
"""
from __future__ import annotations

__all__ = ["prepare_salmon", "remove_duplicate_genes"]


def _print_colorful(text: str, color: str) -> None:
    """Upstream colored print with the repo's graceful fallback."""
    try:
        from iobrpy.utils.print_colorful_message import print_colorful_message

        print_colorful_message(text, color)
    except Exception:  # pragma: no cover
        print(text)


def _print_iobrpy_banner():
    """Upstream banner (verbose console protocol only)."""
    print("   ")
    _print_colorful("#########################################################", "blue")
    _print_colorful(" IOBRpy: Immuno-Oncology Biological Research using Python ", "cyan")
    _print_colorful(" If you encounter any issues, please report them at ", "cyan")
    _print_colorful(" https://github.com/IOBR/IOBRpy/issues ", "cyan")
    _print_colorful("#########################################################", "blue")
    print(" Author: Haonan Huang, Dongqiang Zeng")
    print(" Email: interlaken@smu.edu.cn ")
    _print_colorful("#########################################################", "blue")
    print("   ")


def remove_duplicate_genes(eset, column_of_symbol: str = "Name"):
    """VERBATIM upstream ``remove_duplicate_genes`` (tqdm bar dropped).

    Remove duplicate genes by averaging numeric expression values and
    taking the first non-numeric annotation for duplicates.
    """
    import pandas as pd

    # group once
    group = eset.groupby(column_of_symbol)

    # numeric columns: compute group mean per column (ORIGINAL per-column
    # loop; group[col].mean() returns a Series indexed by group keys)
    numeric_cols = eset.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        numeric_dict = {}
        for col in numeric_cols:
            numeric_dict[col] = group[col].mean()
        numeric_means = pd.DataFrame(numeric_dict)
        numeric_means.index.name = column_of_symbol
        numeric_means = numeric_means.reset_index()
    else:
        # no numeric columns: create a base frame with group keys so merges later work
        numeric_means = pd.DataFrame({column_of_symbol: list(group.groups.keys())})

    # non-numeric columns: merge first non-null value per group
    non_numeric = [c for c in eset.select_dtypes(exclude=["number"]).columns if c != column_of_symbol]
    if non_numeric:
        for col in non_numeric:
            first_vals = group[col].first().reset_index()
            numeric_means = numeric_means.merge(first_vals, on=column_of_symbol, how="left")

    return numeric_means


def prepare_salmon(input, output, return_feature: str = "symbol",
                   remove_version: bool = False, verbose: bool = False,
                   engine: str = "fast"):
    """Process a Salmon quantification matrix into a deduplicated TPM matrix.

    Parameters
    ----------
    input : str or path-like
        merge_salmon TPM matrix (TSV or TSV.GZ) with the GENCODE 8-field
        pipe-ID ``Name`` column.
    output : str or path-like
        Output path — ALWAYS written as a comma-separated CSV with
        ``index=False`` (upstream quirk, bug-compatible). NOT written when
        any step fails (the error is swallowed, see below).
    return_feature : {'symbol', 'ENST', 'ENSG'}, default 'symbol'
        Which pipe-field annotation to keep as ``Name`` (matched
        case-insensitively, as the upstream function does).
    remove_version : bool, default False
        Strip ``.N`` version suffixes from the selected IDs.
    verbose : bool, default False
        Reproduce the ORIGINAL's ``>>>`` stdout protocol, green ``Done!``
        and IOBRpy banner (tqdm bars dropped, per iobrx convention).
    engine : {'fast', 'original'}, default 'fast'
        'original' calls the untouched upstream
        ``prepare_salmon_tpm(eset_path, output_matrix, return_feature,
        remove_version)`` (returns None).

    Returns
    -------
    pandas.DataFrame or None
        The deduplicated matrix as written (``Name`` + sample columns,
        rows in symbol-alphabetical groupby order). ``None`` when the flow
        failed — BUG-COMPATIBLE with upstream, whose try/except swallows
        EVERY exception, prints a red ``Error occurred: ...`` line and
        leaves the output file unwritten (the CLI still exits rc=0).
    """
    if engine == "original":
        from iobrpy.workflow.prepare_salmon import prepare_salmon_tpm

        prepare_salmon_tpm(
            eset_path=str(input), output_matrix=str(output),
            return_feature=return_feature, remove_version=remove_version,
        )
        return None

    import pandas as pd

    eset_path = str(input)
    output_matrix = str(output)

    # BUG-COMPAT: upstream wraps the WHOLE flow in try/except and only
    # prints a red one-liner on failure (no raise, no output file).
    try:
        if verbose:
            print(f">>> Loading file: {eset_path}")
        df = pd.read_csv(eset_path, sep="\t", compression="infer")

        if "Name" not in df.columns:
            raise ValueError("'Name' column is missing in the input file. Ensure Salmon output is correct.")

        if verbose:
            print(">>> Parsing annotation from 'Name' column")
        anno = df["Name"].str.split("|", expand=True).iloc[:, :8]
        anno.columns = ["ENST", "ENSG", "OTTHUMG", "OTTHUMT", "symbol2", "symbol", "length", "biotype"]

        if verbose:
            print(">>> Selecting gene feature to return")
        rf = return_feature.lower()
        if rf == "enst":
            df["Name"] = anno["ENST"]
        elif rf == "ensg":
            df["Name"] = anno["ENSG"]
        elif rf == "symbol":
            df["Name"] = anno["symbol"]
        else:
            raise ValueError(f"Invalid return_feature: {return_feature}. Choose from ENST, ENSG, symbol.")

        if remove_version:
            df["Name"] = df["Name"].str.split(".").str[0]

        if verbose:
            print(">>> Removing duplicate genes by averaging expression values")
            print(">>> Deduplicating (progress will be shown for non-numeric columns)...")
        df = remove_duplicate_genes(df, column_of_symbol="Name")

        if verbose:
            expr_cols = df.select_dtypes(include=["number"]).columns
            print(f">>> Expression range: {df[expr_cols].min().min():.3f} to {df[expr_cols].max().max():.3f}")
            print(f">>> Saving to {output_matrix}")
        df.to_csv(output_matrix, index=False)
        if verbose:
            _print_colorful("Done!", "green")
            _print_iobrpy_banner()
        return df

    except Exception as e:
        _print_colorful(f"Error occurred: {e}", "red")
        return None
