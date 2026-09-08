"""mouse2human_fast: accelerated drop-in for the
``iobrpy.workflow.mouse2human_eset`` CLI stage (``python -m iobrpy.main
mouse2human_eset -i ... -o ... [--is_matrix] [--column_of_symbol COL]``).

Same behaviour as upstream (mouse2human_eset.py, IOBRpy 0.2.0): convert a
mouse-symbol expression set to human symbols with the packaged
``mus_human.pkl`` mapping (20,342 rows; ``gene_symbol_mus`` ->
``gene_symbol_human``, many-to-many) by delegating to the
``anno_eset(method='mean')`` core, then write with the upstream MANUAL
row writer (blank top-left header cell, NaN -> empty string, ``str()``
full-precision values, ``.gz`` auto-compression, extension-inferred
separator).

Two modes, as upstream:
  * matrix mode (``is_matrix=True``): row index = mouse symbols;
  * table mode: the symbol column named by ``column_of_symbol`` is
    de-duplicated first (row score = mean of numeric columns, sort
    DESCENDING quicksort, keep first per symbol — the ORIGINAL
    ``_remove_duplicate_genes`` statements verbatim).

Mapping-core semantics (VERBATIM ``anno_eset(method='mean')``, module_spec
§4.4): annotation rows are selected in mus_human.pkl order (NOT input
order) and unmapped mouse symbols are SILENTLY DROPPED; one mouse -> many
human symbols expands the row once per human symbol; many mouse -> one
human symbol keeps the single highest-row-mean occurrence (selection, NOT
averaging) — when any duplicate human symbol exists upstream sorts the
whole merged frame by that row-mean score DESCENDING (quicksort), so the
final row order is the score order (pkl order only survives in the
no-duplicates case); finally all-zero rows, all-NA rows and rows with NA
in the FIRST sample column are dropped.

Bit-exactness strategy (frozen official gate: 2,000 mouse symbols x 6
samples matrix-mode fixture -> human_matrix.csv, sha256 ``174aec77...``
byte-identical to the recorded iobrpy 0.2.0 CLI baseline):

  1. The mapping core REUSES ``iobrx._fast.anno_eset_fast.anno_eset`` —
     the repo's already gate-verified bit-exact vectorization of the very
     same upstream function (np.repeat probe expansion, strided-view
     row-mean scoring, nargsort-descending + factorize dedup, faithful
     pandas fallback for structural edge cases). Its unconditional stdout
     info lines are captured and re-emitted only under ``verbose=True``.
  2. File readers, separator inference, table-mode dedup, pkl loader and
     the manual output writer are copied VERBATIM — the writer's per-value
     ``str()`` of numpy scalars is the byte contract; vectorizing it
     (e.g. ``astype(str)``) would risk repr corner cases (-0.0, inf,
     subnormals) for negligible gain at real matrix sizes.
  3. The acceleration is the import floor + lazy resources: upstream's
     1.32 s median wall clock is almost entirely the ~1.4 s
     ``iobrpy.main`` full-import cold start (module_spec §6); this module
     lazily imports pandas and the 350 KB ``mus_human.pkl`` (memoized per
     process) only on call, drops the always-on tqdm save bar and routes
     the ``[iobrpy] Converted matrix saved to:`` line + IOBRpy banner
     through ``verbose=True``.
"""
from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path

__all__ = ["mouse2human"]

_MUS_HUMAN = None


# ---------------------- utilities (VERBATIM upstream) ----------------------
def _infer_sep(path: str, explicit=None) -> str:
    """Infer separator from extension unless explicitly provided.

    Rules:
    - .tsv, .tsv.gz, .tab, .tab.gz, .txt, .txt.gz  -> '\\t'
    - otherwise                                     -> ','
    """
    if explicit is not None:
        return explicit
    p = str(path).lower()
    tab_exts = (".tsv", ".tsv.gz", ".tab", ".tab.gz", ".txt", ".txt.gz")
    if p.endswith(tab_exts):
        return "\t"
    return ","


def _read_matrix(path: str, sep=None):
    """Read expression matrix with row index as the first column."""
    import pandas as pd

    sep = _infer_sep(path, sep)
    df = pd.read_csv(path, sep=sep, index_col=0, compression="infer")
    return df


def _read_table_with_symbol(path: str, sep=None):
    """Read non-matrix table (gene symbol is a normal column)."""
    import pandas as pd

    sep = _infer_sep(path, sep)
    df = pd.read_csv(path, sep=sep, compression="infer")
    return df


def _remove_duplicate_genes(df, column_of_symbol: str, method: str = "mean"):
    """VERBATIM upstream table-mode dedup: score rows by the aggregation of
    numeric columns, sort DESCENDING (quicksort), keep the first row per
    symbol, reindex by symbol, keep the remaining columns in original order.
    """
    import pandas as pd

    df = df.copy()

    if column_of_symbol not in df.columns:
        raise KeyError(
            f"Column '{column_of_symbol}' not found in the input table. "
            f"Available columns: {list(df.columns)}"
        )

    sym_col = column_of_symbol
    value_cols = [c for c in df.columns if c != sym_col]

    # Keep only numeric columns for scoring (non-numeric are ignored)
    numeric_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found for duplicate resolution.")

    # Compute score per row
    if method == "mean":
        score = df[numeric_cols].mean(axis=1, skipna=True)
    elif method == "sd":
        score = df[numeric_cols].std(axis=1, skipna=True)
    elif method == "sum":
        score = df[numeric_cols].sum(axis=1, skipna=True)
    else:
        score = df[numeric_cols].mean(axis=1, skipna=True)

    df["_score"] = score
    # Sort by score descending so the first duplicate kept is the highest score
    df.sort_values("_score", ascending=False, inplace=True)
    df = df.drop(columns=["_score"])

    # Drop duplicates keeping the first (highest score)
    df = df.drop_duplicates(subset=[sym_col], keep="first")

    # Set index to symbols and keep only expression columns
    df = df.set_index(sym_col)
    df = df[value_cols]

    return df


def _load_mus_human_df():
    """VERBATIM upstream pkl loader (DataFrame / dict-of-DataFrame tolerant),
    memoized per process (the packaged resource is immutable)."""
    global _MUS_HUMAN
    if _MUS_HUMAN is not None:
        return _MUS_HUMAN

    import pandas as pd
    from importlib.resources import files as ir_files

    resource_path = ir_files("iobrpy.resources").joinpath("mus_human.pkl")
    with resource_path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, dict):
        # choose the only DataFrame or raise if ambiguous
        dfs = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
        if len(dfs) == 1:
            df = dfs[0].copy()
        elif len(dfs) > 1:
            print("Warning: multiple DataFrames found in mus_human.pkl; using the first one.")
            df = dfs[0].copy()
        else:
            df = pd.DataFrame(obj)
    else:
        df = pd.DataFrame(obj)

    needed = {"gene_symbol_mus", "gene_symbol_human"}
    if not needed.issubset(set(df.columns)):
        raise KeyError(
            f"mus_human.pkl does not contain required columns {needed}. "
            f"Available columns: {list(df.columns)}"
        )
    _MUS_HUMAN = df
    return df


def _open_text_writer(path: Path):
    """Open a text writer; use gzip if path ends with .gz. (VERBATIM)"""
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "wt", newline="")
    return path.open("w", newline="")


def _write_with_progress(df, out_path: Path, sep: str, show_progress: bool = False):
    """VERBATIM upstream manual writer (the byte contract): blank top-left
    header cell, NaN -> empty string, ``str()`` full-precision values,
    ``.gz`` compression by extension. The tqdm bar is dropped per iobrx
    convention (``show_progress`` accepted but inert)."""
    import pandas as pd

    cols = list(map(str, df.columns.tolist()))

    with _open_text_writer(out_path) as f:
        # header: blank cell + column names
        f.write(sep.join([""] + cols) + "\n")

        for row in df.itertuples(index=True, name=None):  # (index, v1, v2, ...)
            idx = "" if row[0] is None else str(row[0])
            # Replace NaN with empty string and convert to str
            values = [("" if pd.isna(v) else v) for v in row[1:]]
            f.write(idx + sep + sep.join(map(str, values)) + "\n")


def _print_iobrpy_banner():
    """Upstream banner (verbose console protocol only)."""
    print("   ")

    def _pc(text, color):
        try:
            from iobrpy.utils.print_colorful_message import print_colorful_message

            print_colorful_message(text, color)
        except Exception:  # pragma: no cover
            print(text)

    _pc("#########################################################", "blue")
    _pc(" IOBRpy: Immuno-Oncology Biological Research using Python ", "cyan")
    _pc(" If you encounter any issues, please report them at ", "cyan")
    _pc(" https://github.com/IOBR/IOBRpy/issues ", "cyan")
    _pc("#########################################################", "blue")
    print(" Author: Haonan Huang, Dongqiang Zeng")
    print(" Email: interlaken@smu.edu.cn ")
    _pc("#########################################################", "blue")
    print("   ")


def _run_original_cli(argv_tail: list) -> None:
    """engine='original': the UNTOUCHED upstream ``mouse2human_eset.main()``
    via an argv bridge (upstream exposes the stage only as CLI main())."""
    import importlib
    import sys

    mod = importlib.import_module("iobrpy.workflow.mouse2human_eset")
    saved = sys.argv
    try:
        sys.argv = ["mouse2human_eset"] + argv_tail
        mod.main()
    finally:
        sys.argv = saved


# ----------------------------- public entry -------------------------------
def mouse2human(input, output=None, is_matrix: bool = False,
                column_of_symbol=None, sep=None, out_sep=None,
                verbose: bool = False, engine: str = "fast"):
    """Convert mouse gene symbols to human gene symbols, accelerated.

    Parameters
    ----------
    input : str, path-like or pandas.DataFrame
        Matrix mode: genes (mouse symbols, index) x samples. Table mode: a
        table containing the symbol column named by ``column_of_symbol``.
        Paths are read with the ORIGINAL extension-driven separator rule
        (``.tsv/.tab/.txt(+.gz)`` -> tab, else comma; ``sep`` overrides).
    output : str, path-like or None
        Output path (``.gz`` auto-compresses; separator by extension or
        ``out_sep``). When None, nothing is written.
    is_matrix : bool, default False
        Treat ``input`` as a symbol-indexed matrix (upstream
        ``--is_matrix``).
    column_of_symbol : str, optional
        Symbol column for table mode; required when ``is_matrix=False``
        (ValueError with the ORIGINAL message otherwise).
    sep, out_sep : str, optional
        Input/output separator overrides (upstream ``--sep``/``--out_sep``).
    verbose : bool, default False
        Reproduce the ORIGINAL's info lines (input shape / mode, mapping
        shape, ``anno_eset`` match-rate statistics, ``[iobrpy] Converted
        matrix saved to:`` and the IOBRpy banner). The always-on upstream
        tqdm save bar is dropped per iobrx convention.
    engine : {'fast', 'original'}, default 'fast'
        'original' runs the untouched upstream CLI ``main()`` through an
        argv bridge (file ``input`` AND ``output`` required; returns None).

    Returns
    -------
    pandas.DataFrame or None
        Human-symbol-indexed matrix (rows in mus_human.pkl order) — the
        same frame the manual writer serializes. engine='original' returns
        None.
    """
    if engine == "original":
        if not isinstance(input, (str, os.PathLike)) or output is None:
            raise ValueError(
                "engine='original' (backend='python') needs file path input and "
                "output — it runs the untouched upstream CLI main()"
            )
        argv = ["-i", str(input), "-o", str(output)]
        if is_matrix:
            argv.append("--is_matrix")
        if column_of_symbol is not None:
            argv += ["--column_of_symbol", str(column_of_symbol)]
        if verbose:
            argv.append("--verbose")
        if sep is not None:
            argv += ["--sep", sep]
        if out_sep is not None:
            argv += ["--out_sep", out_sep]
        _run_original_cli(argv)
        return None

    # ---- read input (ORIGINAL readers / separator inference) ----
    if isinstance(input, (str, os.PathLike)):
        in_path = str(input)
        if is_matrix:
            df = _read_matrix(in_path, sep=sep)
        else:
            df = _read_table_with_symbol(in_path, sep=sep)
    else:
        df = input

    if verbose:
        print(f"Loaded input shape: {df.shape}")
        print(f"is_matrix={is_matrix}, column_of_symbol={column_of_symbol}")

    # ---- table mode: ORIGINAL dedup first ----
    if not is_matrix:
        if not column_of_symbol:
            raise ValueError("When is_matrix=False, --column_of_symbol must be specified.")
        df = _remove_duplicate_genes(df, column_of_symbol=column_of_symbol, method="mean")

    # ---- mapping (mus_human.pkl, lazily memoized) ----
    probe_data = _load_mus_human_df()
    if verbose:
        print("Loaded mus_human mapping:", probe_data.shape)

    # ---- anno_eset(method='mean') core via the repo's gate-verified
    #      bit-exact vectorization; its info prints are verbose-gated ----
    from iobrx._fast.anno_eset_fast import anno_eset as _anno_eset

    if verbose:
        result = _anno_eset(
            eset_df=df, annotation=probe_data,
            symbol="gene_symbol_human", probe="gene_symbol_mus", method="mean",
        )
    else:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            result = _anno_eset(
                eset_df=df, annotation=probe_data,
                symbol="gene_symbol_human", probe="gene_symbol_mus", method="mean",
            )

    # ---- ORIGINAL manual writer (the byte contract) ----
    if output is not None:
        out_sep_eff = _infer_sep(str(output), out_sep)
        out_path = Path(str(output))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_with_progress(result, out_path, sep=out_sep_eff)
        if verbose:
            print(f"[iobrpy] Converted matrix saved to: {out_path.resolve()}")
            _print_iobrpy_banner()

    return result
