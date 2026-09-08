"""log2_eset_fast: accelerated drop-in for the ``iobrpy.workflow.log2_eset``
CLI stage (``python -m iobrpy.main log2_eset -i ... -o ...``).

Same behaviour as upstream (log2_eset.py, IOBRpy 0.2.0): three-level input
delimiter detection (``csv.Sniffer`` over the first 64 KiB -> character
frequency heuristic over the first 10 non-empty lines -> extension hint ->
tab), ``pd.read_csv(sep, index_col=0)``, all-column ``pd.to_numeric(errors=
'coerce')``, the ``min < -1`` hard error and the ``-1 <= min < 0`` warning,
``np.log2(df + 1.0)`` and ``to_csv`` with the output-extension separator
rule (``.csv`` -> comma, ``.tsv`` -> tab, otherwise mirror the input
delimiter); ``.gz`` handled transparently by pandas/gzip as upstream.

Bit-exactness strategy (frozen official gate: eset_stad.csv 60,483x10
official testdata -> 60,483x10 CSV, sha256 ``edc2664a...`` byte-identical
to the recorded iobrpy 0.2.0 CLI baseline):

  1. The delimiter detectors, transform and writers are copied VERBATIM —
     ``np.log2`` and pandas' ``to_csv`` shortest-repr float serialization
     are already vectorized upstream; there is no arithmetic left to
     accelerate without changing bytes.
  2. The only optimization is a provable no-op guard: when every column is
     already an integer/unsigned/float dtype, ``pd.to_numeric(errors=
     'coerce')`` is the identity on the values (pandas returns numeric
     dtypes unchanged), so the per-column ``df.apply`` is skipped. Object /
     string / bool / categorical columns always take the ORIGINAL apply.
  3. The acceleration proper is the import floor: the ORIGINAL wall clock
     is ~1.42 s ``iobrpy.main`` full-import cold start + ~0.3 s of work
     (module_spec §4.10); this module lazily imports csv/gzip/numpy/pandas
     only on call and never touches ``iobrpy.main``.

Contract note (repo convention, cf. ``iobrx.nmf``): upstream failure paths
call ``sys.exit(1)`` after printing a ``❌`` line to stderr; the API raises
``ValueError`` carrying the ORIGINAL message text instead (a SystemExit
from a library call would kill interactive sessions). engine='original'
keeps the untouched upstream behaviour, SystemExit included.
"""
from __future__ import annotations

import os

__all__ = ["log2_eset"]

CANDIDATE_SEPS = [",", "\t", ";", "|"]


# --------------------------------------------------------------------------
# delimiter handling — VERBATIM upstream copies (pure stdlib, no iobrpy deps)
# --------------------------------------------------------------------------
def _open_text(path: str):
    """Open plain or gzipped text file in UTF-8 with replacement for bad bytes."""
    import gzip

    if path.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace", newline="")
    return open(path, mode="r", encoding="utf-8", errors="replace", newline="")


def _ext_lower(path: str) -> str:
    """Return lowercase extension without .gz suffix, e.g., .csv for *.csv.gz."""
    p = path.lower()
    if p.endswith(".gz"):
        p = p[:-3]
    _, ext = os.path.splitext(p)
    return ext  # includes leading dot, e.g., ".csv", ".tsv", ".txt", or ""


def _detect_input_sep(path: str) -> str:
    """Detect delimiter using csv.Sniffer first; fall back to simple frequency
    heuristic over the first N non-empty lines. (VERBATIM upstream, except
    that the open-failure ``sys.exit(1)`` becomes the raised OSError.)"""
    import csv
    import sys

    sample = ""
    try:
        with _open_text(path) as f:
            # Read up to ~64 KiB for sniffing
            sample = f.read(65536)
    except Exception as e:
        raise ValueError(
            f"❌ Failed to open '{path}' for delimiter detection: {e}"
        ) from e

    # Try csv.Sniffer
    if sample:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_SEPS))
            if getattr(dialect, "delimiter", None) in CANDIDATE_SEPS:
                return dialect.delimiter
        except Exception:
            pass  # fall through to heuristic

    # Heuristic: count occurrences of candidates across first ~10 non-empty lines
    counts = {sep: 0 for sep in CANDIDATE_SEPS}
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    for ln in lines[:10]:
        for sep in CANDIDATE_SEPS:
            counts[sep] += ln.count(sep)

    # Choose the sep with the highest count; break ties by a preference order
    best_sep = max(CANDIDATE_SEPS, key=lambda s: (counts[s], -CANDIDATE_SEPS.index(s)))
    if counts[best_sep] > 0:
        return best_sep

    # As a last resort, use extension hint
    ext = _ext_lower(path)
    if ext == ".csv":
        return ","
    if ext == ".tsv":
        return "\t"

    # Default to tab (TSV) if totally ambiguous
    return "\t"


def _choose_output_sep(out_path: str, in_sep: str) -> str:
    """Decide output delimiter (VERBATIM upstream):
      - *.csv or *.csv.gz -> ','
      - *.tsv or *.tsv.gz -> '\\t'
      - otherwise -> mirror input delimiter
    """
    ext = _ext_lower(out_path)
    if ext == ".csv":
        return ","
    if ext == ".tsv":
        return "\t"
    # Mirror input if extension is ambiguous (e.g., .txt, no extension, etc.)
    return in_sep or "\t"


def _run_original_cli(input_path: str, output_path: str):
    """engine='original': the UNTOUCHED upstream ``log2_eset.main()`` via an
    argv bridge (upstream exposes the stage only as CLI main(); its
    ``sys.exit(1)`` failure paths propagate as SystemExit)."""
    import importlib
    import sys

    mod = importlib.import_module("iobrpy.workflow.log2_eset")
    saved = sys.argv
    try:
        sys.argv = ["log2_eset", "-i", str(input_path), "-o", str(output_path)]
        mod.main()
    finally:
        sys.argv = saved


def log2_eset(input, output, verbose: bool = False, engine: str = "fast"):
    """Apply log2(x+1) to a genes x samples expression matrix, accelerated.

    Parameters
    ----------
    input : str or path-like
        Input matrix (csv/tsv/txt, ``.gz`` supported); first column = gene
        IDs (index). The delimiter is detected with the ORIGINAL three-level
        sniffing chain.
    output : str or path-like
        Output path; ``.csv(.gz)`` -> comma, ``.tsv(.gz)`` -> tab, otherwise
        the input delimiter is mirrored (ORIGINAL rule). ``.gz`` output is
        compressed by pandas as upstream.
    verbose : bool, default False
        Reproduce the ORIGINAL's stderr diagnostics (detected/output
        delimiter, coercion and negative-value warnings, ✅ Done line).
    engine : {'fast', 'original'}, default 'fast'
        'original' runs the untouched upstream CLI ``main()`` through an
        argv bridge (SystemExit failure semantics included; returns None).

    Returns
    -------
    pandas.DataFrame or None
        The log2(x+1) matrix as written (index = gene IDs, index name
        preserved). engine='original' returns None.

    Raises
    ------
    ValueError
        On the ORIGINAL's ``sys.exit(1)`` paths (unreadable/empty input,
        values < -1, transform or write failure), carrying the ORIGINAL
        message text (repo convention; see module docstring).
    """
    in_path = str(input)
    out_path = str(output)

    if engine == "original":
        _run_original_cli(in_path, out_path)
        return None

    import sys

    import numpy as np
    import pandas as pd

    # Auto-detect input delimiter
    in_sep = _detect_input_sep(in_path)
    sep_name = {"\t": "TAB", ",": "COMMA", ";": "SEMICOLON", "|": "PIPE"}.get(in_sep, repr(in_sep))
    if verbose:
        print(f"ℹ️ Detected input delimiter: {sep_name}", file=sys.stderr)

    # Read matrix with detected delimiter; keep first column as gene index
    try:
        df = pd.read_csv(in_path, sep=in_sep, index_col=0)
    except Exception as e:
        raise ValueError(
            f"❌ Failed to read input file '{in_path}' with sep={repr(in_sep)}: {e}"
        ) from e

    if df.empty:
        raise ValueError(
            "❌ Input matrix is empty after reading. Check delimiter and index column."
        )

    # Convert all columns to numeric (coerce errors to NaN) while preserving
    # shape. PROVABLE NO-OP GUARD: pd.to_numeric returns int/uint/float
    # dtypes unchanged (identity on the values), so the per-column apply is
    # skipped only when every column already has such a dtype; anything else
    # (object/string/bool/categorical) takes the ORIGINAL apply.
    if not all(df[c].dtype.kind in "fiu" for c in df.columns):
        df_before_na = df.isna().sum().sum()
        df = df.apply(pd.to_numeric, errors="coerce")
        coerced_cells = int(df.isna().sum().sum() - df_before_na)
        if coerced_cells > 0 and verbose:
            print(
                f"⚠️ Detected non-numeric entries; coerced to NaN in {coerced_cells} cells.",
                file=sys.stderr,
            )

    # Domain check for log2(x+1)  (ORIGINAL semantics: < -1 is a hard error)
    min_val = df.min().min(skipna=True)
    if pd.notna(min_val) and min_val < -1:
        raise ValueError(
            f"❌ Found values < -1 (min={min_val}). log2(x+1) is undefined for x < -1."
        )
    if pd.notna(min_val) and min_val < 0 and verbose:
        print(
            f"⚠️ Found negative values (min={min_val}). Proceeding with log2(x+1); "
            f"verify this is expected.",
            file=sys.stderr,
        )

    # Transform
    try:
        out_df = np.log2(df + 1.0)
    except Exception as e:
        raise ValueError(f"❌ Failed during log2(x+1) transformation: {e}") from e

    # Choose output delimiter
    out_sep = _choose_output_sep(out_path, in_sep)
    out_sep_name = {"\t": "TAB", ",": "COMMA", ";": "SEMICOLON", "|": "PIPE"}.get(out_sep, repr(out_sep))
    if verbose:
        print(f"ℹ️ Output delimiter: {out_sep_name}", file=sys.stderr)

    # Write output
    try:
        out_df.to_csv(out_path, sep=out_sep, index=True)
    except Exception as e:
        raise ValueError(f"❌ Failed to write output file '{out_path}': {e}") from e

    if verbose:
        print(f"✅ Done. Wrote log2(x+1) matrix to: {out_path}", file=sys.stderr)

    return out_df
