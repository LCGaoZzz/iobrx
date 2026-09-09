"""merge_star_count_fast: accelerated drop-in for the
``iobrpy.workflow.merge_star_count`` CLI stage (``python -m iobrpy.main
merge_star_count --path ... --project ...``).

Same behaviour as upstream (merge_star_count.py, IOBRpy 0.2.0): discover
``*_ReadsPerGene.out.tab`` files in ONE directory level (``os.listdir``,
non-recursive), read column 0 (gene ID) and column 1 (unstranded count) of
every file (sample name = file name minus the suffix), concatenate the
per-sample columns on the index union and persist

    <path>/<project>.STAR.count.tsv.gz

as a gzip(level 5) TSV written with ``pandas.DataFrame.to_csv(f, sep="\t")``
(index=True, index name ``ID``) — the EXACT upstream write path, so value
tokens, the ``ID`` first-column header and the per-column dtype quirks are
byte-identical to the original decompressed output.

BUG-COMPAT (upstream defect preserved on purpose): STAR's four leading
global-stat rows (N_unmapped / N_multimapping / N_noFeature / N_ambiguous —
bare integers in the file, no text labels) are NOT skipped; they enter the
matrix as data rows. Because the stat integers differ per sample, the index
union grows by up to ``4 x n_samples`` NaN-padded junk rows: the frozen
8-sample gate reproduces the upstream (60032, 8) output — 60,000 gene rows +
32 stat rows, each stat row non-null only in its owning sample.

Bit-exactness strategy (validated on the frozen 8 x 60,004-row STAR fixture:
decompressed content token-for-token equal to the recorded iobrpy 0.2.0
baseline after column alignment, 480,256 (label, sample) cells, and equal
to a freshly regenerated original run):

  1. Reading: per-file ``pd.read_csv`` is replaced by C-level byte surgery
     (same machinery as ``merge_salmon_fast``) — one ``data.split(b"\t")``
     per file (valid because the strict per-line tab-count check gives the
     pieces a fixed stride), stride slices for columns 0/1 and one
     ``np.fromstring(sep=' ')`` for the counts. Equivalence guards (any
     trip routes the WHOLE run to the original pandas statements): CR
     bytes, ragged/blank lines, non-constant column count, >2GiB files,
     duplicate ID labels, pandas NA-value tokens or an all-numeric ID
     column (read_csv dtype inference would not be raw unique strings),
     non-integer-form or overlong count tokens, and a canonical-form round
     trip (``str(int64)`` must equal every source count token byte-for-
     byte) which pins pandas' int64 column inference AND the exact float64
     value behind the NaN-upcast ``"1234.0"`` tokens.
  2. Merging: ``pd.concat(axis=1)`` is replaced by the union-append row
     order pandas uses with ``sort=False`` (first file's labels in order,
     then each later file's new labels appended in its within-file order —
     verified against the golden output's stat-row layout), a preallocated
     NaN-filled float64 matrix, and a per-column int64 down-cast exactly
     for the columns that cover the whole union (pandas keeps those int64;
     NaN-carrying columns up-cast to float64 — verified per-column dtype
     behaviour on pandas 2.3.3).
  3. Anything outside the guarded canonical form falls back, for the WHOLE
     run, to the original pandas statements (upstream ``_read_star_file`` +
     ``pd.concat``), keeping error messages and dtype quirks bug-compatible
     (duplicate ID labels raise the same ``InvalidIndexError``, NA-token
     labels become the same NaN index entries, int64/float64 token forms
     are whatever upstream produces).

Documented deviation (per the porting contract): the ORIGINAL enumerates
with ``os.listdir`` (unordered) and collects through
``concurrent.futures.as_completed`` → BOTH the output column (sample) order
AND the union row anchor (which file's labels come first) are
nondeterministic run-to-run. Here files are processed in sorted
sample-name order: same column set, same row-label set, same value in
every (label, sample) cell, deterministic order. The console head preview
therefore also shows sorted columns.

Thread discipline: the fast parser runs sequentially — its C-level bytes
surgery and ``np.fromstring`` are GIL-bound (the same machinery measured
strictly faster sequential than at any thread count on the merge_salmon
gate). The ThreadPoolExecutor is kept only for the fallback pandas loader
(mirrors upstream, whose C tokenizer releases the GIL). Worker counts never
exceed 32; the fast path itself spawns no threads at all.
"""
from __future__ import annotations

import gzip
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Optional progress bar (same graceful degradation as upstream)
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

__all__ = ["merge_star_count", "merge_star_count_original"]

_SUFFIX = "_ReadsPerGene.out.tab"
# Count tokens longer than 18 decimal digits do not always fit int64; the
# canonical-form round trip below additionally pins values to <= 2**53
# (exact float64), so anything beyond routes to the original parser.
_MAX_INT_TOKEN = 18
_WORKERS_CAP = 32  # deployment guardrail (upstream: min(32, cpu + 4))

# pandas' default NA-value tokens: any of these in column 0 becomes a NaN
# index label in read_csv (written back as an empty field) -> the fast path
# refuses them and lets the original parser decide.
try:  # keep the guard pinned to the installed parser's own list
    from pandas._libs.parsers import STR_NA_VALUES as _STR_NA_VALUES
    _NA_TOKENS = frozenset(t.encode() for t in _STR_NA_VALUES) | {b""}
except Exception:  # pragma: no cover - documented pandas 2.x defaults
    _NA_TOKENS = frozenset(
        t.encode() for t in (
            "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN",
            "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL",
            "NaN", "None", "n/a", "nan", "null",
        )
    )

# bytes that could make pandas infer a NUMERIC dtype for the ID column
# (digits, sign, point, exponent letters) plus the '\n' joiner used for the
# whole-column check -> an all-match column takes the original pandas path
_ID_NUMERIC_LUT = np.zeros(256, dtype=bool)
_ID_NUMERIC_LUT[list(b"0123456789+-.eE\n")] = True
# bytes allowed in a canonical int-form count token (' ' is the joiner)
_INTBYTE_LUT = np.zeros(256, dtype=bool)
_INTBYTE_LUT[list(b"0123456789+- ")] = True


class _NeedsPandasPath(Exception):
    """Internal signal: route the WHOLE run through the original parser."""


def _print_iobrpy_banner():
    """Print the IOBRpy banner (same helper as merge_salmon_fast)."""
    print(" ")
    try:
        from iobrpy.utils.print_colorful_message import print_colorful_message
        print_colorful_message("#########################################################", "blue")
        print_colorful_message(" IOBRpy: Immuno-Oncology Biological Research using Python ", "cyan")
        print_colorful_message(" If you encounter any issues, please report them at ", "cyan")
        print_colorful_message(" https://github.com/IOBR/IOBRpy/issues ", "cyan")
        print_colorful_message("#########################################################", "blue")
        print(" Author: Haonan Huang, Dongqiang Zeng")
        print(" Email: interlaken@smu.edu.cn ")
        print_colorful_message("#########################################################", "blue")
    except Exception:
        # Fallback without colors
        print("#########################################################")
        print(" IOBRpy: Immuno-Oncology Biological Research using Python ")
        print(" If you encounter any issues, please report them at ")
        print(" https://github.com/IOBR/IOBRpy/issues ")
        print("#########################################################")
        print(" Author: Haonan Huang, Dongqiang Zeng")
        print(" Email: interlaken@smu.edu.cn ")
        print("#########################################################")
    print(" ")


def _list_star_files(path: str):
    """List files in ``path`` ending with the STAR suffix (upstream logic:
    ``os.listdir``, single directory level, unsorted, non-recursive)."""
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(_SUFFIX)
    ]


def _sample_of(file_path: str) -> str:
    # upstream: file_name.replace("_ReadsPerGene.out.tab", "") — replace()
    # (not removesuffix) kept bug-compat: internal occurrences of the
    # suffix are stripped too.
    return os.path.basename(file_path).replace(_SUFFIX, "")


def _resolve_workers(num_processes) -> int:
    if num_processes is None:
        # upstream runs ThreadPoolExecutor() with Python's default
        # max_workers = min(32, (os.cpu_count() or 1) + 4)
        want = min(_WORKERS_CAP, (os.cpu_count() or 1) + 4)
    else:
        want = int(num_processes)
    return max(1, min(_WORKERS_CAP, want))


def _parse_star_file_fast(path: str):
    """Parse one ``*_ReadsPerGene.out.tab`` into ``(labels, counts)``.

    ``labels`` is the list of raw column-0 token bytes (ALL rows — the four
    leading STAR stat rows are data rows here, exactly as upstream);
    ``counts`` is an int64 array of the column-1 values. Raises
    :class:`_NeedsPandasPath` for any input shape not proven equivalent to
    the pandas parse.

    Mechanics: the strict per-line tab-count check guarantees
    ``data.split(b"\t")`` yields a fixed piece stride. Column 1 is a clean
    stride slice whenever the file has >= 3 columns; column 0 shares its
    pieces with the previous line's LAST column (separated by ``\\n``) —
    joining those pieces with ``\\n`` and splitting once exposes both token
    streams. 2-column files (stride 1) take both columns from the merged
    stream. All at C speed, no per-row Python.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if b"\r" in data:
        raise _NeedsPandasPath("CR byte present")
    if not data.endswith(b"\n"):
        data = data + b"\n"  # pandas tolerates a missing final newline
    if len(data) >= 2 ** 31:
        raise _NeedsPandasPath("file too large for the fast path")
    buf = np.frombuffer(data, dtype=np.uint8)
    nl = np.flatnonzero(buf == 10)
    if nl.size < 1:
        raise _NeedsPandasPath("no data rows")
    tb = np.flatnonzero(buf == 9)
    row_of_tab = np.searchsorted(nl, tb, side="left")
    per_row = np.bincount(row_of_tab, minlength=nl.size)
    stride = int(per_row[0])
    if stride < 1 or not bool((per_row == stride).all()):
        # ragged rows or blank lines: pandas skips blank lines and raises
        # its own errors on ragged ones — let the original parser decide
        raise _NeedsPandasPath("ragged or tab-less line layout")
    n = int(nl.size)
    pieces = data.split(b"\t")
    if len(pieces) != n * stride + 1:
        raise _NeedsPandasPath("unexpected piece count")

    # column 0 (merged with each line's LAST column via '\n')
    merged = pieces[0::stride]
    if len(merged) != n + 1:
        raise _NeedsPandasPath("merged-piece count")
    tokens = b"\n".join(merged).split(b"\n")
    if len(tokens) != 2 * n + 1:
        raise _NeedsPandasPath("col0 token layout")
    labels = tokens[0:2 * n:2]
    if len(labels) != n:
        raise _NeedsPandasPath("col0 token count")
    # column 1: clean stride slice for >= 3 columns; merged stream for 2
    counts_toks = pieces[1::stride] if stride >= 2 else tokens[1:2 * n:2]
    if len(counts_toks) != n:
        raise _NeedsPandasPath("col1 token count")

    # ---- column-0 guards (pandas must see raw unique str labels) ----
    lab_set = set(labels)
    if len(lab_set) != n:
        raise _NeedsPandasPath("duplicate ID labels")
    if lab_set & _NA_TOKENS:
        raise _NeedsPandasPath("pandas NA-value token in the ID column")
    joined0 = b"\n".join(labels)
    arr0 = np.frombuffer(joined0, dtype=np.uint8)
    if arr0.size == 0 or bool(_ID_NUMERIC_LUT[arr0].all()):
        raise _NeedsPandasPath("numeric-form ID column (dtype inference)")

    # ---- column-1 guards (pandas must infer int64 with canonical tokens) ----
    if max(map(len, counts_toks)) > _MAX_INT_TOKEN:
        raise _NeedsPandasPath("overlong count token")
    text1 = b" ".join(counts_toks)
    arr1 = np.frombuffer(text1, dtype=np.uint8)
    if arr1.size == 0 or not bool(_INTBYTE_LUT[arr1].all()):
        raise _NeedsPandasPath("non-integer-form count column")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = np.fromstring(text1, dtype=np.float64, sep=" ")
    if arr.size != n:
        raise _NeedsPandasPath("count parse count mismatch")
    counts = arr.astype(np.int64)
    # canonical-form round trip: str(int64) must equal every source token —
    # pins pandas' int64 inference (no leading zeros / '+' / whitespace /
    # '5-' forms), exact float64 representability (<= 2**53) and hence the
    # ".0" tokens pandas writes after the NaN-driven float64 upcast
    if b" ".join(np.char.encode(counts.astype(str))) != text1:
        raise _NeedsPandasPath("count token round-trip mismatch")
    return labels, counts


def _assemble_fast(parsed, order, samples):
    """Reproduce ``pd.concat(frames, axis=1)`` for the guarded canonical
    form: union-append row order, preallocated matrix, per-column dtype."""
    labs = [parsed[fp][0] for fp in order]
    vals = [parsed[fp][1] for fp in order]

    # union row order: file-0 labels as-is, then each later file's new
    # labels appended in within-file order — the Index.union(sort=False)
    # chain concat(axis=1, sort=False) performs (verified against the
    # golden output's stat-row layout: file-0 stats first, later files'
    # stats appended after the shared gene block)
    union = list(labs[0])
    seen = set(union)
    for l in labs[1:]:
        new = [x for x in l if x not in seen]
        if new:
            union.extend(new)
            seen.update(new)
    del seen
    try:
        names = [u.decode("utf-8") for u in union]
    except UnicodeDecodeError:
        raise _NeedsPandasPath("non-utf8 ID label") from None

    n_rows = len(union)
    n_files = len(order)
    idx_bytes = pd.Index(union)
    mat = np.full((n_rows, n_files), np.nan, dtype=np.float64)
    for j, l in enumerate(labs):
        rows = idx_bytes.get_indexer(np.array(l, dtype=object))
        if rows.min() < 0:  # pragma: no cover - the union covers every label
            raise _NeedsPandasPath("label missing from the union index")
        mat[rows, j] = vals[j]

    index = pd.Index(names, name="ID")
    data = {}
    for j, s in enumerate(samples):
        col = mat[:, j]
        if not np.isnan(col).any():
            # column covers the whole union: pandas concat keeps its int64
            # dtype (no NaN introduced) -> integer tokens on disk
            col = col.astype(np.int64)
        data[s] = col
    return pd.DataFrame(data, index=index)


def _merge_pandas_path(files, order, workers):
    """Original-statements merge (bug-compatible), with deterministic
    sorted-sample column order. Used whenever the fast path's equivalence
    guards trip."""
    from iobrpy.workflow.merge_star_count import _read_star_file

    loaded = {}
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(_read_star_file, fp): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="Reading STAR count files (pandas)", unit="file"):
            loaded[futs[fut]] = fut.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return pd.concat([loaded[p] for p in order], axis=1)


def merge_star_count(path, project, num_processes=None, verbose=True):
    """Merge all ``*_ReadsPerGene.out.tab`` in *path* into one count matrix.

    Bit-exact drop-in for the ``iobrpy.workflow.merge_star_count`` CLI on
    the file output (decompressed content identical after column alignment;
    output columns sorted by sample name and the union row order anchored
    to the sorted-first sample instead of the original's nondeterministic
    ``os.listdir``/``as_completed`` orders). The four leading STAR stat
    rows are NOT skipped — the upstream pollution defect is preserved.

    Parameters
    ----------
    path : str
        Directory (single level, non-recursive) containing the STAR count
        files; also the destination of the output file (upstream forces
        output dir == input dir).
    project : str
        Output prefix: ``<project>.STAR.count.tsv.gz``.
    num_processes : int, optional
        Loader threads for the fallback pandas path (the upstream CLI has
        no such flag and uses Python's ThreadPoolExecutor default).
        ``None`` → ``min(32, (os.cpu_count() or 1) + 4)``; always capped
        at 32. The fast path spawns no threads.
    verbose : bool, default True
        Reproduce the upstream console protocol ("Saving merged matrix...",
        head preview, row/column counts, "Saved to:", IOBRpy banner).

    Returns
    -------
    pandas.DataFrame or None
        The matrix as written: rows = union of per-file ID labels (gene
        rows plus the unpurged stat rows; index name ``ID``) x samples
        (sorted columns); per-column dtype int64 (column covers the union)
        or float64 (NaN-padded). ``None`` when no STAR file was found
        (upstream prints a notice and writes nothing).
    """
    files = _list_star_files(path)
    if not files:
        if verbose:
            print("No files ending with '_ReadsPerGene.out.tab' were found "
                  "in the given path.")
            _print_iobrpy_banner()
        return None

    order = sorted(files, key=lambda p: (_sample_of(p), p))
    samples = [_sample_of(p) for p in order]
    workers = _resolve_workers(num_processes)

    # ---- fast parse (byte surgery + np.fromstring), sequential ----
    parsed = {}
    fast_ok = len(set(samples)) == len(samples)  # duplicate sample names:
    # upstream concat would emit duplicate columns -> original statements
    if fast_ok:
        try:
            for fp in tqdm(order, desc="Reading STAR count files", unit="file"):
                parsed[fp] = _parse_star_file_fast(fp)
        except _NeedsPandasPath:
            fast_ok = False

    merged_df = None
    if fast_ok:
        try:
            merged_df = _assemble_fast(parsed, order, samples)
        except _NeedsPandasPath:
            merged_df = None
    parsed.clear()
    if merged_df is None:
        merged_df = _merge_pandas_path(files, order, workers)

    # ---- persist with the EXACT upstream write path ----
    output_file = os.path.join(path, f"{project}.STAR.count.tsv.gz")
    if verbose:
        print("Saving merged matrix...")
    with gzip.open(output_file, "wt", compresslevel=5) as f:
        merged_df.to_csv(f, sep="\t")

    if verbose:
        print("Head of merged file:")
        print(merged_df.head())
        print("Number of rows:", merged_df.shape[0])
        print("Number of columns:", merged_df.shape[1])
        print(f"Saved to: {output_file}")
        _print_iobrpy_banner()
    return merged_df


def merge_star_count_original(path, project):
    """Run the untouched upstream CLI ``main()`` (argv rewrite, exactly how
    ``iobrpy.main`` dispatches this subcommand). Escape hatch for
    ``backend='python'``; returns None since upstream returns nothing."""
    from iobrpy.workflow import merge_star_count as _orig

    argv = ["merge_star_count", "--path", str(path), "--project", str(project)]
    saved = sys.argv
    try:
        sys.argv = argv
        _orig.main()
    finally:
        sys.argv = saved
    return None
