"""merge_salmon_fast: accelerated drop-in for the ``iobrpy.workflow.merge_salmon``
CLI stage (``python -m iobrpy.main merge_salmon --path_salmon ... --project ...``).

Same behaviour as upstream (merge_salmon.py, IOBRpy 0.2.0): recursively
discover ``quant.sf`` under *path_salmon* (``os.walk`` + ``sorted``), read the
``Name`` / ``TPM`` / ``NumReads`` columns of every file (sample name = parent
directory name), outer-join them across samples on ``Name`` and persist

    <path_salmon>/<project>_salmon_tpm.tsv.gz
    <path_salmon>/<project>_salmon_count.tsv.gz

as gzip(level 5) TSVs written with ``pandas.DataFrame.to_csv(f, sep="\\t")``
(index=True, index name ``Name``) — the EXACT upstream write path, so float
tokens (pandas shortest-repr), the ``Name`` first-column header and the row
order are byte-identical to the original decompressed outputs.

Bit-exactness strategy (validated on the frozen 8 x 60,000-transcript
official-format quant.sf fixture: decompressed content sha256 identical to
the recorded iobrpy 0.2.0 baseline after column alignment):

  1. Reading: per-file ``pd.read_csv`` is replaced by C-level byte surgery —
     one ``data.split(b"\\t")`` per file (valid because the strict per-line
     tab-count check gives the pieces a fixed stride), stride slices for the
     ``Name`` and float columns, ``b" ".join`` + one ``np.fromstring(sep=' ')``
     per float column. ``np.fromstring``'s ASCII strtod is bitwise identical
     to pandas 2.3.3's default C float converter on ALL 960,000 fixture
     values and on an 85,024-token adversarial corpus (fixed / scientific
     notation, 1-20-digit mantissas, exponents -320..308, signs, zeros,
     inf/nan spellings, over/underflow): 0 mismatches.
  2. Merging: ``pd.concat(axis=1)`` is replaced by filling a preallocated
     (n_transcripts x n_samples) float64 matrix — valid only while every
     file's ``Name`` token list is byte-identical to the first sample's
     (same rows, same order), which the real salmon layout always satisfies.
     No index alignment work, no intermediate Series.
  3. Anything outside the guarded canonical form falls back, for the WHOLE
     run, to the original pandas statements (upstream ``_load_one_quant`` +
     ``pd.concat``), keeping error messages, outer-join union semantics and
     dtype quirks bug-compatible: CR bytes, ragged/empty rows, missing or
     duplicated header fields, ``Name`` not in column 0, '#'-comment headers
     (upstream dies with ``ValueError: Usecols do not match columns`` — the
     fallback reproduces it), integer-form or numeric-looking columns that
     would flip pandas' dtype inference, float tokens longer than the
     longest tested-equivalent form, empty/junk fields (``np.fromstring``
     count guard), duplicate ``Name`` entries, differing row sets/order
     across samples, >2GiB files and non-UTF-8 text.

Documented deviation (per the porting contract): the ORIGINAL collects
samples through ``concurrent.futures.as_completed`` → the output column
(sample) order is nondeterministic run-to-run. Here output columns are
sorted by (sample name, quant.sf path): same column set, same values in
every cell, deterministic order. Console preview lines therefore also show
sorted columns.

Thread discipline: the fast parser runs sequentially — its C-level bytes
surgery and ``np.fromstring`` are GIL-bound and measured loader threads
strictly hurt (8 x 60k fixture: 0.43 s sequential vs 2.2 s at 8 threads).
ThreadPoolExecutor is kept where it provably helps: the fallback pandas
loader (mirrors upstream, whose C tokenizer releases the GIL) and the two
independent gzip writer streams (0.71 s vs 0.99 s sequential, byte-equal
decompressed content). Worker counts never exceed 32 (deployment
guardrail), while the original defaults to ``os.cpu_count()``; this affects
I/O overlap only, never output bytes.

Rust read path (``engine='rust'`` / ``'auto'``): ``_rust.merge_salmon_parse``
parses all quant.sf files in parallel (rayon over files, GIL released) with
the crate's verbatim pandas ``precise_xstrtod`` port and assembles the
(n_transcripts x n_samples) float64 matrices directly. Its accept domain is
a strict subset of the sequential parser's, restricted to float tokens where
every converter in play (the pandas C-engine fast path, its
overflow-triggered correctly-rounded slow path, ``float_precision=
'round_trip'``, CPython ``float()``, ``np.fromstring``) is provably
bit-identical: clean float grammar, <=15 mantissa digits, decimal scale
exponent in [-22, 22]. Any declined run falls back to the UNTOUCHED
sequential path below, so the engine choice never changes output bytes.
Kernel values were re-validated bitwise against ``pd.read_csv`` on the
frozen fixture (1,920,000 cells, 0 mismatches) and on the 85k-token
adversarial corpus (33,712 in-domain tokens, 0 mismatches three-way).
Rust parse threads are capped at 16; results are thread-count invariant
(validated n_threads 1/4/8/16 bitwise identical).
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

__all__ = ["merge_salmon", "merge_salmon_original"]

_USECOLS = ("Name", "TPM", "NumReads")
# Longest token in the parser-equivalence stress corpus was 27 bytes
# ("-9.9999999999999999999e-320", bitwise-equal to pandas); beyond this
# bound the run routes to the original pandas parser.
_MAX_FLOAT_TOKEN = 32
_WORKERS_CAP = 32  # deployment guardrail (upstream uses os.cpu_count())
# Rust parse threads: per-file parallelism saturates at the file count and
# the stage is memory-bandwidth bound; 16 keeps it inside the deployment
# guardrail (results are thread-count invariant, validated 1/4/8/16).
_RUST_PARSE_THREADS_CAP = 16

# bytes that could make pandas infer a NUMERIC dtype for the Name column
# (digits, sign, point, exponent letters) plus the '\n' used to join names
# for the check -> an all-match column takes the original pandas path
_NAME_NUMERIC_LUT = np.zeros(256, dtype=bool)
_NAME_NUMERIC_LUT[list(b"0123456789+-.eE\n")] = True
# bytes proving a value column is float-form for pandas ('.', exponent or
# inf/nan letters); a column without any of them would be inferred int64 by
# read_csv -> pandas path
_FLOATMARK_LUT = np.zeros(256, dtype=bool)
_FLOATMARK_LUT[list(b".eEinNaAiIfF")] = True


class _NeedsPandasPath(Exception):
    """Internal signal: this quant.sf must go through the original parser."""


def _print_iobrpy_banner():
    """Print the IOBRpy banner (verbatim upstream behaviour, after output paths)."""
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


def _find_quant_files(root: str):
    """Recursively find all ``quant.sf`` under ``root`` (upstream logic)."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "quant.sf" in filenames:
            hits.append(os.path.join(dirpath, "quant.sf"))
    return sorted(hits)


def _sample_of(quant_path: str) -> str:
    return os.path.basename(os.path.dirname(quant_path))


def _resolve_workers(num_processes) -> int:
    if num_processes is None:
        want = os.cpu_count() or 4
    else:
        want = int(num_processes)
    return max(1, min(_WORKERS_CAP, want))


def _parse_quant_fast(path: str):
    """Parse one quant.sf into ``(names, tpm, numreads)``.

    ``names`` is the list of raw ``Name`` token bytes; ``tpm`` / ``numreads``
    are float64 arrays. Raises :class:`_NeedsPandasPath` for any input shape
    not proven equivalent to the pandas parse.

    Mechanics: the strict per-line tab-count check guarantees
    ``data.split(b"\\t")`` yields exactly ``ncols - 1`` pieces per line, so
    column ``k`` of every line lives at a fixed stride. The last column
    shares its pieces with the next line's ``Name`` (separated by ``\\n``);
    joining those pieces with ``\\n`` and splitting once exposes ``Name`` and
    the last column as the odd/even token streams — all at C speed, no
    per-row Python and no numpy gathers.
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
    if nl.size < 2:
        raise _NeedsPandasPath("no data rows")
    h_end = int(nl[0])
    try:
        header = data[:h_end].decode("utf-8").split("\t")
    except UnicodeDecodeError:
        raise _NeedsPandasPath("non-utf8 header") from None
    ncols = len(header)
    if ncols < 3 or len(set(header)) != ncols:
        raise _NeedsPandasPath("degenerate or duplicated header")
    for need in _USECOLS:
        if need not in header:
            # upstream dies here with ValueError(Usecols do not match
            # columns) — the pandas fallback reproduces it verbatim
            raise _NeedsPandasPath("usecol missing from header")
    i_name = header.index("Name")
    i_tpm = header.index("TPM")
    i_cnt = header.index("NumReads")
    if i_name != 0:
        raise _NeedsPandasPath("Name is not the first column")

    tb = np.flatnonzero(buf == 9)
    # every line (header AND data) must carry exactly ncols-1 tabs, which
    # also rejects blank lines (pandas skips those, the stride layout must
    # not) and pins the piece-stride algebra below
    row_of_tab = np.searchsorted(nl, tb, side="left")
    per_row = np.bincount(row_of_tab, minlength=nl.size)
    if not bool((per_row == ncols - 1).all()):
        raise _NeedsPandasPath("ragged tab layout")
    n = nl.size - 1
    stride = ncols - 1

    pieces = data.split(b"\t")
    if len(pieces) != (n + 1) * stride + 1:
        raise _NeedsPandasPath("unexpected piece count")
    # merged pieces: last column of line m-1 + '\n' + Name of line m
    # (m = 1..n), then the file tail (last column of line n + '\n')
    cat = b"\n".join(pieces[stride::stride])
    tokens = cat.split(b"\n")
    if len(tokens) != 2 * n + 2:
        raise _NeedsPandasPath("name/last-column token layout")
    names = tokens[1:-1:2]
    last_col = tokens[2::2]
    if len(names) != n or len(last_col) != n:
        raise _NeedsPandasPath("name/last-column token count")

    def _mid_col(k):
        toks = pieces[k::stride]
        if len(toks) != n + 1:
            raise _NeedsPandasPath("stride layout")
        return toks[1:]

    def _to_floats(toks, tag):
        text = b" ".join(toks)
        if not bool(_FLOATMARK_LUT[np.frombuffer(text, dtype=np.uint8)].any()):
            # all-integer-form column: read_csv would infer int64 and to_csv
            # would re-print tokens without ".0" -> original parser
            raise _NeedsPandasPath(f"integer-form {tag} column")
        if max(map(len, toks)) > _MAX_FLOAT_TOKEN:
            raise _NeedsPandasPath(f"overlong {tag} token")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = np.fromstring(text, dtype=np.float64, sep=" ")
        if arr.size != n:
            # empty or unparsable field: pandas gives NaN / raises its own
            # ParserError — let the original parser decide
            raise _NeedsPandasPath(f"{tag} parse count mismatch")
        return arr

    tpm = _to_floats(last_col if i_tpm == ncols - 1 else _mid_col(i_tpm), "TPM")
    cnt = _to_floats(last_col if i_cnt == ncols - 1 else _mid_col(i_cnt), "NumReads")
    return names, tpm, cnt


def _merge_pandas_path(quants, order, workers):
    """Original-statements merge (bug-compatible), with deterministic
    (sample, path)-sorted column order. Used whenever the fast path's
    equivalence guards trip."""
    from iobrpy.workflow.merge_salmon import _load_one_quant

    loaded = {}
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(_load_one_quant, q): q for q in quants}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="Loading quant.sf (pandas)", unit="file"):
            loaded[futs[fut]] = fut.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    tpm_df = pd.concat([loaded[p][0] for p in order], axis=1)
    cnt_df = pd.concat([loaded[p][1] for p in order], axis=1)
    return tpm_df, cnt_df


def _rust_parse_fn(strict: bool):
    """Resolve ``iobrx._rust.merge_salmon_parse`` (None when unavailable).

    ``strict=True`` (``engine='rust'``) raises when the native extension or
    the kernel is missing (stale ``.so``); ``strict=False``
    (``engine='auto'``) degrades silently to the sequential Python parser.
    """
    try:
        import iobrx._rust as _ir

        fn = getattr(_ir, "merge_salmon_parse", None)
    except (ImportError, OSError):
        fn = None
    if fn is None and strict:
        raise RuntimeError(
            "merge_salmon engine='rust' requires iobrx._rust.merge_salmon_parse "
            "(native extension missing or stale); rebuild the extension or use "
            "engine='python'"
        )
    return fn


def _merge_rust_engine(fn, order, samples, workers):
    """Rust read path: parallel quant.sf parse + matrix assembly in
    ``_rust.merge_salmon_parse`` (values bitwise equal to the gold
    ``pd.read_csv`` parse; see the module docstring for the accept domain).

    Returns ``(tpm_df, cnt_df)``, or None when the kernel declines the input
    (any equivalence guard trips) — the caller then runs the untouched
    sequential fast path, which itself declines to the original pandas
    statements, so ``engine='rust'`` output is byte-identical to
    ``engine='python'`` on every input.
    """
    n_threads = max(1, min(workers, _RUST_PARSE_THREADS_CAP))
    ok, _reason, names_cat, tpm, cnt = fn([str(p) for p in order], n_threads)
    if not ok:
        return None
    names = np.asarray(names_cat).tobytes().decode("utf-8").split("\n")
    index = pd.Index(names, name="Name")
    tpm_df = pd.DataFrame(np.asarray(tpm), index=index, columns=samples)
    cnt_df = pd.DataFrame(np.asarray(cnt), index=index, columns=samples)
    return tpm_df, cnt_df


def merge_salmon(path_salmon, project, num_processes=None, verbose=True,
                 engine="auto"):
    """Merge all ``quant.sf`` under *path_salmon* into TPM / count matrices.

    Bit-exact drop-in for the ``iobrpy.workflow.merge_salmon`` CLI on the
    file outputs (decompressed content identical; output columns sorted by
    sample name instead of the original's nondeterministic ``as_completed``
    order).

    Parameters
    ----------
    path_salmon : str
        Root directory searched recursively for ``quant.sf``; also the
        destination of the two output files (upstream forces output dir =
        input dir).
    project : str
        Output prefix: ``<project>_salmon_tpm.tsv.gz`` and
        ``<project>_salmon_count.tsv.gz``.
    num_processes : int, optional
        Loader threads (upstream name). ``None`` → ``os.cpu_count()``;
        always capped at 32 (deployment guardrail). Also bounds the Rust
        parse threads (additionally capped at 16); the sequential Python
        parser stays single-threaded by design.
    verbose : bool, default True
        Reproduce the upstream console protocol (head previews, "Saving…"
        lines, output paths, IOBRpy banner).
    engine : {'auto', 'rust', 'python'}, default 'auto'
        Read-path selector. ``'python'`` = the sequential numpy-fromstring
        fast parser below. ``'rust'`` = parallel Rust parse + assembly
        (``_rust.merge_salmon_parse``; RuntimeError when the kernel is
        missing). ``'auto'`` = ``'rust'`` when the kernel is importable,
        else ``'python'``. Output bytes are identical for every engine on
        every input: the Rust kernel's accept domain is a strict subset of
        the sequential parser's and declines whole runs to it (which itself
        declines to the original pandas statements) whenever any
        equivalence guard trips.

    Returns
    -------
    (tpm_df, cnt_df) : tuple of pandas.DataFrame
        Transcript x sample float64 matrices as written to disk, columns in
        sorted (sample, path) order, index name ``Name``. ``(None, None)``
        when no ``quant.sf`` was found (upstream prints a notice and exits
        without writing files).
    """
    if engine not in {"auto", "rust", "python"}:
        raise ValueError("engine must be 'auto', 'rust', or 'python'")
    quants = _find_quant_files(path_salmon)
    if not quants:
        if verbose:
            print("No quant.sf files were found under the given --path_salmon.")
            _print_iobrpy_banner()
        return None, None

    order = sorted(quants, key=lambda p: (_sample_of(p), p))
    samples = [_sample_of(p) for p in order]
    workers = _resolve_workers(num_processes)

    # ---- rust read path (parallel parse + assembly in _rust) ----
    tpm_df = cnt_df = None
    rust_fn = None
    if engine == "rust":
        rust_fn = _rust_parse_fn(strict=True)
    elif engine == "auto":
        rust_fn = _rust_parse_fn(strict=False)
    if rust_fn is not None:
        # one kernel call parses+assembles every file in parallel; the tqdm
        # bar completes when the call returns (stdout is not under contract)
        bar = tqdm(total=len(order), desc="Loading quant.sf (rust)", unit="file")
        try:
            res = _merge_rust_engine(rust_fn, order, samples, workers)
        finally:
            bar.update(len(order))
            bar.close()
        if res is not None:
            tpm_df, cnt_df = res

    if tpm_df is None:
        # engine='python', or the Rust kernel declined this input (any
        # equivalence guard tripped): the untouched sequential fast parse.
        # ---- fast parse (byte surgery + np.fromstring) ----
        # Sequential on purpose: the C-level bytes operations (split/join)
        # and np.fromstring are GIL-bound, and measured loader threads
        # strictly hurt (8 files: 0.43 s sequential vs 0.57 s @2 / 2.2 s @8
        # threads on 224-core Xeon). num_processes still governs the
        # fallback pandas loader (which does release the GIL, mirroring
        # upstream) and the dual-stream writer below.
        parsed = {}
        fast_ok = True
        try:
            for q in tqdm(quants, desc="Loading quant.sf", unit="file"):
                parsed[q] = _parse_quant_fast(q)
        except _NeedsPandasPath:
            fast_ok = False

        if fast_ok:
            ref_names = parsed[order[0]][0]
            n = int(parsed[order[0]][1].size)
            # pandas-dtype guards on the reference Name column (every other
            # file is byte-compared against it, so one check covers all)
            joined = b"\n".join(ref_names)
            arr = np.frombuffer(joined, dtype=np.uint8)
            if arr.size == 0 or bool(_NAME_NUMERIC_LUT[arr].all()):
                fast_ok = False  # pandas would infer a numeric/NaN Name column
            elif len(set(ref_names)) != n:
                fast_ok = False  # duplicate labels: original concat semantics
            else:
                try:
                    names = [nm.decode("utf-8") for nm in ref_names]
                except UnicodeDecodeError:
                    fast_ok = False
        if fast_ok:
            n_files = len(order)
            tpm_mat = np.empty((n, n_files), dtype=np.float64)
            cnt_mat = np.empty((n, n_files), dtype=np.float64)
            for j, q in enumerate(order):
                nm, tpm, cnt = parsed[q]
                if j and nm != ref_names:
                    # differing row sets/order: upstream outer-joins via concat
                    fast_ok = False
                    break
                tpm_mat[:, j] = tpm
                cnt_mat[:, j] = cnt
            if fast_ok:
                index = pd.Index(names, name="Name")
                tpm_df = pd.DataFrame(tpm_mat, index=index, columns=samples)
                cnt_df = pd.DataFrame(cnt_mat, index=index, columns=samples)
        parsed.clear()

    if tpm_df is None:
        tpm_df, cnt_df = _merge_pandas_path(quants, order, workers)

    # ---- persist with the EXACT upstream write path ----
    out_tpm = os.path.join(path_salmon, f"{project}_salmon_tpm.tsv.gz")
    out_cnt = os.path.join(path_salmon, f"{project}_salmon_count.tsv.gz")

    if verbose:
        print("TPM head:")
        print(tpm_df.head())
        print("Count head:")
        print(cnt_df.head())
        print("Saving TPM matrix...")
        print("Saving Count matrix...")

    def _write(df, path):
        with gzip.open(path, "wt", compresslevel=5) as f:
            df.to_csv(f, sep="\t")

    if workers >= 2:
        # two independent streams; per-file bytes are deterministic
        with ThreadPoolExecutor(max_workers=2) as wex:
            f1 = wex.submit(_write, tpm_df, out_tpm)
            f2 = wex.submit(_write, cnt_df, out_cnt)
            f1.result()
            f2.result()
    else:
        _write(tpm_df, out_tpm)
        _write(cnt_df, out_cnt)

    if verbose:
        # IMPORTANT: output paths FIRST...
        print(f"Saved to (TPM): {out_tpm}", flush=True)
        print(f"Saved to (Count): {out_cnt}", flush=True)
        # ...THEN the banner
        _print_iobrpy_banner()
    return tpm_df, cnt_df


def merge_salmon_original(path_salmon, project, num_processes=None):
    """Run the untouched upstream CLI ``main()`` (argv rewrite, exactly how
    ``iobrpy.main`` dispatches this subcommand). Escape hatch for
    ``backend='python'``; returns ``(None, None)`` since upstream returns
    nothing."""
    from iobrpy.workflow import merge_salmon as _orig

    argv = ["merge_salmon", "--path_salmon", str(path_salmon),
            "--project", str(project)]
    if num_processes is not None:
        argv += ["--num_processes", str(int(num_processes))]
    saved = sys.argv
    try:
        sys.argv = argv
        _orig.main()
    finally:
        sys.argv = saved
    return None, None
