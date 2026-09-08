"""trust4_fast: accelerated drop-in for the ``iobrpy.workflow.trust4``
CLI stage (``python -m iobrpy.main trust4 [TRUST4 options...]``).

Same behaviour as upstream (trust4.py, IOBRpy 0.2.0). The wrapper:

* resolves the TRUST4 runner exactly as upstream — ``os.environ.get(
  "TRUST4_BIN", "run-trust4")`` + ``shutil.which`` (missing -> the identical
  stderr message + exit code 127); ``trust4_bin=`` only overrides that
  resolution (default ``None`` = upstream behaviour);
* accepts the FULL upstream CLI surface through the verbatim argparse
  parser (``-b`` file/dir, ``-1/-2``, ``-u``, ``--fqdir``, ``-f``/``--ref``
  with the ``iobrpy.resources`` defaults ``hg38_bcrtcr.fa`` /
  ``human_IMGT+C.fa`` extracted to atexit-cleaned temp dirs, ``-o``/``--od``,
  ``-t``/``-k``, barcode/UMI options, mode flags, ``--stage``/``--clean``)
  plus ``parse_known_args`` pass-through of unknown options into every
  ``run-trust4`` call;
* batch modes (``-b <dir>`` / ``--fqdir``): SEQUENTIAL per-sample loop with
  tqdm progress (upstream scheduling, verbatim), per-sample output dir
  ``<o>/<folder>`` with prefix ``TRUST_<prefix_base>`` built by the upstream
  regexes, ``<folder>.TRUST4.done`` resume flags, ``run-trust4`` inheriting
  stdout/stderr, and the batch exit code = last failing sample's rc;
* single-run mode: one ``run-trust4`` call, then immune post-processing on
  the upstream-inferred output root (``--od`` > ``-o`` dir > ``dirname(-o)``
  > cwd);
* after ALL runs: ``_run_immune_postprocessing`` stages every recursively
  found ``*_report.tsv`` into a temp dir as ``<derived-sample>_report.tsv``
  (upstream naming regexes + collision counters + symlink-or-copy staging)
  and calls the immune post-processor, writing
  ``<output_root>/trust4_immdata.csv`` and
  ``<output_root>/trust4_immune_indices.csv``.

ACCELERATED periphery (the only computation in this stage that is not the
external binary): ``process_immune_data_batch_fast`` replaces the upstream
``process_immune_data_batch``. It (1) reads the per-sample ``*_report.tsv``
files through a bounded ThreadPoolExecutor (pandas' C tokenizer releases the
GIL; frames land at their INPUT index so the ``pd.concat`` order — and
therefore the ``trust4_immdata.csv`` bytes — equals the sequential
upstream order), and (2) hoists the per-group ``astype(str).str.len()`` /
``to_numpy(dtype=float)`` conversions out of the per-sample diversity loop
into single whole-column passes, slicing per-group views by positional
index (``pd.factorize(sort=True)`` + stable argsort reproduces
``groupby("Sample")``'s sorted-key, original-row-order iteration exactly;
NaN keys are excluded like ``groupby(dropna=True)``). Every per-sample
statistic then runs the ORIGINAL numpy statement sequence on the IDENTICAL
ordered float64 value array, so all sums/sorts/log values — and the CSV
float tokens written by the untouched ``to_csv`` calls — are bit-identical
(verified byte-for-byte against the original on synthetic multi-sample
report fixtures and on the frozen real-data TRUST4 gate, including the
degenerate header-only report -> 1-byte ``'\\n'`` indices file). Any
hoisted-conversion failure (pathological dtypes) falls back to the verbatim
original loop; ``engine='original'`` forces it.

Known nondeterminism (identical in original and port): ``glob.glob`` file
order drives the immdata row-block order in BOTH implementations (same
call, same staging dir contents); run-trust4's own console log embeds
timestamps and temp reference paths — the DATA outputs (report.tsv,
cdr3.out, airr tsv, fa, fq) are deterministic for a fixed binary, reference
and ``-t``.
"""
from __future__ import annotations

import argparse
import atexit
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional immune post-processing (TRUST4 report -> immune indices) — the
# ORIGINAL functions stay the semantic reference (aliases, gini formula).
try:  # pragma: no cover - optional dependency
    from iobrpy.utils.process_immune_data_batch import (
        process_immune_data_batch,
        standardize_columns,
        gini_coefficient,
    )
except Exception:  # pragma: no cover
    process_immune_data_batch = None  # type: ignore
    standardize_columns = None  # type: ignore
    gini_coefficient = None  # type: ignore

# tqdm progress bar (with graceful fallback if not installed) — verbatim
# upstream shim
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class _SimpleTqdm:
        """Minimal fallback when tqdm is unavailable."""

        def __init__(self, total=None, desc=None, unit=None, dynamic_ncols=None):
            self.total = total or 0
            self.desc = desc or "progress"
            self.n = 0

        def __enter__(self):
            print(f"[{self.desc}] total={self.total}")
            return self

        def update(self, n: int = 1):
            self.n += n
            print(f"[{self.desc}] {self.n}/{self.total}", flush=True)

        def set_postfix_str(self, s: str):
            print(f"[{self.desc}] processing {s}", flush=True)

        def __exit__(self, exc_type, exc, tb):
            pass

    def tqdm(*args, **kwargs):  # type: ignore
        return _SimpleTqdm(*args, **kwargs)


__all__ = ["main", "trust4_original", "process_immune_data_batch_fast",
           "print_iobrpy_banner"]


def print_iobrpy_banner():
    """Print the IOBRpy banner — the upstream ``iobrpy.main`` dispatch
    prints it after the trust4 stage ends in ``SystemExit`` (ANY exit code),
    before re-raising the code; the iobrx public wrapper reproduces that."""
    print("   ")
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
        print("#########################################################")
        print(" IOBRpy: Immuno-Oncology Biological Research using Python ")
        print(" If you encounter any issues, please report them at ")
        print(" https://github.com/IOBR/IOBRpy/issues ")
        print("#########################################################")
        print(" Author: Haonan Huang, Dongqiang Zeng")
        print(" Email: interlaken@smu.edu.cn ")
        print("#########################################################")
    print("   ")

DEFAULT_F = "hg38_bcrtcr.fa"
DEFAULT_REF = "human_IMGT+C.fa"
RES_PKG = "iobrpy.resources"

_READ_WORKERS_CAP = 8  # report files are small; GIL-bound beyond this

_TRUST4_TMP_DIRS: List[str] = []


def _cleanup_tmp_dirs():
    for d in list(_TRUST4_TMP_DIRS):
        try:
            if d and os.path.isdir(d) and os.path.basename(d).startswith("iobrpy_trust4_"):
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    _TRUST4_TMP_DIRS.clear()


atexit.register(_cleanup_tmp_dirs)


def _extract_resource_to_tmp(name: str) -> Optional[str]:
    """Read a resource file from iobrpy.resources and write it to a temp file."""
    from importlib.resources import files
    try:
        data = files(RES_PKG).joinpath(name).read_bytes()
    except Exception:
        return None
    tmpdir = tempfile.mkdtemp(prefix="iobrpy_trust4_")
    _TRUST4_TMP_DIRS.append(tmpdir)
    outp = os.path.join(tmpdir, name)
    with open(outp, "wb") as f:
        f.write(data)
    return outp


def _append_opt(cmd: List[str], flag: str, value, is_bool: bool = False):
    """Append a CLI option to command if value is present."""
    if is_bool:
        if value:
            cmd.append(flag)
    else:
        if value is not None:
            cmd.extend([flag, str(value)])


def _list_bams(bamdir: str) -> List[str]:
    """List *.bam files (non-recursive) in bamdir."""
    if not os.path.isdir(bamdir):
        raise FileNotFoundError(f"-b path is a directory but not found: {bamdir}")
    out: List[str] = []
    for fn in sorted(os.listdir(bamdir)):
        if fn.endswith(".bam"):
            p = os.path.join(bamdir, fn)
            if os.path.isfile(p):
                out.append(p)
    return out


def _infer_sample_from_fastq_name(filename: str) -> Optional[Tuple[str, str]]:
    """Return (sample_prefix, 'R1'|'R2') from _1/_2 fastq/fq.gz suffixes."""
    suffixes = [
        ("_1.fastq.gz", "R1"),
        ("_2.fastq.gz", "R2"),
        ("_1.fq.gz", "R1"),
        ("_2.fq.gz", "R2"),
    ]

    for suffix, read_type in suffixes:
        if filename.endswith(suffix):
            return (filename[: -len(suffix)], read_type)
    return None


def _pair_fastqs(fqdir: str) -> Dict[str, Tuple[str, str]]:
    """Find paired FASTQs in fqdir (verbatim upstream pairing + errors)."""
    if not os.path.isdir(fqdir):
        raise FileNotFoundError(f"--fqdir not found: {fqdir}")

    r1_map: Dict[str, List[str]] = {}
    r2_map: Dict[str, List[str]] = {}

    for fn in sorted(os.listdir(fqdir)):
        if not (fn.endswith(".fastq.gz") or fn.endswith(".fq.gz")):
            continue
        parsed = _infer_sample_from_fastq_name(fn)
        if parsed is None:
            continue
        prefix, which = parsed
        full = os.path.join(fqdir, fn)
        if which == "R1":
            r1_map.setdefault(prefix, []).append(full)
        else:
            r2_map.setdefault(prefix, []).append(full)

    pairs: Dict[str, Tuple[str, str]] = {}
    samples = sorted(set(r1_map.keys()) | set(r2_map.keys()))
    missing: List[Tuple[str, int, int]] = []
    multi: List[Tuple[str, int, int]] = []
    for s in samples:
        r1s = r1_map.get(s, [])
        r2s = r2_map.get(s, [])
        if len(r1s) != 1 or len(r2s) != 1:
            if len(r1s) == 0 or len(r2s) == 0:
                missing.append((s, len(r1s), len(r2s)))
            else:
                multi.append((s, len(r1s), len(r2s)))
            continue
        pairs[s] = (r1s[0], r2s[0])

    errs: List[str] = []
    if missing:
        errs.append(
            "Unmatched sample pairs (need exactly one R1 and one R2): "
            + ", ".join([f"{s}[R1={n1},R2={n2}]" for s, n1, n2 in missing])
        )
    if multi:
        errs.append(
            "Ambiguous samples with multiple R1 or R2 (please merge lanes or reduce to one pair per sample): "
            + ", ".join([f"{s}[R1={n1},R2={n2}]" for s, n1, n2 in multi])
        )
    if errs:
        raise ValueError("; ".join(errs))

    return pairs


def _build_common_options(args, f_path: str, ref_path: Optional[str]) -> List[str]:
    """Build the TRUST4 options common to all runs (verbatim upstream)."""
    cmd: List[str] = []
    _append_opt(cmd, "-f", f_path)
    if ref_path:
        _append_opt(cmd, "--ref", ref_path)
    _append_opt(cmd, "-t", args.t)
    _append_opt(cmd, "-k", args.k)

    _append_opt(cmd, "--barcode", args.barcode)
    _append_opt(cmd, "--barcodeLevel", args.barcodeLevel)
    _append_opt(cmd, "--barcodeWhitelist", args.barcodeWhitelist)
    _append_opt(cmd, "--barcodeTranslate", args.barcodeTranslate)
    _append_opt(cmd, "--UMI", args.UMI)
    _append_opt(cmd, "--readFormat", args.readFormat)

    _append_opt(cmd, "--repseq", args.repseq, is_bool=True)
    _append_opt(cmd, "--contigMinCov", args.contigMinCov)
    _append_opt(cmd, "--minHitLen", args.minHitLen)
    _append_opt(cmd, "--mateIdSuffixLen", args.mateIdSuffixLen)
    _append_opt(cmd, "--skipMateExtension", args.skipMateExtension, is_bool=True)
    _append_opt(cmd, "--abnormalUnmapFlag", args.abnormalUnmapFlag, is_bool=True)
    _append_opt(cmd, "--assembleWithRef", args.assembleWithRef, is_bool=True)
    _append_opt(cmd, "--noExtraction", args.noExtraction, is_bool=True)
    _append_opt(cmd, "--outputReadAssignment", args.outputReadAssignment, is_bool=True)

    _append_opt(cmd, "--stage", args.stage)
    _append_opt(cmd, "--clean", args.clean)
    return cmd


def _infer_single_output_root(args) -> str:
    """Infer where TRUST4 wrote its single-run outputs (verbatim upstream)."""
    if getattr(args, "od", None):
        return args.od
    o_val = getattr(args, "o", None)
    if o_val and os.path.isdir(o_val):
        return o_val
    if o_val:
        o_dir = os.path.dirname(o_val)
        if o_dir and os.path.isdir(o_dir):
            return o_dir
    return os.getcwd()


def _collect_report_files(root: str) -> List[str]:
    """Recursively collect all *_report.tsv files under root (sorted)."""
    report_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("_report.tsv"):
                report_files.append(os.path.join(dirpath, fn))
    report_files.sort()
    return report_files


def _derive_sample_name_from_path(src: str, output_root: str) -> str:
    """Derive a clean sample name from a *_report.tsv path (verbatim)."""
    rel = os.path.relpath(src, output_root)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        # Sample lives in a per-sample subdirectory
        sample = parts[0]
    else:
        base = os.path.basename(src)
        if base.endswith("_report.tsv"):
            base = base[: -len("_report.tsv")]
        sample = base

    # Remove TRUST_ prefix if present
    if sample.startswith("TRUST_"):
        sample = sample[len("TRUST_"):]

    # Remove suffixes related to STAR-style alignment names
    sample = re.sub(
        r"([._]Aligned)\.sortedByCoord\.out$",
        "",
        sample,
        flags=re.IGNORECASE,
    )
    sample = re.sub(
        r"\.sortedByCoord\.out$",
        "",
        sample,
        flags=re.IGNORECASE,
    )

    return sample


# ---------------------------------------------------------------------------
# accelerated immune post-processing (bit-identical to the original)
# ---------------------------------------------------------------------------
def _indices_loop_original(immdata_combined: pd.DataFrame,
                           immune_indices_out_csv: str) -> pd.DataFrame:
    """The ORIGINAL per-sample diversity loop, verbatim (fallback path)."""
    results = []
    for sample_name, group in immdata_combined.groupby("Sample"):
        cnts = group["read_count"].to_numpy(dtype=float)
        cdr3 = group["CDR3_dna"].astype(str)
        nc = cdr3.str.len().to_numpy(dtype=float)

        Nreads = cnts.sum()

        if Nreads <= 0:
            Nclones = int((cnts != 0).sum())
            rec = {
                "Sample": sample_name,
                "Nreads": Nreads,
                "Nclones": Nclones,
                "Length_CDR3": np.nan,
                "Shannon_Index": np.nan,
                "Evenness": np.nan,
                "Top_clone": np.nan,
                "Second_top_clone": np.nan,
                "Rare_clone": np.nan,
                "Second_Rare_clone": np.nan,
                "Gini": np.nan,
                "Gini_Simpson": np.nan,
            }
            results.append(rec)
            continue

        Nclones = int((cnts != 0).sum())
        length_cdr3 = float((cnts * nc).sum() / Nreads)
        p = cnts / Nreads
        mask = p > 0
        if mask.any():
            shannon = float(-(p[mask] * np.log(p[mask])).sum())
            evenness = float(shannon / np.log(mask.sum()))
        else:
            shannon = 0.0
            evenness = np.nan
        gini = gini_coefficient(p)
        gini_simpson = float(1.0 - (p ** 2).sum())
        p_sorted_desc = np.sort(p)[::-1]
        p_sorted_asc = np.sort(p)
        top = float(p_sorted_desc[0]) if len(p_sorted_desc) > 0 else np.nan
        second_top = float(p_sorted_desc[1]) if len(p_sorted_desc) > 1 else np.nan
        rare = float(p_sorted_asc[0]) if len(p_sorted_asc) > 0 else np.nan
        second_rare = float(p_sorted_asc[1]) if len(p_sorted_asc) > 1 else np.nan

        rec = {
            "Sample": sample_name,
            "Nreads": Nreads,
            "Nclones": Nclones,
            "Length_CDR3": length_cdr3,
            "Shannon_Index": shannon,
            "Evenness": evenness,
            "Top_clone": top,
            "Second_top_clone": second_top,
            "Rare_clone": rare,
            "Second_Rare_clone": second_rare,
            "Gini": gini,
            "Gini_Simpson": gini_simpson,
        }
        results.append(rec)

    immune_index_results_df = pd.DataFrame(results)
    immune_index_results_df.to_csv(immune_indices_out_csv, index=False)
    return immune_index_results_df


def process_immune_data_batch_fast(
    path_to_result: str,
    immdata_out_csv: str,
    immune_indices_out_csv: str,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Accelerated, BIT-IDENTICAL drop-in for the original
    ``process_immune_data_batch`` (see module docstring for the exactness
    argument). Same signature, same two CSV artifacts, same returned frame.
    """
    if standardize_columns is None or gini_coefficient is None:
        if process_immune_data_batch is None:
            raise ImportError(
                "iobrpy.utils.process_immune_data_batch is unavailable"
            )
        return process_immune_data_batch(
            path_to_result, immdata_out_csv, immune_indices_out_csv
        )

    # Find all files whose name ends with "_report.tsv" (SAME glob call as
    # the original -> same file order -> same concat/immdata row order)
    pattern = os.path.join(path_to_result, "*_report.tsv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No *_report.tsv files found in {path_to_result!r}")

    def _read_one(file_path):
        df = pd.read_csv(file_path, sep="\t")
        df = standardize_columns(df)
        sample_name = os.path.basename(file_path)
        if sample_name.endswith("_report.tsv"):
            sample_name = sample_name[: -len("_report.tsv")]
        df["Sample"] = sample_name
        return df

    def _sample_of(fp):
        b = os.path.basename(fp)
        return b[: -len("_report.tsv")] if b.endswith("_report.tsv") else b

    all_samples: List[Optional[pd.DataFrame]] = [None] * len(files)
    n_workers = 1 if len(files) < 2 else min(
        len(files), max_workers or min(_READ_WORKERS_CAP, os.cpu_count() or 1)
    )
    with tqdm(
        total=len(files),
        desc="Immune data post-processing",
        unit="sample",
        dynamic_ncols=True,
    ) as pbar:
        if n_workers <= 1:
            for i, fp in enumerate(files):
                all_samples[i] = _read_one(fp)
                pbar.set_postfix_str(_sample_of(fp))
                pbar.update(1)
        else:
            # pd.read_csv's C tokenizer releases the GIL; frames are stored
            # at their INPUT index, so the concat order (and immdata bytes)
            # equals the sequential original loop regardless of completion
            # order.
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_read_one, fp): i for i, fp in enumerate(files)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    all_samples[i] = fut.result()
                    pbar.set_postfix_str(_sample_of(files[i]))
                    pbar.update(1)

    # Combine all samples into one DataFrame (clone-level data)
    immdata_combined = pd.concat(all_samples, ignore_index=True)

    # Save combined raw data as CSV (untouched original write path)
    immdata_combined.to_csv(immdata_out_csv, index=False)

    # Check required columns for downstream calculations
    required_cols = {"Sample", "read_count", "CDR3_dna"}
    missing = required_cols - set(immdata_combined.columns)
    if missing:
        raise ValueError(f"Missing required columns in combined data: {missing}")

    # ---- vectorized hoist of the per-group conversions -----------------
    # Whole-column conversions are row-wise identical to the per-group
    # originals; per-group views are sliced by position so every numpy
    # statement below sees the exact ordered float64 array the original
    # loop saw -> bitwise-identical statistics.
    try:
        cnt_all = immdata_combined["read_count"].to_numpy(dtype=float)
        nc_all = (
            immdata_combined["CDR3_dna"].astype(str).str.len().to_numpy(dtype=float)
        )
        codes, uniques = pd.factorize(immdata_combined["Sample"], sort=True)
    except (ValueError, TypeError):
        # pathological dtypes: run the ORIGINAL loop so its exception
        # order/messages are preserved verbatim
        return _indices_loop_original(immdata_combined, immune_indices_out_csv)

    # groupby("Sample") iteration order == sorted unique keys, original row
    # order within each group, NaN keys dropped. factorize(sort=True) +
    # stable argsort reproduces exactly that (code -1 == NaN sorts first and
    # is excluded by the searchsorted bounds below).
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    n_groups = len(uniques)
    starts = np.searchsorted(sorted_codes, np.arange(n_groups), side="left")
    ends = np.searchsorted(sorted_codes, np.arange(n_groups), side="right")

    results = []
    for g in range(n_groups):
        pos = order[starts[g]:ends[g]]
        cnts = cnt_all[pos]
        nc = nc_all[pos]

        Nreads = cnts.sum()

        # If total reads <= 0, fill with NaN to avoid division by zero
        if Nreads <= 0:
            Nclones = int((cnts != 0).sum())
            rec = {
                "Sample": uniques[g],
                "Nreads": Nreads,
                "Nclones": Nclones,
                "Length_CDR3": np.nan,
                "Shannon_Index": np.nan,
                "Evenness": np.nan,
                "Top_clone": np.nan,
                "Second_top_clone": np.nan,
                "Rare_clone": np.nan,
                "Second_Rare_clone": np.nan,
                "Gini": np.nan,
                "Gini_Simpson": np.nan,
            }
            results.append(rec)
            continue

        # Number of non-zero clones
        Nclones = int((cnts != 0).sum())

        # CDR3 length weighted by clone counts
        length_cdr3 = float((cnts * nc).sum() / Nreads)

        # Relative clone frequencies
        p = cnts / Nreads

        # Filter strictly positive probabilities for log
        mask = p > 0
        if mask.any():
            shannon = float(-(p[mask] * np.log(p[mask])).sum())
            evenness = float(shannon / np.log(mask.sum()))
        else:
            shannon = 0.0
            evenness = np.nan

        # Gini and Gini-Simpson
        gini = gini_coefficient(p)
        gini_simpson = float(1.0 - (p ** 2).sum())

        # Top and rare clone frequencies
        p_sorted_desc = np.sort(p)[::-1]
        p_sorted_asc = np.sort(p)

        top = float(p_sorted_desc[0]) if len(p_sorted_desc) > 0 else np.nan
        second_top = float(p_sorted_desc[1]) if len(p_sorted_desc) > 1 else np.nan
        rare = float(p_sorted_asc[0]) if len(p_sorted_asc) > 0 else np.nan
        second_rare = float(p_sorted_asc[1]) if len(p_sorted_asc) > 1 else np.nan

        rec = {
            "Sample": uniques[g],
            "Nreads": Nreads,
            "Nclones": Nclones,
            "Length_CDR3": length_cdr3,
            "Shannon_Index": shannon,
            "Evenness": evenness,
            "Top_clone": top,
            "Second_top_clone": second_top,
            "Rare_clone": rare,
            "Second_Rare_clone": second_rare,
            "Gini": gini,
            "Gini_Simpson": gini_simpson,
        }
        results.append(rec)

    # Combine per-sample results and save as CSV (untouched original write
    # path — reproduces the degenerate 1-byte '\n' file for empty results)
    immune_index_results_df = pd.DataFrame(results)
    immune_index_results_df.to_csv(immune_indices_out_csv, index=False)

    return immune_index_results_df


def _run_immune_postprocessing(output_root: Optional[str], engine: str = "fast") -> None:
    """Run the immune post-processing on all *_report.tsv under output_root
    (verbatim upstream staging; ``engine`` selects the accelerated or the
    ORIGINAL processor — both write byte-identical CSVs)."""
    if not output_root:
        return

    if engine == "fast" and standardize_columns is not None and gini_coefficient is not None:
        fn = process_immune_data_batch_fast
    else:
        # engine='original', or the upstream helpers are unavailable and the
        # ORIGINAL (equally unavailable -> skip message below) must decide
        fn = process_immune_data_batch
    if fn is None:
        # process_immune_data_batch is optional; skip if unavailable
        print(
            "[IOBRpy|trust4] process_immune_data_batch not available; skip immune post-processing.",
            file=sys.stderr,
        )
        return

    output_root = os.path.abspath(output_root)
    if not os.path.isdir(output_root):
        print(
            f"[IOBRpy|trust4] Output root for immune post-processing is not a directory: {output_root}",
            file=sys.stderr,
        )
        return

    report_files = _collect_report_files(output_root)
    if not report_files:
        print(
            f"[IOBRpy|trust4] No *_report.tsv files found under {output_root!r}; skip immune post-processing.",
            file=sys.stderr,
        )
        return

    # Create a temporary directory containing symlinks/copies to all report TSVs.
    # File names in this staging directory are <sample>_report.tsv so that
    # downstream tools use the clean sample names in their outputs (both
    # trust4_immdata.csv and trust4_immune_indices.csv).
    staging_dir = tempfile.mkdtemp(prefix="iobrpy_trust4_reports_")
    name_counts: Dict[str, int] = {}
    for src in report_files:
        sample = _derive_sample_name_from_path(src, output_root)
        base_name = f"{sample}_report.tsv"

        # Avoid name collisions by appending a numeric suffix if needed
        count = name_counts.get(base_name, 0)
        name_counts[base_name] = count + 1
        if count == 0:
            dst_name = base_name
        else:
            dst_name = f"{sample}_{count}_report.tsv"

        dst = os.path.join(staging_dir, dst_name)

        try:
            os.symlink(src, dst)
        except Exception:
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(
                    f"[IOBRpy|trust4] WARNING: could not stage {src} -> {dst}: {e}",
                    file=sys.stderr,
                )

    immdata_out_csv = os.path.join(output_root, "trust4_immdata.csv")
    indices_out_csv = os.path.join(output_root, "trust4_immune_indices.csv")

    try:
        print(
            f"[IOBRpy|trust4] Running immune post-processing on TRUST4 reports under: {output_root}",
            flush=True,
        )
        fn(staging_dir, immdata_out_csv, indices_out_csv)
        print(
            f"[IOBRpy|trust4] Immune clone-level data saved to: {immdata_out_csv}",
            flush=True,
        )
        print(
            f"[IOBRpy|trust4] Immune diversity indices saved to: {indices_out_csv}",
            flush=True,
        )
    except FileNotFoundError as e:
        print(
            f"[IOBRpy|trust4] No *_report.tsv files found for immune post-processing: {e}",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[IOBRpy|trust4] Failed to run immune post-processing: {e}",
            file=sys.stderr,
        )
    finally:
        # Clean up staging directory
        try:
            shutil.rmtree(staging_dir)
        except Exception:
            pass


def main(argv: Optional[List[str]] = None, trust4_bin: Optional[str] = None,
         engine: str = "fast"):
    """TRUST4 orchestration entry point (verbatim upstream ``main``).

    ``trust4_bin`` overrides the runner resolution (default: the upstream
    ``TRUST4_BIN`` env var / ``"run-trust4"`` on PATH); ``engine`` selects
    the immune post-processor (``'fast'`` accelerated bit-identical, or
    ``'original'``). Ends in ``sys.exit(<rc>)`` on every path, exactly like
    the upstream CLI (the iobrx public wrapper catches it and returns the
    code).
    """
    runner = trust4_bin if trust4_bin is not None else os.environ.get("TRUST4_BIN", "run-trust4")
    if shutil.which(runner) is None:
        print(
            f"[IOBRpy|trust4] Cannot find TRUST4 executable '{runner}'. "
            f"Install TRUST4 (e.g. conda install -c bioconda trust4) or set TRUST4_BIN.",
            file=sys.stderr,
        )
        sys.exit(127)

    p = argparse.ArgumentParser(
        prog="iobrpy trust4",
        description="Run TRUST4 (TCR/BCR reconstruction) with IOBRpy defaults for -f/--ref if unspecified.",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs
    p.add_argument(
        "-b",
        dest="bam",
        help="Path to BAM file OR a directory of *.bam (batch mode)",
    )
    p.add_argument("-1", dest="r1", help="Paired-end read 1 FASTQ")
    p.add_argument("-2", dest="r2", help="Paired-end read 2 FASTQ")
    p.add_argument("-u", dest="ru", help="Single-end read FASTQ")

    # Batch (FASTQ) mode
    p.add_argument(
        "--fqdir",
        dest="fqdir",
        help="Directory containing paired FASTQs (*.fastq.gz) with suffixes _1/_2",
    )

    # Reference files
    p.add_argument(
        "-f",
        dest="f",
        help="FASTA with coordinates/sequences of V/D/J/C genes",
    )
    p.add_argument(
        "--ref",
        dest="ref",
        help="Detailed V/D/J/C reference FASTA (e.g., IMGT)",
    )

    # General
    p.add_argument(
        "-o",
        dest="o",
        help="Single-run: output prefix; Batch: OUTPUT ROOT directory",
    )
    p.add_argument(
        "--od",
        dest="od",
        help="Output directory (TRUST4 native; ignored in batch modes)",
    )
    p.add_argument("-t", dest="t", type=int, help="Threads")
    p.add_argument("-k", dest="k", type=int, help="Starting k-mer size")

    # Single-cell / barcodes / UMI
    p.add_argument("--barcode", dest="barcode", help="Barcode field/file")
    p.add_argument(
        "--barcodeLevel",
        dest="barcodeLevel",
        choices=["cell", "molecule"],
        help="Barcode level",
    )
    p.add_argument(
        "--barcodeWhitelist",
        dest="barcodeWhitelist",
        help="Barcode whitelist",
    )
    p.add_argument(
        "--barcodeTranslate",
        dest="barcodeTranslate",
        help="Barcode translate file",
    )
    p.add_argument("--UMI", dest="UMI", help="UMI field/file")
    p.add_argument(
        "--readFormat",
        dest="readFormat",
        help="Format spec for reads/barcodes/UMI",
    )

    # Modes & fine controls
    p.add_argument(
        "--repseq",
        action="store_true",
        help="Bulk non-UMI-based TCR/BCR-seq",
    )
    p.add_argument(
        "--contigMinCov",
        type=int,
        help="Min bases covered by reads for contigs",
    )
    p.add_argument(
        "--minHitLen",
        type=int,
        help="Minimal hit length for valid overlap",
    )
    p.add_argument(
        "--mateIdSuffixLen",
        type=int,
        help="Mate suffix length in read id",
    )
    p.add_argument(
        "--skipMateExtension",
        action="store_true",
        help="Do not extend assemblies with mate info",
    )
    p.add_argument(
        "--abnormalUnmapFlag",
        action="store_true",
        help="Unmapped read-pair flag is nonconcordant",
    )
    p.add_argument(
        "--assembleWithRef",
        action="store_true",
        help="Assemble with --ref file",
    )
    p.add_argument(
        "--noExtraction",
        action="store_true",
        help="Directly use provided FASTQ files",
    )
    p.add_argument(
        "--outputReadAssignment",
        action="store_true",
        help="Output read assignment to *_assign.out",
    )

    # Pipeline stage / cleanup
    p.add_argument(
        "--stage",
        type=int,
        choices=[0, 1, 2, 3],
        help="0:begin; 1:assembly; 2:annotation; 3:report",
    )
    p.add_argument(
        "--clean",
        type=int,
        choices=[0, 1, 2],
        help="0:keep all; 1:clean intermediates; 2:only keep AIRR files",
    )

    args, unknown = p.parse_known_args(argv)

    # Determine mode
    batch_fq = args.fqdir is not None
    batch_bamdir = False
    single_bam = False

    if args.bam is not None:
        if os.path.isdir(args.bam):
            batch_bamdir = True
        elif os.path.isfile(args.bam):
            single_bam = True
        else:
            p.error(f"-b path not found: {args.bam}")

    has_pe = args.r1 is not None and args.r2 is not None
    has_se = args.ru is not None

    # Conflicts
    if batch_fq and (single_bam or batch_bamdir or has_pe or has_se):
        p.error("Conflicting inputs: --fqdir cannot be combined with -b/-1/-2/-u.")
    if batch_bamdir and (has_pe or has_se):
        p.error(
            "Conflicting inputs: -b <dir> (batch mode) cannot be combined with -1/-2/-u."
        )

    # Validate at least one input mode
    if not (batch_fq or batch_bamdir or single_bam or has_pe or has_se):
        p.error(
            "Provide one of: -b <BAM> | -b <DIR_OF_BAM> | -1 R1 -2 R2 | -u READS | --fqdir DIR"
        )

    # Resolve -f and --ref defaults if not provided
    f_path = args.f
    ref_path = args.ref
    if f_path is None:
        f_path = _extract_resource_to_tmp(DEFAULT_F)
        if f_path is None:
            p.error(
                f"Default -f resource '{DEFAULT_F}' not found in {RES_PKG}. "
                f"Place the file under iobrpy/resources/ or pass -f explicitly."
            )
    if ref_path is None:
        ref_path = _extract_resource_to_tmp(DEFAULT_REF)
        if ref_path is None:
            print(
                f"[IOBRpy|trust4] Warning: default --ref '{DEFAULT_REF}' not found in {RES_PKG}. "
                f"Continuing without --ref (less recommended).",
                file=sys.stderr,
            )

    # Common options (shared for all runs) -- DO NOT include --od here
    common_opts = _build_common_options(args, f_path, ref_path)
    common_opts.extend(unknown)  # pass-through any extra options

    # -------------------------
    # Batch over directory of BAMs via -b <dir>
    # -------------------------
    if batch_bamdir:
        if not args.o:
            p.error("When using -b <DIR>, please provide -o as an OUTPUT ROOT directory.")
        bams = _list_bams(args.bam)
        if not bams:
            p.error(f"No *.bam found under directory: {args.bam}")

        print(f"[IOBRpy|trust4] Found {len(bams)} BAM samples in: {args.bam}")
        overall_rc = 0
        with tqdm(
            total=len(bams),
            desc="TRUST4 (BAM batch)",
            unit="sample",
            dynamic_ncols=True,
        ) as pbar:
            for bam in bams:
                sample = os.path.splitext(os.path.basename(bam))[0]
                # Folder name: drop trailing _Aligned.sortedByCoord.out (or .Aligned...); case-insensitive
                folder = re.sub(
                    r"([._]Aligned)\.sortedByCoord\.out$",
                    "",
                    sample,
                    flags=re.IGNORECASE,
                )
                sample_dir = os.path.join(args.o, folder)
                os.makedirs(sample_dir, exist_ok=True)

                # Per-sample done flag: if this file exists, skip re-running TRUST4
                done_flag = os.path.join(sample_dir, f"{folder}.TRUST4.done")
                if os.path.exists(done_flag):
                    print(
                        f"[IOBRpy|trust4] Skip sample {folder}: found done flag {done_flag}",
                        flush=True,
                    )
                    # Still advance the progress bar so that the batch summary is correct
                    pbar.update(1)
                    continue

                # File prefix: keep _Aligned, only drop .sortedByCoord.out
                prefix_base = re.sub(
                    r"\.sortedByCoord\.out$",
                    "",
                    sample,
                    flags=re.IGNORECASE,
                )
                prefix = f"TRUST_{prefix_base}"

                cmd: List[str] = [runner]
                _append_opt(cmd, "-b", bam)
                cmd.extend(common_opts)
                _append_opt(cmd, "--od", sample_dir)
                _append_opt(cmd, "-o", prefix)

                print(
                    f"[IOBRpy|trust4] Running ({folder}):",
                    " ".join(map(str, cmd)),
                    flush=True,
                )
                try:
                    rc = subprocess.run(cmd, check=False).returncode
                except KeyboardInterrupt:
                    sys.exit(130)

                # If TRUST4 finishes successfully for this sample, create the done flag
                if rc == 0:
                    try:
                        with open(done_flag, "w") as f:
                            f.write(
                                f"SUCCESS: TRUST4 finished for sample {folder}\n"
                            )
                    except Exception as e:
                        # Do not fail the whole batch just because we could not write the flag
                        print(
                            f"[IOBRpy|trust4] WARNING: could not write done flag for {folder}: {e}",
                            file=sys.stderr,
                        )
                else:
                    overall_rc = rc

                if rc != 0:
                    overall_rc = rc
                pbar.update(1)

        # After all TRUST4 BAM batch runs, run immune post-processing on -o root
        _run_immune_postprocessing(args.o, engine=engine)
        sys.exit(overall_rc)

    # -------------------------
    # Batch over FASTQ directory
    # -------------------------
    if batch_fq:
        if not args.o:
            p.error("When using --fqdir, please provide -o as an OUTPUT ROOT directory.")
        try:
            pairs = _pair_fastqs(args.fqdir)
        except (FileNotFoundError, ValueError) as e:
            p.error(str(e))
        if not pairs:
            p.error(
                f"No paired FASTQ (*.fastq.gz with _1/_2) found in --fqdir: {args.fqdir}"
            )

        samples = sorted(pairs.keys())
        print(f"[IOBRpy|trust4] Found {len(samples)} FASTQ samples in: {args.fqdir}")
        overall_rc = 0
        with tqdm(
            total=len(samples),
            desc="TRUST4 (FASTQ batch)",
            unit="sample",
            dynamic_ncols=True,
        ) as pbar:
            for sample in samples:
                r1p, r2p = pairs[sample]
                # Folder name: drop trailing _Aligned.sortedByCoord.out if present
                folder = re.sub(
                    r"([._]Aligned)\.sortedByCoord\.out$",
                    "",
                    sample,
                    flags=re.IGNORECASE,
                )
                sample_dir = os.path.join(args.o, folder)
                os.makedirs(sample_dir, exist_ok=True)

                # Per-sample done flag: if this file exists, skip re-running TRUST4
                done_flag = os.path.join(sample_dir, f"{folder}.TRUST4.done")
                if os.path.exists(done_flag):
                    print(
                        f"[IOBRpy|trust4] Skip sample {folder}: found done flag {done_flag}",
                        flush=True,
                    )
                    pbar.update(1)
                    continue

                # File prefix: keep _Aligned, only drop .sortedByCoord.out
                prefix_base = re.sub(
                    r"\.sortedByCoord\.out$",
                    "",
                    sample,
                    flags=re.IGNORECASE,
                )
                prefix = f"TRUST_{prefix_base}"

                cmd = [runner]
                _append_opt(cmd, "-1", r1p)
                _append_opt(cmd, "-2", r2p)
                cmd.extend(common_opts)
                _append_opt(cmd, "--od", sample_dir)
                _append_opt(cmd, "-o", prefix)

                print(
                    f"[IOBRpy|trust4] Running ({folder}):",
                    " ".join(map(str, cmd)),
                    flush=True,
                )
                try:
                    rc = subprocess.run(cmd, check=False).returncode
                except KeyboardInterrupt:
                    sys.exit(130)

                # If TRUST4 finishes successfully for this sample, create the done flag
                if rc == 0:
                    try:
                        with open(done_flag, "w") as f:
                            f.write(
                                f"SUCCESS: TRUST4 finished for sample {folder}\n"
                            )
                    except Exception as e:
                        print(
                            f"[IOBRpy|trust4] WARNING: could not write done flag for {folder}: {e}",
                            file=sys.stderr,
                        )
                else:
                    overall_rc = rc

                pbar.update(1)

        # After all TRUST4 FASTQ batch runs, run immune post-processing on -o root
        _run_immune_postprocessing(args.o, engine=engine)
        sys.exit(overall_rc)

    # -------------------------
    # Single-run mode (BAM or FASTQ directly)
    # -------------------------
    cmd: List[str] = [runner]
    if single_bam:
        _append_opt(cmd, "-b", args.bam)
    if has_pe:
        _append_opt(cmd, "-1", args.r1)
        _append_opt(cmd, "-2", args.r2)
    if has_se:
        _append_opt(cmd, "-u", args.ru)

    cmd.extend(common_opts)
    _append_opt(cmd, "-o", args.o)
    _append_opt(cmd, "--od", args.od)  # single-run only; optional

    print("[IOBRpy|trust4] Running:", " ".join(map(str, cmd)), flush=True)
    try:
        proc = subprocess.run(cmd, check=False)
        rc = proc.returncode

        out_root = _infer_single_output_root(args)
        _run_immune_postprocessing(out_root, engine=engine)

        sys.exit(rc)
    except KeyboardInterrupt:
        sys.exit(130)


def trust4_original(argv: Optional[List[str]] = None):
    """Run the untouched upstream ``iobrpy.workflow.trust4.main(argv)``.
    Escape hatch for ``backend='python'``; propagates the upstream
    ``SystemExit`` (the iobrx public wrapper catches it and returns the
    code)."""
    from iobrpy.workflow.trust4 import main as _orig_main

    return _orig_main(argv)


if __name__ == "__main__":
    # Support `python -m iobrx._fast.trust4_fast ...` mirroring the upstream
    # `python -m iobrpy.workflow.trust4 ...` entry.
    main()
