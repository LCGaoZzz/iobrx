"""batch_salmon_fast: accelerated drop-in for the ``iobrpy.workflow.batch_salmon``
CLI stage (``python -m iobrpy.main batch_salmon --index ... --path_fq ... --path_out ...``).

Same behaviour as upstream (batch_salmon.py, IOBRpy 0.2.0): resolve the R2
suffix from ``--suffix1`` with the upstream inference ladder (``_1``->``_2``,
``R1``->``R2``, regex fallbacks, else the identical ``ValueError``), discover
R1s with ``sorted(glob(path_fq/*<suffix1>))``, warn-and-skip samples whose
R2 is missing, ``random.shuffle`` the pairs, then run one ``salmon quant``
child per sample through ``multiprocessing.Pool(processes=max(1,
batch_size))`` + ``imap_unordered`` (the EXACT upstream scheduling
primitives). Per sample: resume via non-empty ``quant.sf`` + ``task.complete``
marker, the upstream command line token-for-token

    salmon quant -i <index> -l ISF --gcBias -1 <r1> -2 <r2> -p <threads>
                 -o <out_dir> --validateMappings [-g <gtf>]

with the child's stdout/stderr sent to DEVNULL (salmon keeps its own logs
under ``<out_dir>/logs``), the friendly version-mismatch guidance on
``CalledProcessError``, and the ``task.complete`` ``"ok\\n"`` marker. The
optional ``IOBRPY_SALMON_VERBOSE`` preflight (salmon ``--version`` parse +
index ``versionInfo.json`` / ``meta_info.json`` / ``info.json`` best-effort
read) is reproduced verbatim.

The salmon binary is resolved exactly as upstream resolves it — the bare
name ``"salmon"`` through the child process PATH; ``salmon_bin`` only
replaces that argv token (default ``None`` = upstream behaviour).

Exit-code contract: the upstream CLI ``main()`` ends in ``sys.exit(2)``
(bad suffix / no FASTQs / no pairs), ``sys.exit(1)`` (some samples failed,
details on stderr) or implicit 0. This port RETURNS that code inside the
result dict (``{"rc": 0|1|2, ...}``) instead of raising ``SystemExit`` —
same messages on the same streams, caller-friendly termination.

Known nondeterminism (identical in original and port): ``random.shuffle``
dispatch order + ``imap_unordered`` completion order -> ``[Start]/[Done]``
console line order varies run-to-run; on-disk per-sample outputs are
order-invariant. ``quant.sf`` bytes depend only on (salmon version, index,
reads, parameters INCLUDING ``-p`` thread count) — matched by construction
when the same salmon binary and arguments are used.
"""
from __future__ import annotations
from iobrx._run_state import signature, completed, complete

import glob
import json
import os
import random
import re
import subprocess
import sys
from multiprocessing import Pool

__all__ = ["batch_salmon", "batch_salmon_original", "process_sample"]


def _print_iobrpy_banner():
    """Print the IOBRpy banner (the upstream ``iobrpy.main`` dispatch prints
    it after a SUCCESSFUL batch_salmon run; ``sys.exit(1|2)`` skips it)."""
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
        # Fallback without colors
        print("#########################################################")
        print(" IOBRpy: Immuno-Oncology Biological Research using Python ")
        print(" If you encounter any issues, please report them at ")
        print(" https://github.com/IOBR/IOBRpy/issues ")
        print("#########################################################")
        print(" Author: Haonan Huang, Dongqiang Zeng")
        print(" Email: interlaken@smu.edu.cn ")
        print("#########################################################")
    print("   ")


# ---------------------------------------------------------------------------
# upstream helpers (verbatim)
# ---------------------------------------------------------------------------
def _salmon_version_tuple(salmon_bin: str = "salmon"):
    """Return Salmon version as (major, minor, patch) or None if unavailable."""
    try:
        out = subprocess.check_output([salmon_bin, "--version"], text=True).strip()
        # expected: "salmon 1.10.3"
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
        if m:
            return tuple(map(int, m.groups()))
    except Exception:
        pass
    return None


def _read_index_meta(index_dir: str):
    """Best-effort read of index meta json; return dict or None (verbatim)."""
    for name in ("versionInfo.json", "meta_info.json", "info.json"):
        p = os.path.join(index_dir, name)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"__raw__": "unreadable"}
    return None


def _infer_suffix2_from_suffix1(s1: str) -> str:
    """Infer read2 suffix from read1 suffix (verbatim upstream ladder)."""
    # Prefer specific tokens first
    if "_1" in s1:
        return s1.replace("_1", "_2")
    if "R1" in s1:
        return s1.replace("R1", "R2")

    # Fallback near file stem: ...1.fastq(.gz)? / 1.fq(.gz)?
    s2 = re.sub(r"1(\.f(?:ast)?q(?:\.gz)?)$", r"2\1", s1)
    if s2 != s1:
        return s2

    # Last resort: try delimiter-number pattern
    s2 = re.sub(r"([._-])1(\.f(?:ast)?q(?:\.gz)?)$", r"\g<1>2\2", s1)
    if s2 != s1:
        return s2

    raise ValueError(
        f"Unable to infer suffix2 from suffix1='{s1}'. Please rename FASTQs or add a "
        "clear pattern like '_1'/'_2' or 'R1'/'R2'."
    )


def _pair_from_r1(path_r1: str, suffix1: str, suffix2: str):
    """Given an R1 path and suffixes, return (sample_id, r1, r2) (verbatim)."""
    if not path_r1.endswith(suffix1):
        raise ValueError(
            f"File does not end with suffix1: {path_r1} (suffix1={suffix1})"
        )
    base = os.path.basename(path_r1)[: -len(suffix1)]
    dirn = os.path.dirname(path_r1)
    r2 = os.path.join(dirn, base + suffix2)
    return base, path_r1, r2


def _exists_nonempty(p: str) -> bool:
    return os.path.exists(p) and os.path.getsize(p) > 0


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _run_salmon_one(
    sample_id: str,
    r1: str,
    r2: str,
    index: str,
    out_root: str,
    threads: int,
    gtf: str = None,
    salmon_bin: str = "salmon",
    verbose: bool = True,
):
    """One ``salmon quant`` run (verbatim upstream logic and command line)."""
    out_dir = os.path.join(out_root, sample_id)
    _ensure_dir(out_dir)

    # Resume logic
    quant_sf = os.path.join(out_dir, "quant.sf")
    done_flag = os.path.join(out_dir, "task.complete")
    run_signature = signature([r1, r2, index] + ([gtf] if gtf else []),
                              {"threads": threads, "libtype": "ISF", "gcBias": True,
                               "validateMappings": True}, [salmon_bin])
    if completed(done_flag, run_signature, [quant_sf]):
        if verbose:
            print(f"[Skip] {sample_id} already finished; skipping.")
        return sample_id, True, None

    if verbose:
        print(f"[Start] {sample_id} is running...", flush=True)

    cmd = [
        salmon_bin,
        "quant",
        "-i",
        index,
        "-l",
        "ISF",
        "--gcBias",
        "-1",
        r1,
        "-2",
        r2,
        "-p",
        str(threads),
        "-o",
        out_dir,
        "--validateMappings",
    ]
    if gtf:
        cmd += ["-g", gtf]

    try:
        # suppress salmon stdout/stderr so they don't clutter the terminal
        # (salmon still writes its own logs into <out_dir>/logs)
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # Friendly guidance for common cause
        msg = (
            "[error] salmon quant failed for sample {sid} (exit {code}).\n"
            "        Command: {cmd}\n"
            "        Common cause: index built by newer Salmon than your runtime,\n"
            "        leading to rapidjson assertion failures or 'invoked improperly'.\n"
            "        Fix: upgrade Salmon to the version used to build the index "
            "(or newer), or rebuild the index using your current Salmon version."
        ).format(sid=sample_id, code=e.returncode, cmd=" ".join(cmd))
        return sample_id, False, msg

    try:
        complete(done_flag, run_signature, [quant_sf], "ok\n")
    except RuntimeError as exc:
        return sample_id, False, str(exc)

    if verbose:
        print(f"[Done] {sample_id} finished successfully. Saved to: {quant_sf}", flush=True)
    return sample_id, True, None


def process_sample(args_tuple):
    """Pool wrapper that returns (sample_id, success, error_message_or_None)."""
    return _run_salmon_one(*args_tuple)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def batch_salmon(index, path_fq, path_out, suffix1="_1.fastq.gz",
                 batch_size=1, num_threads=8, gtf=None, salmon_bin=None,
                 verbose=True):
    """Batch-run Salmon quantification over paired-end FASTQs (upstream
    ``main()`` scheduling, verbatim; returns the CLI exit code instead of
    ``sys.exit``).

    Parameters
    ----------
    index : str
        Salmon index directory (upstream ``--index``).
    path_fq : str
        Directory containing the FASTQ files (upstream ``--path_fq``;
        non-recursive ``glob('*' + suffix1)`` discovery, as upstream).
    path_out : str
        Output root; one ``<sample_id>/`` subdirectory per sample (upstream
        ``--path_out``).
    suffix1 : str, default '_1.fastq.gz'
        R1 suffix; R2 inferred with the upstream ladder (``--suffix1``).
    batch_size : int, default 1
        ``multiprocessing.Pool`` size = concurrent samples (``--batch_size``;
        ``max(1, int(...))`` as upstream).
    num_threads : int, default 8
        Threads per salmon process (``--num_threads`` -> ``-p``).
    gtf : str, optional
        Optional GTF for gene-level quant (``--gtf`` -> ``-g``).
    salmon_bin : str, optional
        Binary path/name override; ``None`` = bare ``"salmon"`` resolved
        through PATH (upstream behaviour).
    verbose : bool, default True
        Reproduce the upstream console protocol; stderr diagnostics and the
        failure summary print regardless.

    Returns
    -------
    dict
        ``{"rc": int, "failures": [(sample_id, message), ...],
        "n_pairs": int, "results": [(sample_id, ok, err), ...]}`` where
        ``rc`` is the exit code the upstream CLI would produce (0 success,
        1 sample failures, 2 usage/discovery error). Upstream returns None.
    """
    salmon_cmd = salmon_bin if salmon_bin is not None else "salmon"

    # Preflight info (verbatim, env-driven)
    if os.environ.get("IOBRPY_SALMON_VERBOSE"):
        sv = _salmon_version_tuple(salmon_cmd)
        if sv:
            print(f"[preflight] salmon version: {'.'.join(map(str, sv))}")
        else:
            print(
                "[preflight] salmon version: <unknown> (salmon not found on PATH?)"
            )

        meta = _read_index_meta(index)
        if isinstance(meta, dict):
            # Print a short summary of meta keys; don't over-parse
            keys = list(meta.keys())[:6]
            print(
                f"[preflight] index meta keys: {keys if keys else '<none>'}"
            )
        else:
            print("[preflight] index meta: <not found>")

    # Resolve suffix2 from suffix1
    try:
        suffix2 = _infer_suffix2_from_suffix1(suffix1)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return {"rc": 2, "failures": [], "n_pairs": 0, "results": []}

    # Discover R1s and pair
    pattern = os.path.join(path_fq, f"*{suffix1}")
    r1_files = sorted(glob.glob(pattern))
    if not r1_files:
        print(f"[error] No FASTQ files matched: {pattern}", file=sys.stderr)
        return {"rc": 2, "failures": [], "n_pairs": 0, "results": []}

    pairs = []
    for r1 in r1_files:
        sample_id, r1p, r2p = _pair_from_r1(r1, suffix1, suffix2)
        if not os.path.exists(r2p):
            print(
                f"[warn] Missing R2 for {sample_id}: {r2p}. Skipping this sample.",
                file=sys.stderr,
            )
            continue
        pairs.append(
            (
                sample_id,
                r1p,
                r2p,
                index,
                path_out,
                num_threads,
                gtf,
                salmon_cmd,
                verbose,
            )
        )

    random.shuffle(pairs)

    if not pairs:
        print("[error] No valid R1/R2 pairs to process.", file=sys.stderr)
        return {"rc": 2, "failures": [], "n_pairs": 0, "results": []}

    # Ensure output root exists
    _ensure_dir(path_out)

    # Run
    if verbose:
        print(
            f"[Plan] Processing {len(pairs)} samples (Order: RANDOM SHUFFLED).\n "
            f"      Batch size: {batch_size}\n       Threads per job: {num_threads}"
        )
    failures = []
    results = []
    with Pool(processes=max(1, int(batch_size))) as pool:
        for sid, ok, err in pool.imap_unordered(process_sample, pairs):
            results.append((sid, ok, err))
            if not ok:
                failures.append((sid, err))

    if failures:
        print("\n[summary] Some samples failed:", file=sys.stderr)
        for sid, err in failures:
            print(f"--- {sid} ---", file=sys.stderr)
            print(err, file=sys.stderr)
        rc = 1
    else:
        if verbose:
            print("\n[summary] All samples finished successfully.")
        rc = 0
    if rc == 0 and verbose:
        # upstream main.py dispatch prints the banner only after a
        # successful (non-SystemExit) stage run
        _print_iobrpy_banner()
    return {"rc": rc, "failures": failures, "n_pairs": len(pairs),
            "results": results}


def batch_salmon_original(index, path_fq, path_out, suffix1="_1.fastq.gz",
                          batch_size=1, num_threads=8, gtf=None):
    """Run the untouched upstream CLI ``main()`` (argv rewrite, exactly how
    ``iobrpy.main`` dispatches this subcommand). Escape hatch for
    ``backend='python'``; returns ``{"rc": ...}`` from the upstream
    ``sys.exit`` codes (upstream itself returns nothing)."""
    from iobrpy.workflow import batch_salmon as _orig

    argv = ["batch_salmon",
            "--index", str(index),
            "--path_fq", str(path_fq),
            "--path_out", str(path_out),
            "--suffix1", str(suffix1),
            "--batch_size", str(int(batch_size)),
            "--num_threads", str(int(num_threads))]
    if gtf:
        argv += ["--gtf", str(gtf)]
    saved = sys.argv
    try:
        sys.argv = argv
        try:
            _orig.main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            return {"rc": code, "failures": None, "n_pairs": None,
                    "results": None}
        return {"rc": 0, "failures": None, "n_pairs": None, "results": None}
    finally:
        sys.argv = saved
