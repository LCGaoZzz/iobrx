"""hla_typing_fast: verbatim port of the ``iobrpy.workflow.hla_typing``
batch HLA-typing workflow (CLI ``python -m iobrpy.main hla_typing -b
BAM_DIR -r {hg19,hg38} -o OUTDIR [-j THREADS] [-u {0,1}]``, IOBRpy 0.2.0
gold standard), including the ``iobrpy.SpecHLA.extract_hla_read``
dependency/extraction helpers it wires in.

Contract with the original
--------------------------
Sample discovery (``infer_sample_id`` / ``collect_samples`` with the
``_Aligned.sortedByCoord.out.bam`` suffix rule and sorted glob), the
SEQUENTIAL ExtractHLAread phase with ``<id>.ExtractHLAread.done`` resume
markers, the SpecHLA phase (environment ensured ONCE, then per-sample
``<id>.SpecHLA.done`` markers gated on ``hla_result_has_sample``), the
FASTQ-pair preference in ``find_fastqs_for_sample``, the merged-table
writer ``merge_hla_results`` (version line + single header + data lines,
duplicate-header suppression) and every protocol line / warning / exit
path are BYTE-VERBATIM copies of the upstream modules. External binaries
are scheduled unchanged: ``bash <root>/script/ExtractHLAread.sh -s -b -r
-o`` and ``bash <root>/script/whole/SpecHLA_RNAseq.sh -n -1 -2 -o -j -u``.

The ONLY port-side differences:

1. SpecHLA asset-root resolution — upstream ``run_extraction`` derives the
   script directory from ``__file__`` (it lives inside iobrpy/SpecHLA) and
   upstream ``run_spechla_phase`` calls ``detect_spec_hla_root()`` (also
   ``__file__``-based). The port threads the resolved root through
   (``spec_hla_root=`` argument > ``$SPECHLA_ROOT`` > the installed
   ``iobrpy`` package tree — see ``spechla_fast._resolve_spec_hla_root``);
   the printing ``detect_spec_hla_root(...)`` call stays at the SAME
   position in ``run_spechla_phase``, so the stdout protocol order is
   unchanged (``main`` resolves silently for the extract phase).
2. ``ensure_extract_deps`` (upstream's import alias of
   ``extract_hla_read.ensure_dependencies``) is the locally defined
   verbatim copy ``ensure_dependencies``.
3. ``main(argv=None, spec_hla_root=None)`` threads the override; flow and
   banner are upstream.
4. tqdm is imported with a graceful fallback shim (upstream hard-imports
   it); with tqdm installed — the parity-relevant case — behaviour is
   identical.
5. ``hla_typing_original(argv)`` — escape hatch running the untouched
   upstream ``iobrpy.workflow.hla_typing.main`` (used by
   ``iobrx.hla_typing(backend='python')``).

No periphery is re-implemented "faster": discovery/merge logic is
millisecond-scale text I/O next to hours of external-tool phasing
(orchestration ceiling speedup ~ 1), so verbatim fidelity wins.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from iobrx._fast.spechla_fast import (
    _resolve_spec_hla_root,
    detect_spec_hla_root,
    ensure_python_deps,
    ensure_external_tools,
    ensure_spechap_built,
    ensure_bowtie2_index,
    detect_bowtie2_build,
    run_spechla_rnaseq,
    print_colorful_message,
)

# tqdm progress bar (with graceful fallback if not installed) — upstream
# hard-imports tqdm; the fallback only keeps the port importable without it.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class _SimpleTqdm:
        """Minimal iterable fallback when tqdm is unavailable."""

        def __init__(self, iterable=None, total=None, desc=None, unit=None,
                     **kwargs):
            self.iterable = iterable
            self.total = total if total is not None else (
                len(iterable)
                if iterable is not None and hasattr(iterable, "__len__")
                else 0)
            self.desc = desc or "progress"
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)

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

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return _SimpleTqdm(iterable, *args, **kwargs)

    def _tqdm_write(msg, *args, **kwargs):
        print(msg)

    tqdm.write = staticmethod(_tqdm_write)  # type: ignore[attr-defined]


__all__ = ["main", "hla_typing_original", "infer_sample_id",
           "collect_samples", "run_extract_phase", "run_spechla_phase",
           "merge_hla_results", "hla_result_has_sample",
           "find_fastqs_for_sample", "ensure_dependencies",
           "run_extraction", "build_arg_parser"]

# ---------------------------------------------------------------------------
# ExtractHLAread helpers — verbatim from iobrpy/SpecHLA/extract_hla_read.py
# ---------------------------------------------------------------------------
# Mapping from required binary name -> conda package name.
# ExtractHLAread.sh uses "samtools" and "bam" (from bamUtil). :contentReference[oaicite:0]{index=0}
REQUIRED_TOOLS: Dict[str, str] = {
    "samtools": "samtools=1.21",
    "bam": "bamutil",
}

# Conda-level libraries that must exist with an exact version.
# We will check them via `conda list` before running ExtractHLAread.
REQUIRED_CONDA_PACKAGES: Dict[str, str] = {
    "libdeflate": "1.25",
    "htslib": "1.21",
}

LIBCURL_SOLVER_SPEC = "libcurl>=8.11.1,<9.0a0"

def _running_in_conda() -> bool:
    """
    Return True if the current Python process looks like it is running
    inside a conda environment.
    """
    return bool(os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"))


def _detect_conda_executable() -> str | None:
    """
    Try to find a conda-like executable in PATH (mamba, conda, micromamba).
    Returns the path to the first one that is found, or None if none exist.
    """
    for exe in ("conda", "mamba", "micromamba"):
        path = shutil.which(exe)
        if path is not None:
            return path
    return None

def _get_conda_package_version(conda_exe: str, pkg_name: str) -> str | None:
    """
    Query `conda list` to get the installed version of a package.

    This uses `conda list <pkg> --json` to robustly parse the version.
    Returns the version string if the package is present in the environment,
    otherwise returns None.
    """
    try:
        # Use JSON output to make parsing robust.
        out = subprocess.check_output(
            [conda_exe, "list", pkg_name, "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
    except Exception:
        return None

    for rec in data:
        if rec.get("name") == pkg_name and "version" in rec:
            return rec["version"]

    return None

def ensure_dependencies(auto_install: bool = True) -> None:
    """
    Check whether low-level libraries and command-line tools are available.

    Order is important:

    1. First, use `conda list` to make sure libdeflate and htslib are present
       with the exact required versions (libdeflate=1.25, htslib=1.21).
       If they are missing or have the wrong version and auto_install is True,
       try to fix them via conda/mamba.
    2. Then, check that `samtools` and `bam` (bamUtil) are on PATH; install
       them via conda/mamba if needed.

    Raises RuntimeError if dependencies cannot be satisfied.
    """
    # Determine which binaries are currently missing. This is used in step 2.
    missing_binaries: List[str] = [
        bin_name for bin_name in REQUIRED_TOOLS if shutil.which(bin_name) is None
    ]

    # We will reuse the same conda-like executable for both steps.
    conda_exe = _detect_conda_executable()

    # ---------- Step 1: enforce libdeflate / htslib versions ----------
    # Only do this if we are inside a conda environment; otherwise we have
    # no reliable way to query / modify these packages.
    if _running_in_conda():
        if conda_exe is None:
            raise RuntimeError(
                "Cannot find conda/mamba executable in PATH, so I cannot check or "
                "fix libdeflate/htslib via 'conda list'. "
                "Please make sure libdeflate=1.25 and htslib=1.21 are installed."
            )

        packages_to_fix: List[str] = []
        for pkg_name, required_version in REQUIRED_CONDA_PACKAGES.items():
            current_version = _get_conda_package_version(conda_exe, pkg_name)
            if current_version != required_version:
                packages_to_fix.append(f"{pkg_name}={required_version}")

        if packages_to_fix:
            # Always solve with a compatible libcurl range together with htslib/libdeflate
            install_specs = packages_to_fix + [LIBCURL_SOLVER_SPEC]

            if not auto_install:
                raise RuntimeError(
                    "The following conda packages are missing or have incompatible versions:\n"
                    "  - " + "\n  - ".join(packages_to_fix) + "\n"
                    "Please install them manually, for example:\n"
                    f"  {conda_exe} install -c bioconda -c conda-forge "
                    + " ".join(install_specs)
                )

            cmd = [
                conda_exe,
                "install",
                "-y",
                "-c",
                "bioconda",
                "-c",
                "conda-forge",
            ] + install_specs

            print(
                f"[extract_hla_read] Installing / updating core libraries via: {' '.join(cmd)}",
                file=sys.stderr,
            )
            subprocess.run(cmd, check=True)

            # Re-check to make sure we actually got the desired libdeflate/htslib versions.
            still_bad: List[str] = []
            for pkg_name, required_version in REQUIRED_CONDA_PACKAGES.items():
                current_version = _get_conda_package_version(conda_exe, pkg_name)
                if current_version != required_version:
                    still_bad.append(
                        f"{pkg_name}: required={required_version}, "
                        f"installed={current_version or 'missing'}"
                    )

            if still_bad:
                raise RuntimeError(
                    "Failed to ensure required library versions for some packages:\n  - "
                    + "\n  - ".join(still_bad)
                )

    # ---------- Step 2: check CLI tools (samtools / bam) ----------
    # If we are not allowed to auto-install and something is missing, fail early.
    if missing_binaries and not auto_install:
        missing_str = ", ".join(missing_binaries)
        raise RuntimeError(
            f"Missing required tools: {missing_str}. "
            "Please install them manually, for example:\n"
            "  conda install -c bioconda -c conda-forge samtools=1.21 bamutil\n"
            "and also make sure libdeflate=1.25 and htslib=1.21 are installed."
        )

    # If something is missing and we are allowed to install, we must be in conda
    # and have a conda-like executable available.
    if missing_binaries:
        if not _running_in_conda():
            missing_str = ", ".join(missing_binaries)
            raise RuntimeError(
                "Missing required tools and current process does not appear to run "
                "inside a conda environment. Auto-installation is disabled.\n"
                f"Please install these tools manually: {missing_str}.\n"
                "Example:\n"
                "  conda install -c bioconda -c conda-forge samtools=1.21 bamutil"
            )

        if conda_exe is None:
            missing_str = ", ".join(missing_binaries)
            raise RuntimeError(
                "Missing required tools but no conda/mamba executable was found in PATH.\n"
                f"Please install these tools manually: {missing_str}."
            )

        # Map binaries to conda packages
        packages = sorted({REQUIRED_TOOLS[bin_name] for bin_name in missing_binaries})

        cmd = [
            conda_exe,
            "install",
            "-y",
            "-c",
            "bioconda",
            "-c",
            "conda-forge",
        ] + packages

        print(
            f"[extract_hla_read] Installing missing tools via: {' '.join(cmd)}",
            file=sys.stderr,
        )
        subprocess.run(cmd, check=True)

        # Re-check after installation
        still_missing = [
            bin_name for bin_name in missing_binaries if shutil.which(bin_name) is None
        ]
        if still_missing:
            raise RuntimeError(
                "Failed to install some required tools automatically. "
                f"Still missing: {', '.join(still_missing)}"
            )

def run_extraction(sample_id: str, bam_path: Path, ref: str, outdir: Path,
                   spec_hla_root: str | None = None) -> None:
    """
    Locate ExtractHLAread.sh under <spec_hla_root>/script and run it via bash
    with the provided arguments.

    Upstream derives the script directory from ``__file__`` (the module
    lives in iobrpy/SpecHLA); the port resolves the SAME tree through
    ``_resolve_spec_hla_root`` — same script, same command tokens.
    """
    root = Path(_resolve_spec_hla_root(spec_hla_root, quiet=True))
    sh_path = root / "script" / "ExtractHLAread.sh"

    if not sh_path.is_file():
        raise FileNotFoundError(
            f"Cannot find ExtractHLAread.sh at {sh_path}. "
            "Please make sure it is included in the installed iobrpy package."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash",
        str(sh_path),
        "-s",
        sample_id,
        "-b",
        str(bam_path),
        "-r",
        ref,
        "-o",
        str(outdir),
    ]

    # Print the command for debugging
    print("[extract_hla_read] Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Sample collection & name inference
# ---------------------------------------------------------------------------

STAR_SUFFIX = "_Aligned.sortedByCoord.out.bam"


def infer_sample_id(bam_path: Path) -> str:
    """
    Infer sample id from a BAM file name.

    Rules
    -----
    1) If the name ends with "_Aligned.sortedByCoord.out.bam",
       strip that suffix.
    2) Otherwise, strip the ".bam" suffix.

    The resulting id will be used as:
        -s for ExtractHLAread.sh
        -n for SpecHLA_RNAseq.sh
    """
    name = bam_path.name
    if name.endswith(STAR_SUFFIX):
        return name[: -len(STAR_SUFFIX)]
    if name.endswith(".bam"):
        return name[: -len(".bam")]

    # In practice we only look at *.bam, so this should not happen.
    raise ValueError(f"Invalid BAM file name (not ending with .bam): {bam_path}")


def collect_samples(bam_dir: Path) -> List[Tuple[str, Path]]:
    """
    Scan the BAM directory and return a list of (sample_id, bam_path).
    """
    bam_files = sorted(bam_dir.glob("*.bam"))
    samples: List[Tuple[str, Path]] = []
    for bam in bam_files:
        sample_id = infer_sample_id(bam)
        samples.append((sample_id, bam.resolve()))
    return samples

# ---------------------------------------------------------------------------
# ExtractHLAread phase
# ---------------------------------------------------------------------------

def run_extract_phase(
    samples: List[Tuple[str, Path]],
    ref: str,
    extract_root: Path,
    spec_hla_root: str | None = None,
) -> None:
    """
    Run ExtractHLAread.sh for all samples.

    For each sample, results are written under:
        <extract_root>/<sample_id>/

    A per-sample "done" marker file is used to support resume / skip:
        <extract_root>/<sample_id>/<sample_id>.ExtractHLAread.done

    If the marker already exists, the sample is skipped. The marker is
    created only after ExtractHLAread.sh finishes successfully.
    """
    if not samples:
        print("[HLA_typing] No BAM files found; nothing to do.")
        return

    print(f"[HLA_typing] Found {len(samples)} samples for ExtractHLAread.")
    extract_root.mkdir(parents=True, exist_ok=True)

    # Dependency check and optional auto-install. Mirrors the CLI wrapper.
    try:
        ensure_dependencies(auto_install=True)
    except RuntimeError as exc:
        print(
            "[HLA_typing] ERROR while checking/installing dependencies for "
            f"ExtractHLAread:\n{exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    total = len(samples)
    print("[HLA_typing] Starting ExtractHLAread for all samples...")

    for idx, (sample_id, bam_path) in enumerate(
        tqdm(samples, desc="[ExtractHLAread]", unit="sample", total=total),
        start=1,
    ):
        sample_outdir = extract_root / sample_id
        sample_outdir.mkdir(parents=True, exist_ok=True)

        # Per-sample "done" marker to support resume / skip.
        done_flag = sample_outdir / f"{sample_id}.ExtractHLAread.done"
        if done_flag.exists():
            tqdm.write(
                f"[HLA_typing] [ExtractHLAread] Sample {sample_id} ({idx}/{total}) "
                f"already completed (found {done_flag.name}); skipping."
            )
            continue

        tqdm.write(f"[HLA_typing] [ExtractHLAread] Sample {sample_id} ({idx}/{total})")
        run_extraction(
            sample_id=sample_id,
            bam_path=bam_path,
            ref=ref,
            outdir=sample_outdir,
            spec_hla_root=spec_hla_root,
        )

        # Create the done marker only after successful completion.
        try:
            done_flag.touch()
        except Exception as e:
            tqdm.write(
                f"[HLA_typing] WARNING: failed to create done marker {done_flag}: {e}"
            )

# ---------------------------------------------------------------------------
# SpecHLA phase: locate FASTQs and call run_spechla_rnaseq
# ---------------------------------------------------------------------------

def find_fastqs_for_sample(sample_dir: Path, sample_id: str) -> Tuple[Path, Path]:
    """
    Find R1 and R2 FASTQ.gz files for a sample under ExtractHLAread/<sample_id>.

    The function searches for files named "*_1.fq.gz" and "*_2.fq.gz"
    and prefers matches whose basename starts with the sample id. If no
    such preference matches exist, it falls back to the first candidate.
    """
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    r1_candidates = sorted(sample_dir.glob("*_1.fq.gz"))
    r2_candidates = sorted(sample_dir.glob("*_2.fq.gz"))

    if not r1_candidates or not r2_candidates:
        raise FileNotFoundError(
            f"Could not find *_1.fq.gz / *_2.fq.gz under {sample_dir} "
            f"(sample {sample_id})."
        )

    def choose(cands):
        # Prefer files whose name starts with the sample id.
        for p in cands:
            if p.name.startswith(sample_id):
                return p
        return cands[0]

    r1 = choose(r1_candidates)
    r2 = choose(r2_candidates)
    return r1, r2

def hla_result_has_sample(sample_dir: Path, sample_id: str) -> bool:
    """
    Check whether hla.result.txt (or hla.results.txt) contains a data row whose
    'Sample' column equals sample_id.
    Return True only if:
      - result file exists
      - header contains 'Sample'
      - at least one data row has Sample == sample_id
    """
    candidates = [
        sample_dir / "hla.result.txt",
        sample_dir / "hla.results.txt",
    ]
    result_path = None
    for p in candidates:
        if p.is_file():
            result_path = p
            break
    if result_path is None:
        return False

    try:
        lines = result_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False

    # drop empty lines
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        return False

    # skip leading comment/version lines like "# version: ..."
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        i += 1
    if i >= len(lines):
        return False

    header = lines[i]
    i += 1

    # Prefer tab-split; fallback to any whitespace
    header_cols = header.split("\t") if "\t" in header else header.split()
    if not header_cols:
        return False

    # find Sample column index
    try:
        sample_idx = header_cols.index("Sample")
    except ValueError:
        # tolerate case variations
        sample_idx = None
        for k, col in enumerate(header_cols):
            if col.strip().lower() == "sample":
                sample_idx = k
                break
        if sample_idx is None:
            return False

    # check data rows
    for ln in lines[i:]:
        if ln.lstrip().startswith("#"):
            continue
        cols = ln.split("\t") if "\t" in ln else ln.split()
        if len(cols) <= sample_idx:
            continue
        if cols[sample_idx] == sample_id:
            return True

    return False


def run_spechla_phase(
    samples: List[Tuple[str, Path]],
    threads: int,
    use_exon: int,
    extract_root: Path,
    spechla_outdir: Path,
    spec_hla_root: str | None = None,
) -> None:
    """
    Run SpecHLA in RNA-seq mode for all samples.

    Parameters
    ----------
    samples :
        List of (sample_id, bam_path). The BAM path is not used in this phase,
        because SpecHLA_RNAseq.sh works from the FASTQs produced by
        ExtractHLAread.sh.
    threads :
        Number of threads for SpecHLA (-j).
    use_exon :
        SpecHLA pipeline type (-u). 1 = exon/RNA (default), 0 = WGS.
    extract_root :
        Root directory that holds ExtractHLAread/<sample_id>/... with FASTQs.
    spechla_outdir :
        Root directory for SpecHLA results.

    Layout and "done" marker
    ------------------------
    The SpecHLA workflow is assumed to create one folder per sample:
        <spechla_outdir>/<sample_id>/

    This function additionally uses a per-sample marker file:
        <spechla_outdir>/<sample_id>/<sample_id>.SpecHLA.done

    If the marker already exists, the sample is skipped. The marker is
    created only after SpecHLA_RNAseq.sh finishes successfully.
    """
    if not samples:
        print("[HLA_typing] No samples to process in SpecHLA phase.")
        return

    spechla_outdir.mkdir(parents=True, exist_ok=True)

    # Set up SpecHLA environment once.
    spec_hla_root = detect_spec_hla_root(spec_hla_root)
    ensure_python_deps()
    ensure_external_tools(spec_hla_root)
    ensure_spechap_built(spec_hla_root, threads)

    bowtie2_build_path = detect_bowtie2_build()
    if use_exon == 1:
        drb_ref_relpath = os.path.join(
            "db", "ref", "hla_gen.format.filter.extend.DRB.no26789.fasta"
        )
    else:
        drb_ref_relpath = os.path.join(
            "db", "ref", "hla_gen.format.filter.extend.DRB.no26789.v2.fasta"
        )
    ensure_bowtie2_index(spec_hla_root, bowtie2_build_path, drb_ref_relpath)

    total = len(samples)
    print(f"[HLA_typing] Starting SpecHLA for {total} samples...")

    for idx, (sample_id, _bam_path) in enumerate(
        tqdm(samples, desc="[SpecHLA]", unit="sample", total=total),
        start=1,
    ):
        # Directory where SpecHLA stores results for this sample.
        spechla_sample_dir = spechla_outdir / sample_id
        done_flag = spechla_sample_dir / f"{sample_id}.SpecHLA.done"

        # Skip sample if the done marker is already present.
        if done_flag.exists():
            tqdm.write(
                f"[HLA_typing] [SpecHLA] Sample {sample_id} ({idx}/{total}) "
                f"already completed (found {done_flag.name}); skipping."
            )
            continue

        # FASTQ files should be located under ExtractHLAread/<sample_id>/.
        sample_dir = extract_root / sample_id
        r1, r2 = find_fastqs_for_sample(sample_dir, sample_id)

        tqdm.write(f"[HLA_typing] [SpecHLA] Sample {sample_id} ({idx}/{total})")
        # SpecHLA_RNAseq.sh is expected to put results into
        #   <spechla_outdir>/<sample_id>/
        # (this is the default layout of the upstream workflow).
        run_spechla_rnaseq(
            spec_hla_root=spec_hla_root,
            sample_name=sample_id,        # -n
            read1=str(r1),                # -1
            read2=str(r2),                # -2
            outdir=str(spechla_outdir),   # -o
            threads=threads,              # -j
            use_exon=use_exon,            # -u
        )

        # Create DONE only if the result table contains this sample in 'Sample' column
        try:
            spechla_sample_dir.mkdir(parents=True, exist_ok=True)

            if hla_result_has_sample(spechla_sample_dir, sample_id):
                done_flag.touch()
            else:
                tqdm.write(
                    f"[HLA_typing] WARNING: SpecHLA finished but '{spechla_sample_dir}/hla.result.txt' "
                    f"does not contain Sample={sample_id}; NOT creating {done_flag.name}."
                )
        except Exception as e:
            tqdm.write(
                f"[HLA_typing] WARNING: failed to create done marker {done_flag}: {e}"
            )

# ---------------------------------------------------------------------------
# Merge per-sample HLA result tables
# ---------------------------------------------------------------------------

def merge_hla_results(
    samples: List[Tuple[str, Path]],
    spechla_outdir: Path,
    outdir: Path,
) -> None:
    """
    Merge per-sample HLA result tables into a single file.

    For each sample_id, we look for:
        <spechla_outdir>/<sample_id>/hla.result.txt
    and, if missing,
        <spechla_outdir>/<sample_id>/hla.results.txt

    The file format is assumed to follow the example provided:

        # version: IPD-IMGT/HLA 3.38.0
        Sample  HLA_A_1 HLA_A_2 ...

    The merged file will contain:
        - The version line and header line from the first sample that has
          a non-empty result file.
        - All data lines from all samples (non-empty, non-comment lines
          after the header).

    The output is written as:
        <outdir>/hla_result_merged.txt
    """
    sample_result_files: List[Tuple[str, Path]] = []

    for sample_id, _bam_path in samples:
        sample_dir = spechla_outdir / sample_id
        candidates = [
            sample_dir / "hla.result.txt",
            sample_dir / "hla.results.txt",
        ]
        found = None
        for path in candidates:
            if path.is_file():
                found = path
                break
        if found is None:
            print(
                f"[HLA_typing] WARNING: no hla.result(s).txt found for sample "
                f"{sample_id} under {sample_dir}; skipping this sample.",
                file=sys.stderr,
            )
            continue
        sample_result_files.append((sample_id, found))

    if not sample_result_files:
        print(
            "[HLA_typing] WARNING: no HLA result files were found; "
            "merged table will not be created.",
            file=sys.stderr,
        )
        return

    merged_path = outdir / "hla_result_merged.txt"
    print(f"[HLA_typing] Merging HLA result tables into: {merged_path}")

    with merged_path.open("w", encoding="utf-8") as out_f:
        wrote_header = False

        for sample_id, hla_file in sample_result_files:
            with hla_file.open("r", encoding="utf-8") as f:
                # Strip trailing newlines but keep empty-line information via strip check
                lines = [line.rstrip("\n") for line in f]

            # Remove leading empty lines
            lines = [ln for ln in lines if ln.strip() != ""]
            if not lines:
                print(
                    f"[HLA_typing] WARNING: HLA result file for sample "
                    f"{sample_id} is empty: {hla_file}",
                    file=sys.stderr,
                )
                continue

            idx = 0
            local_version = None
            # Optional version/comment lines at the top (usually exactly one)
            while idx < len(lines) and lines[idx].startswith("#"):
                if local_version is None:
                    local_version = lines[idx]
                idx += 1

            if idx >= len(lines):
                print(
                    f"[HLA_typing] WARNING: HLA result file for sample "
                    f"{sample_id} does not contain a header line: {hla_file}",
                    file=sys.stderr,
                )
                continue

            local_header = lines[idx]
            idx += 1

            if not wrote_header:
                # Write version line (if any) and header once.
                if local_version is not None:
                    out_f.write(local_version + "\n")
                out_f.write(local_header + "\n")
                wrote_header = True

            # Remaining lines are data lines.
            for data_line in lines[idx:]:
                if data_line.strip() == "":
                    continue
                # Avoid accidentally duplicating header lines
                if data_line.startswith("Sample\t") or data_line.startswith("Sample "):
                    continue
                out_f.write(data_line + "\n")

    print(f"[HLA_typing] HLA result tables merged successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for the hla_typing subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="hla_typing",
        description=(
            "Batch HLA typing: run ExtractHLAread + SpecHLA on all BAM files "
            "in a directory, then merge per-sample results."
        ),
    )
    parser.add_argument(
        "-b",
        "--bam-dir",
        dest="bam_dir",
        required=True,
        help="Directory containing BAM files.",
    )
    parser.add_argument(
        "-r",
        "--ref",
        dest="ref",
        required=True,
        choices=["hg19", "hg38"],
        help="Reference genome passed to ExtractHLAread (hg19 or hg38).",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        required=True,
        help=(
            "Root output directory. "
            "Will create subfolders 'ExtractHLAread' and 'SpecHLA'."
        ),
    )
    parser.add_argument(
        "-j",
        "--threads",
        dest="threads",
        type=int,
        default=8,
        help="Number of threads for SpecHLA (-j). Default: 8.",
    )
    parser.add_argument(
        "-u",
        "--use-exon",
        dest="use_exon",
        type=int,
        choices=[0, 1],
        default=1,
        help="SpecHLA pipeline type (-u). 1 = exon/RNA (default), 0 = WGS.",
    )
    return parser


def main(argv: List[str] | None = None,
         spec_hla_root: str | None = None) -> None:
    """
    Entry point for the hla_typing subcommand.

    Parameters
    ----------
    argv :
        Optional list of arguments; if None, sys.argv[1:] is used.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    bam_dir = Path(os.path.expanduser(args.bam_dir)).resolve()
    if not bam_dir.is_dir():
        parser.error(f"BAM directory does not exist: {bam_dir}")

    outdir = Path(os.path.expanduser(args.outdir)).resolve()
    extract_root = outdir / "ExtractHLAread"
    spechla_outdir = outdir / "SpecHLA"

    # Resolve the SpecHLA asset root once for the extract phase (silent
    # here; the printing detect_spec_hla_root() call stays inside
    # run_spechla_phase, exactly where the upstream prints it).
    resolved_root = _resolve_spec_hla_root(spec_hla_root, quiet=True)

    samples = collect_samples(bam_dir)
    if not samples:
        parser.error(f"No .bam files found under directory: {bam_dir}")

    print(f"[HLA_typing] Using BAM dir : {bam_dir}")
    print(f"[HLA_typing] Output root  : {outdir}")
    print(f"[HLA_typing] Reference    : {args.ref}")
    print(f"[HLA_typing] Threads      : {args.threads}")
    print(f"[HLA_typing] Use exon (-u): {args.use_exon}")
    print(f"[HLA_typing] Detected {len(samples)} sample(s).")

    # 1) Run ExtractHLAread for all samples.
    run_extract_phase(samples, args.ref, extract_root, resolved_root)

    # 2) Run SpecHLA for all samples.
    run_spechla_phase(
        samples,
        args.threads,
        args.use_exon,
        extract_root,
        spechla_outdir,
        resolved_root,
    )

    # 3) Merge per-sample HLA result tables.
    merge_hla_results(samples, spechla_outdir, outdir)

    # ---------------- IOBRpy banner ----------------
    print("   ")
    print_colorful_message("#########################################################", "blue")
    print_colorful_message(" IOBRpy: Immuno-Oncology Biological Research using Python ", "cyan")
    print_colorful_message(" If you encounter any issues, please report them at ", "cyan")
    print_colorful_message(" https://github.com/IOBR/IOBRpy/issues ", "cyan")
    print_colorful_message("#########################################################", "blue")
    print(" Author: Haonan Huang, Dongqiang Zeng")
    print(" Email: interlaken@smu.edu.cn ")
    print_colorful_message("#########################################################", "blue")
    print("   ")

def hla_typing_original(argv=None):
    """Run the untouched upstream ``iobrpy.workflow.hla_typing.main(argv)``.
    Escape hatch for ``backend='python'``; propagates the upstream
    ``SystemExit`` (the iobrx public wrapper catches it and returns the
    code)."""
    from iobrpy.workflow.hla_typing import main as _orig_main

    return _orig_main(argv)


if __name__ == "__main__":
    # Mirrors `python -m iobrpy.main hla_typing ...` (upstream entry).
    main()
