"""tme_profile orchestrator parity gate (pytest.mark.full).

Runs the ORIGINAL ``iobrpy tme_profile`` CLI chain and ``iobrx.tme_profile``
on a reduced 3-sample subset of the frozen STAD TPM matrix and compares ALL
NINE output files:

    01-signatures/calculate_sig_score.csv
    02-tme/{cibersort,IPS,estimate,mcpcounter,quantiseq,epic}_results.csv
    02-tme/deconvo_merged.csv
    03-LR_cal/lr_cal.csv

Contract: byte-identical files, with the single designed exception of the
``P-value_CIBERSORT`` column in ``cibersort_results.csv`` and
``deconvo_merged.csv`` — upstream ``do_perm`` draws OS entropy per run
(``SeedSequence()`` unseeded), so that column is not reproducible by ANY two
runs of the original itself (the frozen gold rep1/rep2 differ in exactly that
column). The comparison strips that one field in TEXT space (csv module, no
float re-parsing) and requires the remaining bytes to be equal.

Input resolution (first hit wins):
  1. ``$IOBRX_TME_PROFILE_INPUT``: path to a genes x samples TPM CSV (the
     frozen ``TPM_stad10.csv`` from the module-spec baseline); its first 3
     sample columns are extracted with pure text processing so the exact
     float text of the frozen file is preserved.
  2. ``iobrx.load_official`` derivation: ``eset_stad`` x ``anno_grch38``
     through the ORIGINAL ``anno_eset`` (same recipe as
     ``test_parity_official.py``), written to a temp CSV.
  3. otherwise the test skips (offline without frozen input).

Modes: the DEFAULT test exercises ``parallel=True`` (the process-split extra
path — its bytes are proven identical to the sequential contract path on the
frozen full-size gold); the sequential default path is additionally covered
by ``test_tme_profile_sequential`` when ``IOBRX_TME_PROFILE_SEQ=1`` (it costs
another ~150 s; the full-size sequential-vs-gold byte parity lives in the
build scratch harness). Both fast runs reuse ONE original-CLI output fixture.

Runtime: ~200 s default (original CLI ~150 s + fast parallel ~45 s).

Enable with::

    IOBRX_TME_PROFILE_INPUT=/path/to/TPM_stad10.csv pytest -q -m full \
        tests/test_parity_tme_profile.py
"""
from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.full

THREADS = 4
N_SAMPLES = 3

REL_FILES = [
    "01-signatures/calculate_sig_score.csv",
    "02-tme/cibersort_results.csv",
    "02-tme/IPS_results.csv",
    "02-tme/estimate_results.csv",
    "02-tme/mcpcounter_results.csv",
    "02-tme/quantiseq_results.csv",
    "02-tme/epic_results.csv",
    "02-tme/deconvo_merged.csv",
    "03-LR_cal/lr_cal.csv",
]
# the ONLY columns exempt from byte equality (upstream unseeded permutations)
PVALUE_EXEMPT = {
    "02-tme/cibersort_results.csv": "P-value_CIBERSORT",
    "02-tme/deconvo_merged.csv": "P-value_CIBERSORT",
}


def _fast_entry():
    """Public API when wired by the package, else the fast module directly."""
    import iobrx

    if hasattr(iobrx, "tme_profile"):
        return iobrx.tme_profile
    from iobrx._fast.tme_profile_fast import tme_profile_fast

    return tme_profile_fast


def _subset_csv(tmpdir: Path) -> Path:
    """3-sample subset of a genes x samples TPM CSV, text-exact per field."""
    src = os.environ.get("IOBRX_TME_PROFILE_INPUT")
    if src and Path(src).is_file():
        src = Path(src)
    else:
        try:
            import iobrx
            from iobrpy.workflow.anno_eset import anno_eset as orig_anno

            stad = iobrx.load_official("eset_stad")
            anno = iobrx.load_official("anno_grch38")
            sym = orig_anno(stad.copy(), anno.copy(), symbol="symbol",
                            probe="id", method="mean")
        except Exception as exc:  # offline and no frozen input
            pytest.skip(
                "tme_profile parity needs $IOBRX_TME_PROFILE_INPUT (frozen "
                f"TPM_stad10.csv) or official data access: {exc}"
            )
        src = tmpdir / "tpm_full.csv"
        sym.to_csv(src)

    sub = tmpdir / "tpm_subset3.csv"
    keep = list(range(N_SAMPLES + 1))  # id column + first 3 samples
    with open(src) as fh:
        header = fh.readline().rstrip("\n")
        cols = header.split(",")
        if len(cols) < N_SAMPLES + 1:
            pytest.skip(f"input has only {len(cols) - 1} samples")
        lines = [",".join(cols[i] for i in keep)]
        for line in fh:
            parts = line.rstrip("\n").split(",")
            lines.append(",".join(parts[i] for i in keep))
    sub.write_text("\n".join(lines) + "\n")
    return sub


def _run_original_cli(inp: Path, outdir: Path) -> None:
    """The ORIGINAL chain: python -m iobrpy.main tme_profile (which itself
    shells out to the `iobrpy` console script per step -> PATH must carry the
    venv bin; do NOT resolve() sys.executable: the venv python may be a
    symlink into a different bin dir)."""
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-m", "iobrpy.main", "tme_profile",
           "-i", str(inp), "-o", str(outdir), "--threads", str(THREADS)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                          env=env)
    assert proc.returncode == 0, (
        f"original CLI rc={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}"
        f"\nstderr tail:\n{proc.stderr[-2000:]}"
    )


def _strip_field(path: Path, colname: str) -> bytes:
    """File bytes with one CSV field removed from every row (text space)."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    idx = rows[0].index(colname)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in rows:
        writer.writerow([v for i, v in enumerate(row) if i != idx])
    return buf.getvalue().encode()


def _assert_outputs_equal(orig_dir: Path, fast_dir: Path) -> None:
    for rel in REL_FILES:
        po, pf = orig_dir / rel, fast_dir / rel
        assert pf.is_file(), f"missing fast output {rel}"
        bo, bf = po.read_bytes(), pf.read_bytes()
        if bo == bf:
            continue
        exempt = PVALUE_EXEMPT.get(rel)
        assert exempt is not None, (
            f"{rel}: bytes differ and the file carries no exempt column"
        )
        assert exempt in bo.decode().splitlines()[0], (
            f"{rel}: exempt column {exempt} not in header"
        )
        so, sf = _strip_field(po, exempt), _strip_field(pf, exempt)
        assert so == sf, f"{rel}: bytes differ outside the exempt {exempt} column"


@pytest.fixture(scope="module")
def original_out(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tme_profile")
    inp = _subset_csv(tmp)
    out = tmp / "orig"
    _run_original_cli(inp, out)
    return inp, out


def test_tme_profile_parallel(original_out):
    """parallel=True (process-split sig_score + pooled light steps)."""
    inp, orig_out = original_out
    fast = _fast_entry()
    with tempfile.TemporaryDirectory(prefix="iobrx-tme-par-") as td:
        fast(str(inp), td, threads=THREADS, parallel=True)
        _assert_outputs_equal(orig_out, Path(td))


@pytest.mark.skipif(os.environ.get("IOBRX_TME_PROFILE_SEQ", "") != "1",
                    reason="sequential contract path opt-in "
                           "(IOBRX_TME_PROFILE_SEQ=1); +~150 s")
def test_tme_profile_sequential(original_out):
    """The default sequential contract path (original step order)."""
    inp, orig_out = original_out
    fast = _fast_entry()
    with tempfile.TemporaryDirectory(prefix="iobrx-tme-seq-") as td:
        fast(str(inp), td, threads=THREADS)
        _assert_outputs_equal(orig_out, Path(td))
