"""Parity gates for the ``iobrx.runall`` end-to-end orchestrator port.

Three tiers, in increasing cost:

1. DRY-RUN CONSOLE PARITY (fast, no tools, default suite): the ORIGINAL
   ``iobrpy.workflow.runall.main`` and the ported ``iobrx._fast.runall_fast
   .runall_argv`` are run over identical argv on stub trees under
   ``--dry_run`` (plus ``--resume`` on completed trees and the argparse /
   missing-matrix error paths); their captured stdout must be BYTE-identical
   and their exit codes equal. This pins the whole verbatim orchestration:
   mode branching, step order, command construction, per-step default
   injection, the sectioned + auto flag routers, legacy concurrency-flag
   absorption, resume/dry protocols and every console line.
2. STRUCTURE / PARSER gates (fast): the shared routing tables and helper
   functions are source-identical to upstream; a seeded router battery
   agrees on 200 random token lists; each step's mirror parser reproduces
   the ``iobrpy.main`` CLI defaults (the LR_cal ``data_type='tpm'`` vs
   function-default ``'count'`` class of traps) including the case-sensitive
   ``--QN`` quirk.
3. FULL-CHAIN MINIATURE (marker ``full``; synthetic mini FASTQ + the R4-style
   deterministic fake fastp/multiqc/salmon/run-trust4 binaries; NO real
   heavy computation): the original orchestrator (spawning real ``iobrpy``
   child CLIs) and ``iobrx.runall`` (in-process ported substeps) run the
   whole salmon chain on ONE miniature sample whose fake ``quant.sf`` is a
   realistic GENCODE-format template covering every downstream resource gene
   set (LM22 / EPIC sigGenes / TIL10 / MCP-counter / ESTIMATE / IPS /
   signature_tme / LR pancan). Gates: identical exit codes; output trees
   byte-identical EXCEPT the documented non-determinism (gzip mtime in the
   merged ``*.tsv.gz`` -> compared decompressed; cibersort's unseeded
   ``P-value`` column and its propagation into ``deconvo_merged.csv`` ->
   compared excluding that column, R3 contract); recorded external-tool
   command lines token-identical after root normalization.

Run with the repo source on the path::

    PYTHONPATH=<repo>/src pytest -q tests/test_parity_runall.py            # tiers 1-2
    PYTHONPATH=<repo>/src pytest -q tests/test_parity_runall.py --run-full # tier 3
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import inspect
import io
import json
import os
import random
import re
import shutil
import stat
import sys

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# module handles
# ---------------------------------------------------------------------------
def _ra():
    return pytest.importorskip(
        "iobrx._fast.runall_fast",
        reason="needs PYTHONPATH=<repo>/src (or an installed iobrx)",
    )


def _orig_runall():
    return pytest.importorskip("iobrpy.workflow.runall")


def _iobrx():
    return pytest.importorskip("iobrx")


# ---------------------------------------------------------------------------
# stub trees for the dry-run / resume gates
# ---------------------------------------------------------------------------
def _stub_tree(outdir: str, mode: str) -> None:
    """Numbered dirs + the artifacts a complete --dry_run pass needs."""
    quant = "02-salmon" if mode == "salmon" else "02-star"
    for d in ["01-qc", "03-tpm", "04-signatures", "05-tme", "06-LR_cal",
              "07-TCRBCR", quant]:
        os.makedirs(os.path.join(outdir, d), exist_ok=True)
    name = ("runall_salmon_tpm.tsv" if mode == "salmon"
            else "runall_star_ReadsPerGene.tsv")
    with open(os.path.join(outdir, quant, name), "w") as f:
        f.write("Name\tTPM\n")
    with open(os.path.join(outdir, "03-tpm", "tpm_matrix.csv"), "w") as f:
        f.write("ID,s1\nGENEA,1.0\n")
    for m in ["cibersort", "IPS", "estimate", "mcpcounter", "quantiseq", "epic"]:
        with open(os.path.join(outdir, "05-tme", f"{m}_results.csv"), "w") as f:
            f.write(f"ID,{m}_x\ns1,0.5\n")


def _completed_tree(outdir: str, mode: str) -> None:
    """Stub tree + every resume flag/output -> --resume skips all steps."""
    _stub_tree(outdir, mode)
    quant = "02-salmon" if mode == "salmon" else "02-star"
    step = "batch_salmon" if mode == "salmon" else "batch_star_count"
    merge_flag = ".merge_salmon.done" if mode == "salmon" else ".merge_star_count.done"
    for d, fn, body in [
        ("01-qc", ".fastq_qc.done", "done\n"),
        ("01-qc", "placeholder.txt", "x"),
        (quant, f".{step}.done", "done\n"),
        (quant, merge_flag, "done\n"),
        ("04-signatures", "calculate_sig_score.csv", "ID,x\ns1,1\n"),
        ("05-tme", "deconvo_merged.csv", "ID,x\ns1,1\n"),
        ("06-LR_cal", "lr_cal.csv", "ID,x\ns1,1\n"),
        ("07-TCRBCR", ".trust4.done", "done\n"),
        ("07-TCRBCR", "placeholder.txt", "x"),
    ]:
        p = os.path.join(outdir, d, fn)
        if not os.path.exists(p):
            with open(p, "w") as f:
                f.write(body)


def _run_both(argv, tmp_path, tag):
    """Run original main() and ported runall_argv() over the SAME argv and
    the SAME paths; assert byte-identical stdout and equal exit codes."""
    orig = _orig_runall()
    port = _ra()
    buf_o, buf_p = io.StringIO(), io.StringIO()
    err_o, err_p = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_o), contextlib.redirect_stderr(err_o):
        try:
            orig.main(list(argv))
            rc_o = 0
        except SystemExit as e:
            rc_o = e.code if isinstance(e.code, int) else 1
    with contextlib.redirect_stdout(buf_p), contextlib.redirect_stderr(err_p):
        rc_p = port.runall_argv(list(argv))
    so, sp = buf_o.getvalue(), buf_p.getvalue()
    assert so == sp, (
        f"[{tag}] stdout differs ({len(so)} vs {len(sp)} bytes):\n"
        + "\n".join(
            __import__("difflib").unified_diff(
                so.splitlines(), sp.splitlines(), "orig", "port", lineterm="", n=1)
        )[:4000]
    )
    assert rc_o == rc_p, f"[{tag}] rc mismatch: orig={rc_o} port={rc_p}"
    return so


def _dry_case(tmp_path, mode, extra, completed=False, resume=False, empty=False):
    outdir = tmp_path / f"out_{mode}"
    fastq = tmp_path / f"fq_{mode}"
    fastq.mkdir(exist_ok=True)
    (fastq / "s1_1.fastq.gz").write_bytes(b"")
    if empty:
        outdir.mkdir(exist_ok=True)
    elif completed:
        _completed_tree(str(outdir), mode)
    else:
        _stub_tree(str(outdir), mode)
    argv = ["--mode", mode, "--outdir", str(outdir), "--fastq", str(fastq)]
    if resume:
        argv.append("--resume")
    else:
        argv.append("--dry_run")
    argv += ["--threads", "4"] + list(extra)
    return _run_both(argv, tmp_path, f"{mode}:{','.join(map(str, extra[:2]))}")


IDX_S = "/refs/salmon_idx"
IDX_T = "/refs/star_idx"


@pytest.mark.parametrize("mode,idx", [("salmon", IDX_S), ("star", IDX_T)])
def test_dry_defaults_plus_index(tmp_path, mode, idx):
    out = _dry_case(tmp_path, mode, ["--index", idx])
    assert "[run] iobrpy fastq_qc" in out
    assert "--num_threads 4" in out
    assert "[dry-run] skipped execution" in out
    assert "[done] runall finished." in out
    if mode == "salmon":
        assert "iobrpy batch_salmon" in out
        assert "--project runall" in out
        assert "--return_feature symbol" in out
        assert "--remove_version" in out
    else:
        assert "iobrpy batch_star_count" in out
        assert "--idtype ensembl" in out


@pytest.mark.parametrize("mode,idx", [("salmon", IDX_S), ("star", IDX_T)])
def test_dry_rich_autorouted_flags(tmp_path, mode, idx):
    extra = ["--index", idx, "--project", "prj1", "--suffix1", "_R1.fastq.gz",
             "--signature", "kegg", "--method", "integration",
             "--mini_gene_count", "5", "--perm", "50", "--platform", "agilent",
             "--features", "ENTREZ_ID", "--reference", "BRef",
             "--data_type", "tpm", "--remove_version", "--se",
             "--length_required", "30"]
    if mode == "salmon":
        extra += ["--gtf", "/refs/a.gtf"]
    out = _dry_case(tmp_path, mode, extra)
    assert "--project prj1" in out          # default 'runall' NOT injected
    assert "--signature kegg" in out        # default 'all' NOT injected
    assert "--mini_gene_count 5" in out


@pytest.mark.parametrize("mode,idx", [("salmon", IDX_S), ("star", IDX_T)])
def test_dry_legacy_scalars_and_warn(tmp_path, mode, idx):
    argv_extra = ["--num_threads", "7", "--parallel_size", "6", "--batch_size", "3",
                  "--index", idx, "--method", "lsei", "--bogus", "zz"]
    outdir = tmp_path / f"out_{mode}"
    fastq = tmp_path / f"fq_{mode}"
    fastq.mkdir(exist_ok=True)
    _stub_tree(str(outdir), mode)
    argv = ["--mode", mode, "--outdir", str(outdir), "--fastq", str(fastq),
            "--dry_run"] + argv_extra
    out = _run_both(argv, tmp_path, f"{mode}:legacy")
    # last legacy threads occurrence wins (parallel_size 6), batch 3
    assert "--num_threads 6" in out
    assert "--batch_size 3" in out
    assert "[warn] Unrecognized flag (ignored by router): --bogus zz" in out
    assert "--method lsei" in out           # routed to quantiseq by value


@pytest.mark.parametrize("mode,idx", [("salmon", IDX_S), ("star", IDX_T)])
def test_dry_sectioned_style(tmp_path, mode, idx):
    merge_step = "merge_salmon" if mode == "salmon" else "merge_star_count"
    extra = ["--index", idx, "fastq_qc", "--se", "cibersort", "--perm", "25",
             merge_step, "--project", "secP"]
    out = _dry_case(tmp_path, mode, extra)
    assert "--perm 25" in out
    assert "--project secP" in out


@pytest.mark.parametrize("mode,idx", [("salmon", IDX_S), ("star", IDX_T)])
def test_resume_completed_tree(tmp_path, mode, idx):
    out = _dry_case(tmp_path, mode, ["--index", idx], completed=True, resume=True)
    assert "[resume] fastq_qc skipped" in out
    assert "[resume] trust4 skipped" in out
    assert "[done] runall finished." in out
    assert "[run]" not in out               # nothing executed


@pytest.mark.parametrize("mode,idx", [("salmon", IDX_S), ("star", IDX_T)])
def test_empty_tree_missing_merged_rc2(tmp_path, mode, idx):
    out = _dry_case(tmp_path, mode, ["--index", idx], empty=True)
    assert "[ERROR] Cannot find merged" in out


def test_invalid_mode_rc2(tmp_path):
    argv = ["--mode", "bogus", "--outdir", str(tmp_path / "x"),
            "--fastq", str(tmp_path / "y")]
    _run_both(argv, tmp_path, "invalid-mode")


# ---------------------------------------------------------------------------
# structure gates: routing tables, helpers, router battery
# ---------------------------------------------------------------------------
def test_shared_structures_identical():
    orig, port = _orig_runall(), _ra()
    assert orig.METHOD_SECTIONS == port.METHOD_SECTIONS
    assert orig.FLAG_BUCKETS == port.FLAG_BUCKETS
    for fn in ["_ensure_dir", "_nonempty", "_normalize_flag_token", "_flag_name",
               "_parse_passthrough_blocks", "_find_latest", "_append_passthrough",
               "_consume_top_level_scalars", "_autobucket"]:
        assert inspect.getsource(getattr(orig, fn)) == inspect.getsource(getattr(port, fn)), fn


def test_router_battery_200():
    orig, port = _orig_runall(), _ra()
    rnd = random.Random(11)
    pool = ["--index", "/x/idx", "--project", "p1", "--suffix1", "_R1.fq.gz", "--se",
            "--num_threads", "4", "--parallel_size", "3", "--num_processes", "5",
            "--batch_size", "2", "--method", "integration", "--method", "lsei",
            "--remove_version", "--gtf", "/x/a.gtf", "--perm", "50", "--qn", "False",
            "--platform", "agilent", "--features", "ENTREZ_ID", "--arrays", "--tumor",
            "--scale_mrna", "--reference", "BRef", "--data_type", "count", "--verbose",
            "--signature", "kegg", "--mini_gene_count", "5", "--adjust_eset",
            "--bogus", "zzz", "fastq_qc", "cibersort", "--idtype", "symbol"]
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(200):
            toks = [rnd.choice(pool) for _ in range(rnd.randint(0, 12))]
            mode = rnd.choice(["salmon", "star"])
            norm = [orig._normalize_flag_token(t) for t in toks]
            assert orig._autobucket(list(norm), mode) == port._autobucket(list(norm), mode)
            assert (orig._consume_top_level_scalars(list(toks))
                    == port._consume_top_level_scalars(list(toks)))
            assert (orig._parse_passthrough_blocks(list(norm))
                    == port._parse_passthrough_blocks(list(norm)))


def test_executor_coverage():
    port = _ra()
    assert set(port._EXECUTORS) | {"trust4"} == {
        "fastq_qc", "batch_salmon", "merge_salmon", "batch_star_count",
        "merge_star_count", "prepare_salmon", "count2tpm", "log2_eset",
        "calculate_sig_score", "cibersort", "IPS", "estimate", "mcpcounter",
        "quantiseq", "epic", "LR_cal", "trust4",
    }


# ---------------------------------------------------------------------------
# mirror-parser defaults (the CLI-vs-function default trap class)
# ---------------------------------------------------------------------------
def test_step_parser_defaults_mirror_cli():
    port = _ra()

    a = port._step_parser("fastq_qc").parse_known_args(
        ["--path1_fastq", "i", "--path2_fastp", "o"])[0]
    assert (a.num_threads, a.suffix1, a.batch_size, a.se, a.length_required) == \
        (8, "_1.fastq.gz", 1, False, 50)

    a = port._step_parser("batch_salmon").parse_known_args(
        ["--index", "i", "--path_fq", "f", "--path_out", "o"])[0]
    assert (a.suffix1, a.batch_size, a.num_threads, a.gtf) == \
        ("_1.fastq.gz", 1, 8, None)

    a = port._step_parser("merge_salmon").parse_known_args(
        ["--path_salmon", "p", "--project", "j"])[0]
    assert a.num_processes is None

    a = port._step_parser("batch_star_count").parse_known_args(
        ["--index", "i", "--path_fq", "f", "--path_out", "o"])[0]
    assert (a.suffix1, a.batch_size, a.num_threads) == ("_1.fastq.gz", 1, 8)

    a = port._step_parser("prepare_salmon").parse_known_args(
        ["--input", "i", "--output", "o"])[0]
    assert (a.eset_path, a.output_matrix, a.return_feature, a.remove_version) == \
        ("i", "o", "symbol", False)

    a = port._step_parser("count2tpm").parse_known_args(
        ["--input", "i", "--output", "o"])[0]
    assert (a.idtype, a.org, a.source, a.id_col, a.length_col, a.gene_symbol_col,
            a.check_data, a.remove_version, a.effLength_csv) == \
        ("ensembl", "hsa", "local", "id", "eff_length", "symbol", False, False, None)

    a = port._step_parser("calculate_sig_score").parse_known_args(
        ["--input", "i", "--output", "o", "--signature", "all"])[0]
    assert (a.score_method, a.mini_gene_count, a.adjust_eset, a.parallel_size) == \
        ("pca", 3, False, 1)

    a = port._step_parser("cibersort").parse_known_args(
        ["--input", "i", "--output", "o"])[0]
    assert (a.perm, a.QN, a.absolute, a.abs_method, a.threads) == \
        (100, True, False, "sig.score", 1)
    # case-sensitivity quirk: --qn is NOT --QN (upstream child ignores it)
    a, leftover = port._step_parser("cibersort").parse_known_args(
        ["--input", "i", "--output", "o", "--qn", "False"])
    assert a.QN is True and leftover == ["--qn", "False"]
    a = port._step_parser("cibersort").parse_known_args(
        ["--input", "i", "--output", "o", "--QN", "False"])[0]
    assert a.QN is False

    a = port._step_parser("estimate").parse_known_args(
        ["--input", "i", "--output", "o"])[0]
    assert a.platform == "affymetrix"

    a = port._step_parser("quantiseq").parse_known_args(
        ["--input", "i", "--output", "o"])[0]
    assert (a.arrays, a.signame, a.tumor, a.mRNAscale, a.method, a.rmgenes) == \
        (False, "TIL10", False, False, "lsei", "unassigned")

    a = port._step_parser("epic").parse_known_args(
        ["--input", "i", "--output", "o"])[0]
    assert a.reference == "TRef"

    # THE documented trap: CLI default is 'tpm' although the plain
    # LR_cal function default is 'count'
    a = port._step_parser("LR_cal").parse_known_args(
        ["--input", "i", "--output", "o"])[0]
    assert (a.data_type, a.id_type, a.cancer_type, a.verbose) == \
        ("tpm", "ensembl", "pancan", False)

    # usage errors keep the upstream exit code 2
    with pytest.raises(SystemExit) as e:
        port._step_parser("merge_salmon").parse_known_args(["--path_salmon", "p"])
    assert e.value.code == 2


# ---------------------------------------------------------------------------
# public API surface
# ---------------------------------------------------------------------------
def test_api_dry_run_and_backends(tmp_path):
    iobrx = _iobrx()
    assert "runall" in iobrx.__all__
    outdir = tmp_path / "api"
    fastq = tmp_path / "fq"
    fastq.mkdir()
    _stub_tree(str(outdir), "salmon")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = iobrx.runall("salmon", outdir, fastq, threads=5, batch_size=2,
                           dry_run=True, unknown=["--index", IDX_S])
    assert res == {"rc": 0}
    out = buf.getvalue()
    assert "--num_threads 5" in out and "--batch_size 2" in out
    assert f"--index {IDX_S}" in out

    # n_threads alias (threads wins when both given)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = iobrx.runall("salmon", outdir, fastq, n_threads=3, dry_run=True,
                           unknown=["--index", IDX_S])
    assert res == {"rc": 0} and "--num_threads 3" in buf.getvalue()

    # invalid mode reproduces the upstream argparse rc 2 (no SystemExit leak)
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        res = iobrx.runall("bogus", outdir, fastq, dry_run=True)
    assert res == {"rc": 2}

    # backend guards
    with pytest.raises(RuntimeError):
        iobrx.runall("salmon", outdir, fastq, backend="rust")
    with pytest.raises(ValueError):
        iobrx.runall("salmon", outdir, fastq, backend="nope")

    # backend='python' = untouched upstream orchestrator (dry: no children)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = iobrx.runall("salmon", outdir, fastq, threads=5, batch_size=2,
                           dry_run=True, unknown=["--index", IDX_S],
                           backend="python")
    assert res == {"rc": 0}
    assert buf.getvalue() == out          # same console as backend='auto'

    # CLI-style main() keeps the upstream SystemExit behaviour
    port = _ra()
    with pytest.raises(SystemExit) as e:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            port.main(["--mode", "bogus", "--outdir", str(outdir),
                       "--fastq", str(fastq)])
    assert e.value.code == 2


# ---------------------------------------------------------------------------
# tier 3: full-chain miniature (fake external tools, synthetic mini FASTQ)
# ---------------------------------------------------------------------------
_FAKE_PRELUDE = '''
import gzip, hashlib, json, os, sys

LOG = os.environ["FAKE_CALL_LOG"]
BIN = os.environ.get("FAKE_BIN_TAG", sys.argv[0])

def record(argv):
    with open(LOG, "a") as f:
        f.write(json.dumps({"bin": BIN, "argv": argv}) + "\\n")

def digest(*paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()

def getflag(argv, flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default

def gz_out(path, payload: bytes):
    with gzip.GzipFile(path, "wb", compresslevel=6, mtime=0) as f:
        f.write(payload)
'''

# fastp / multiqc / run-trust4 fakes: identical to the R4 orchestration
# gates (tests/test_parity_orchestration.py) — proven byte-deterministic
# against both the original and the ported substep modules.
_FAKE_FASTP = _FAKE_PRELUDE + '''
argv = sys.argv[1:]
record(argv)
r1 = getflag(argv, "-i"); o1 = getflag(argv, "-o")
r2 = getflag(argv, "-I"); o2 = getflag(argv, "-O")
html = getflag(argv, "--html"); js = getflag(argv, "--json")
d1 = digest(r1)
gz_out(o1, b"cleaned-r1:" + d1.encode() + b"\\n")
if r2 and o2:
    gz_out(o2, b"cleaned-r2:" + digest(r2).encode() + b"\\n")
if html:
    with open(html, "w") as f:
        f.write("<html>fastp stub " + d1 + "</html>\\n")
if js:
    with open(js, "w") as f:
        json.dump({"stub": True, "digest": d1,
                   "reads": len(d1) % 7}, f, indent=1, sort_keys=True)
'''

_FAKE_MULTIQC = _FAKE_PRELUDE + '''
argv = sys.argv[1:]
record(argv)
outdir = getflag(argv, "--outdir", ".")
fname = getflag(argv, "--filename", "multiqc_report")
os.makedirs(outdir, exist_ok=True)
scan = argv[-1]
digests = []
for fn in sorted(os.listdir(scan)):
    if fn.endswith("_fastp.json"):
        digests.append(digest(os.path.join(scan, fn)))
with open(os.path.join(outdir, fname + ".html"), "w") as f:
    f.write("<html>multiqc stub " + hashlib.sha256("".join(digests).encode()).hexdigest() + "</html>\\n")
os.makedirs(os.path.join(outdir, "multiqc_data"), exist_ok=True)
with open(os.path.join(outdir, "multiqc_data", "multiqc_fastp.txt"), "w") as f:
    f.write("\\n".join(digests) + "\\n")
'''

# salmon fake: emits the REALISTIC quant.sf template built by the test
# (GENCODE 8-field pipe Names covering every downstream resource gene set),
# scaled by a per-sample factor derived from the content digest of the
# cleaned reads (deterministic: same inputs -> same quant.sf on both sides,
# DIFFERENT between samples so the runall-default --adjust_eset zero-variance
# filter keeps genes — a 1-sample or duplicated-column matrix wipes them all,
# which is the upstream single-sample crash reproduced by the real-data run).
_FAKE_SALMON = _FAKE_PRELUDE + '''
argv = sys.argv[1:]
record(argv)
out = getflag(argv, "-o")
r1 = getflag(argv, "-1"); r2 = getflag(argv, "-2")
os.makedirs(out, exist_ok=True)
d = digest(r1, r2)
factor = 1.0 + (int(d[:8], 16) % 500) / 1000.0
with open(os.environ["RUNALL_QUANT_TEMPLATE"]) as fi, \\
        open(os.path.join(out, "quant.sf"), "w") as fo:
    fo.write(fi.readline())
    for line in fi:
        parts = line.rstrip("\\n").split("\\t")
        parts[3] = "%.3f" % (float(parts[3]) * factor)
        parts[4] = "%.1f" % (float(parts[4]) * factor)
        fo.write("\\t".join(parts) + "\\n")
os.makedirs(os.path.join(out, "logs"), exist_ok=True)
with open(os.path.join(out, "logs", "salmon_quant.log"), "w") as f:
    f.write("fake salmon stub\\ndigest: " + d + "\\n")
'''

_FAKE_TRUST4 = _FAKE_PRELUDE + '''
argv = sys.argv[1:]
record(argv)
bam = getflag(argv, "-b"); r1 = getflag(argv, "-1"); r2 = getflag(argv, "-2")
ru = getflag(argv, "-u")
prefix = getflag(argv, "-o", "TRUST_out")
od = getflag(argv, "--od")
srcs = [p for p in (bam, r1, r2, ru) if p]
d = digest(*srcs) if srcs else hashlib.sha256(b"nodir").hexdigest()
base_dir = od if od else (os.path.dirname(prefix) or ".")
stem = os.path.join(base_dir, os.path.basename(prefix))
os.makedirs(base_dir, exist_ok=True)
with open(stem + "_report.tsv", "w") as f:
    f.write("#count\\tfrequency\\tCDR3nt\\tCDR3aa\\tV\\tD\\tJ\\tC\\tcid\\tcid_full_length\\n")
    for i in range(4):
        c = (int(d[i * 4:(i + 1) * 4], 16) % 50) + (0 if i else 1)
        f.write("%d\\t0.%02d\\tTGTGCA%03d\\tCAS%d\\tTRBV%d\\t*\\tTRBJ%d\\tTRBC%d\\tcid%d\\tcid%d\\n" % (
            c, c % 97, i * 7 + 1, i, (i % 5) + 1, (i % 3) + 1, (i % 2) + 1, i, i))
with open(stem + "_cdr3.out", "w") as f:
    f.write("assemble1\\t0\\tTRBV1\\t*\\tTRBJ1\\tTRBC1\\t*\\tTGTGCA\\tCAS\\t0.00\\t1.00\\t0.00\\t0\\n")
with open(stem + "_airr.tsv", "w") as f:
    f.write("sequence_id\\tsequence\\tproductive\\tlocus\\tv_call\\tj_call\\tc_call\\n")
    f.write("cid0\\tTGTGCA001\\tT\\tTRB\\tTRBV1\\tTRBJ1\\tTRBC1\\n")
'''


def _write_fake(dirpath, name, body):
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, name)
    with open(p, "w") as f:
        f.write(f"#!{sys.executable}\n" + body)
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _gene_universe():
    """Union of every gene symbol the downstream resource files need."""
    from importlib.resources import files
    R = files("iobrpy.resources")
    genes = set()
    lm22 = pd.read_csv(str(R.joinpath("lm22.txt")), sep=r"\s+", index_col=0)
    genes |= set(map(str, lm22.index))
    ep = pd.read_pickle(str(R.joinpath("epic_TRef_BRef.pkl")))
    genes |= set(map(str, ep["TRef"]["sigGenes"]))
    qd = pd.read_pickle(str(R.joinpath("quantiseq_data.pkl")))
    genes |= set(map(str, qd["TIL10_signature"].index))
    md = pd.read_pickle(str(R.joinpath("mcp_data.pkl")))
    genes |= set(map(str, md["genes"]["HUGO symbols"].dropna()))
    ed = pd.read_pickle(str(R.joinpath("estimate_data.pkl")))
    for row in ed["SI_geneset"].itertuples(index=False):
        for v in row:
            s = str(v).strip()
            if s and s not in ("estimate", "nan"):
                genes.add(s)
    with open(str(R.joinpath("IPS_genes.txt"))) as f:
        lines = f.read().strip().split("\n")
    for ln in lines[1:]:
        g = ln.split("\t")[0].strip()
        if g:
            genes.add(g)
    cd = pd.read_pickle(str(R.joinpath("calculate_data.pkl")))
    for v in cd["signature_tme"].values():
        genes |= set(map(str, v))
    ld = pd.read_pickle(str(R.joinpath("lr_data.pkl")))
    net = ld["intercell_networks"]["pancan"]
    genes |= set(map(str, net["ligands"])) | set(map(str, net["receptors"]))
    genes = {g for g in genes
             if g and "|" not in g and " " not in g and "." not in g
             and g != "nan" and not g.startswith("-")}
    return sorted(genes)


def _quant_template(path, genes, seed=11):
    rnd = random.Random(seed)
    with open(path, "w") as f:
        f.write("Name\tLength\tEffectiveLength\tTPM\tNumReads\n")
        for i, g in enumerate(genes):
            tpm = round(rnd.lognormvariate(0.0, 1.5), 3)
            nr = round(tpm * 37.0, 0)
            name = (f"ENST{i:011d}.1|ENSG{i:011d}.1|OTTHUMG{i:07d}|"
                    f"OTTHUMT{i:07d}|{g}-IT|{g}|{999 + i % 3000}|protein_coding")
            f.write(f"{name}\t{1000 + i % 4000}\t{950 + i % 3900}.0\t{tpm}\t{nr}\n")


def _mini_fastq(path, n_reads=8, seed_byte=b"A"):
    lines = []
    for i in range(n_reads):
        lines.append(f"@READ{i} len={n_reads}")
        lines.append((seed_byte.decode() * 4 + "ACGT") * 6)
        lines.append("+")
        lines.append("I" * 24)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(("\n".join(lines) + "\n").encode())


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_manifest(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, root)] = _sha(p)
    return out


def _gz_bytes(p):
    with gzip.open(p, "rb") as f:
        return f.read()


def _norm_tok(tok, roots):
    for r in roots:
        tok = tok.replace(r, "@ROOT@")
    tok = re.sub(r"/tmp/iobrpy_trust4_[^/\s]+", "@REFTMP@", tok)
    return tok


def _aligned_frames_equal(pa, pb, sep=None, drop_pvalue=False, ulp=False):
    """Permutation-invariant CSV comparison: same shape, same header multiset,
    same first-column key multiset, values exactly equal per (key, header)
    (or, with ``ulp=True``, within rtol=1e-9/atol=1e-12 — the documented
    class where the ORIGINAL merge_salmon ``as_completed`` column-order draw
    propagates into the joint-SVD PCA legs of calculate_sig_score at ULP
    level, reproducible between two ORIGINAL runs; the port at IDENTICAL
    column order is exact). Used where that documented non-determinism
    propagates into downstream sample order."""
    a = pd.read_csv(pa, sep=sep)
    b = pd.read_csv(pb, sep=sep)
    if drop_pvalue:
        a = a.drop(columns=[c for c in a.columns if str(c).startswith("P-value")])
        b = b.drop(columns=[c for c in b.columns if str(c).startswith("P-value")])
    if a.shape != b.shape or sorted(map(str, a.columns)) != sorted(map(str, b.columns)):
        return False
    a = a.set_index(a.columns[0])
    b = b.set_index(b.columns[0])
    a = a.sort_index().sort_index(axis=1)
    b = b.sort_index().sort_index(axis=1)
    if not ulp:
        return a.equals(b)
    import numpy as _np
    return bool(_np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                             rtol=1e-9, atol=1e-12, equal_nan=True))


@pytest.mark.full
def test_full_chain_mini_salmon(tmp_path, monkeypatch):
    """Original runall (real iobrpy child CLIs + fake tools) vs iobrx.runall
    (in-process ported substeps + the same fake tools) over TWO miniature
    samples (the runall-default --adjust_eset wipes all genes on a single
    sample — the upstream crash the real-data 1-sample run reproduces — so
    the completed-chain gate needs >=2 non-identical columns): trees
    byte-identical modulo the documented exceptions."""
    venv_bin = os.path.dirname(sys.executable)
    if (shutil.which("iobrpy") is None
            and not os.path.exists(os.path.join(venv_bin, "iobrpy"))):
        pytest.skip("original side needs the iobrpy console script on PATH")
    port = _ra()
    orig = _orig_runall()
    iobrx = _iobrx()

    # --- fake tools -------------------------------------------------------
    bindir = tmp_path / "fakebin"
    _write_fake(str(bindir), "fastp", _FAKE_FASTP)
    _write_fake(str(bindir), "multiqc", _FAKE_MULTIQC)
    _write_fake(str(bindir), "salmon", _FAKE_SALMON)
    _write_fake(str(bindir), "run-trust4", _FAKE_TRUST4)
    # keep the venv bin (iobrpy console script) reachable for the child CLIs
    monkeypatch.setenv(
        "PATH", f"{bindir}{os.pathsep}{venv_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("TRUST4_BIN", raising=False)

    # --- miniature inputs (2 samples) ---------------------------------------
    fq = tmp_path / "fastq"
    fq.mkdir()
    _mini_fastq(str(fq / "SRR_MINI1_1.fastq.gz"), 8, b"A")
    _mini_fastq(str(fq / "SRR_MINI1_2.fastq.gz"), 8, b"C")
    _mini_fastq(str(fq / "SRR_MINI2_1.fastq.gz"), 8, b"G")
    _mini_fastq(str(fq / "SRR_MINI2_2.fastq.gz"), 8, b"T")
    idx = tmp_path / "salmon_idx"
    idx.mkdir()
    template = tmp_path / "quant_template.sf"
    _quant_template(str(template), _gene_universe())
    monkeypatch.setenv("RUNALL_QUANT_TEMPLATE", str(template))

    o_log = tmp_path / "logs" / "orig.jsonl"
    p_log = tmp_path / "logs" / "port.jsonl"
    o_log.parent.mkdir(exist_ok=True)
    o_log.write_text("")
    p_log.write_text("")

    out_o = tmp_path / "orig_out"
    out_p = tmp_path / "port_out"
    extra = ["--index", str(idx), "--perm", "10",
             "--signature", "signature_tme", "--method", "integration"]

    # --- original side (spawns iobrpy child CLIs) ---------------------------
    monkeypatch.setenv("FAKE_CALL_LOG", str(o_log))
    argv = ["--mode", "salmon", "--outdir", str(out_o), "--fastq", str(fq),
            "--threads", "4", "--batch_size", "2"] + extra
    try:
        orig.main(list(argv))
        rc_o = 0
    except SystemExit as e:
        rc_o = e.code if isinstance(e.code, int) else 1
    assert rc_o == 0, "original chain failed"

    # --- port side (in-process) ---------------------------------------------
    monkeypatch.setenv("FAKE_CALL_LOG", str(p_log))
    res = iobrx.runall("salmon", out_p, fq, threads=4, batch_size=2,
                       unknown=list(extra))
    assert res == {"rc": 0}, "ported chain failed"

    # --- external-tool command lines token-identical ------------------------
    def calls(lp, out_root):
        with open(lp) as f:
            rows = [json.loads(x) for x in f if x.strip()]
        norm = []
        for c in rows:
            norm.append((c["bin"],
                         tuple(_norm_tok(a, [str(out_root), str(fq), str(idx),
                                             str(bindir), str(template)])
                               for a in c["argv"])))
        return sorted(norm)

    assert calls(o_log, out_o) == calls(p_log, out_p)
    assert len(calls(o_log, out_o)) >= 6   # 2x fastp + multiqc + 2x salmon + trust4

    # --- output trees --------------------------------------------------------
    mo, mp = _tree_manifest(str(out_o)), _tree_manifest(str(out_p))
    assert set(mo) == set(mp), (
        f"file sets differ:\n only-orig: {sorted(set(mo) - set(mp))}\n"
        f" only-port: {sorted(set(mp) - set(mo))}")

    pvalue_files = {"05-tme/cibersort_results.csv", "05-tme/deconvo_merged.csv"}
    byte_diff, gz_aligned, csv_aligned, pv_ok, sig_ulp = [], [], [], [], []
    for rel in sorted(mo):
        po, pp = os.path.join(str(out_o), rel), os.path.join(str(out_p), rel)
        if mo[rel] == mp[rel]:
            continue
        if rel.endswith(".tsv.gz"):
            # gzip mtime + merge_salmon as_completed column order (documented
            # non-determinism) -> decompressed, column-aligned comparison
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tsv") as ta, \
                    tempfile.NamedTemporaryFile(suffix=".tsv") as tb:
                ta.write(_gz_bytes(po)); tb.write(_gz_bytes(pp))
                ta.flush(); tb.flush()
                ok = _aligned_frames_equal(ta.name, tb.name, sep="\t")
            (gz_aligned if ok else byte_diff).append(rel)
            continue
        if rel.endswith(".csv"):
            if rel in pvalue_files:
                ok = _aligned_frames_equal(po, pp, drop_pvalue=True)
                pv_ok.append((rel, ok))
                if not ok:
                    byte_diff.append(rel)
                continue
            if _aligned_frames_equal(po, pp):
                csv_aligned.append(rel)
            elif (rel == "04-signatures/calculate_sig_score.csv"
                  and _aligned_frames_equal(po, pp, ulp=True)):
                sig_ulp.append(rel)
            else:
                byte_diff.append(rel)
            continue
        byte_diff.append(rel)
    assert not byte_diff, f"unexpected differences: {byte_diff}"
    assert all(ok for _, ok in pv_ok), \
        "cibersort P-value files must match excluding P-value"
    assert "05-tme/cibersort_results.csv" in mo
    # NOTE: pv_ok may legitimately be EMPTY — with --perm 10 the unseeded
    # original P-value draw is discrete ((hits+1)/(perm+1)-style) and often
    # coincides with the ported seeded draw, making the files byte-identical
    # (caught by the mo[rel]==mp[rel] short-circuit above). The R3 contract
    # only guarantees equality EXCLUDING the P-value column; byte-equality
    # is a bonus, not a requirement, so it must not be asserted to differ.
    # the sample-column permutation is EXPECTED to hit the merged matrices
    # (original as_completed vs ported sorted order); everything downstream
    # of prepare_salmon is alphabetical-row-order and usually byte-identical.

    # --- key products carry the CLI-layer shapes -----------------------------
    tpm = pd.read_csv(out_p / "03-tpm" / "tpm_matrix.csv", index_col=0)
    assert tpm.shape[1] == 2
    cb = pd.read_csv(out_p / "05-tme" / "cibersort_results.csv", index_col=0)
    assert cb.index.name == "ID"
    assert all(c.endswith("_CIBERSORT") for c in cb.columns)
    est = pd.read_csv(out_p / "05-tme" / "estimate_results.csv", index_col=0)
    assert est.index.name == "ID"
    assert all(c.endswith("_estimate") for c in est.columns)
    mcp = pd.read_csv(out_p / "05-tme" / "mcpcounter_results.csv", index_col=0)
    assert all(c.endswith("_MCPcounter") for c in mcp.columns)
    qs = pd.read_csv(out_p / "05-tme" / "quantiseq_results.csv")
    assert qs.columns[0] == "ID" and qs.shape[0] == 2
    assert any(c.endswith("_quantiseq") for c in qs.columns)
    epc = pd.read_csv(out_p / "05-tme" / "epic_results.csv", index_col=0)
    assert all(c.endswith("_EPIC") for c in epc.columns)
    lr = pd.read_csv(out_p / "06-LR_cal" / "lr_cal.csv")
    assert lr.columns[0] == "ID" and lr.shape[0] == 2
    sig = pd.read_csv(out_p / "04-signatures" / "calculate_sig_score.csv")
    assert sig.shape[0] == 2 and sig.shape[1] > 100
    assert (out_p / "07-TCRBCR" / "trust4_immdata.csv").exists()
    assert (out_p / "07-TCRBCR" / ".trust4.done").exists()

    # --- resume over the finished tree skips everything (both sides) ---------
    n_calls_before = len(calls(p_log, out_p))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res2 = iobrx.runall("salmon", out_p, fq, threads=4, batch_size=2,
                            resume=True, unknown=list(extra))
    assert res2 == {"rc": 0}
    out_r = buf.getvalue()
    assert "[resume] fastq_qc skipped" in out_r
    assert "[resume] trust4 skipped" in out_r
    assert "[run]" not in out_r
    assert len(calls(p_log, out_p)) == n_calls_before   # no tool re-invoked

    n_calls_o_before = len(calls(o_log, out_o))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            orig.main(["--mode", "salmon", "--outdir", str(out_o),
                       "--fastq", str(fq), "--threads", "4", "--batch_size", "2",
                       "--resume"] + extra)
            rc_o2 = 0
        except SystemExit as e:
            rc_o2 = e.code if isinstance(e.code, int) else 1
    assert rc_o2 == 0
    assert buf.getvalue() == out_r          # resume console byte-identical
    assert len(calls(o_log, out_o)) == n_calls_o_before   # no tool re-invoked
