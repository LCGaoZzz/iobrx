"""Orchestration parity gates for the four external-tool stages
(fastq_qc / batch_salmon / batch_star_count / trust4).

Synthetic-miniature contract tests: the REAL fastp / salmon / STAR /
run-trust4 / multiqc binaries are replaced by deterministic fake scripts
(same names, prepended to PATH) that RECORD their exact argv to a JSONL call
log and emit content-addressed stub outputs (sha256 of the input bytes ->
fixed file content, gzip mtime=0). The ORIGINAL ``iobrpy.workflow`` modules
and the ported ``iobrx._fast`` modules are then run on identical miniature
inputs into separate trees and compared:

* output trees BYTE-identical (every regular file, sha256; same file set) —
  cleaned "fastq" stubs, fastp json/html stubs, ``*.task.complete`` markers,
  ``quant.sf`` stubs, ``task.complete`` (``ok\\n``), BAM/GeneCounts stubs,
  TRUST4 report/cdr3 stubs, per-sample done flags, staged post-processing
  outputs ``trust4_immdata.csv`` / ``trust4_immune_indices.csv``;
* recorded tool command lines TOKEN-identical after root/temp normalization
  (Pool stages compared as multisets — dispatch order is ``random.shuffle``d
  in BOTH implementations; sequential stages as sequences);
* resume/skip logic (re-run adds no tool calls), failure paths (salmon rc=1
  + stderr summary vs SystemExit(1); STAR RuntimeError propagation; trust4
  runner-missing 127 / p.error 2), suffix-inference helpers, binary-path
  overrides, and the ``backend='python'`` argv-rewrite escape hatches;
* the accelerated immune post-processing is bit-identical to the ORIGINAL
  ``process_immune_data_batch`` on seeded multi-sample synthetic reports
  (zero-count clones, ties, single-clone and header-only degenerate samples)
  — CSV bytes AND returned frames.

The heavy real-tool parity (official FASTQ/BAM data) runs outside pytest via
``research/build_orchestration/scripts`` — see the port's INTEGRATION.md.

Run with the repo source on the path::

    PYTHONPATH=<repo>/src pytest -q tests/test_parity_orchestration.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# module handles
# ---------------------------------------------------------------------------
def _fq():
    return pytest.importorskip(
        "iobrx._fast.fastq_qc_fast",
        reason="needs PYTHONPATH=<repo>/src (or an installed iobrx)",
    )


def _bs():
    return pytest.importorskip("iobrx._fast.batch_salmon_fast")


def _bst():
    return pytest.importorskip("iobrx._fast.batch_star_count_fast")


def _t4():
    return pytest.importorskip("iobrx._fast.trust4_fast")


def _orig_fastq_qc():
    return pytest.importorskip("iobrpy.workflow.fastq_qc")


def _orig_batch_salmon():
    return pytest.importorskip("iobrpy.workflow.batch_salmon")


def _orig_batch_star():
    return pytest.importorskip("iobrpy.workflow.batch_star_count")


def _orig_trust4():
    return pytest.importorskip("iobrpy.workflow.trust4")


# ---------------------------------------------------------------------------
# fake external tools (deterministic, argv-recording)
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

_FAKE_SALMON = _FAKE_PRELUDE + '''
argv = sys.argv[1:]
record(argv)
fail = [s for s in os.environ.get("FAIL_SAMPLES", "").split(",") if s]
out = getflag(argv, "-o")
r1 = getflag(argv, "-1"); r2 = getflag(argv, "-2")
sid = os.path.basename(out)
if sid in fail:
    sys.stderr.write("fake salmon: forced failure for " + sid + "\\n")
    sys.exit(1)
d = digest(r1, r2)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "quant.sf"), "w") as f:
    f.write("Name\\tLength\\tEffectiveLength\\tTPM\\tNumReads\\n")
    for i in range(10):
        f.write("TX%d\\t%d\\t%d.0\\t%d.0\\t%d.0\\n" % (
            i, 500 + i, 450 + i, (int(d[:8], 16) + i) % 1000, (int(d[8:16], 16) + i) % 977))
os.makedirs(os.path.join(out, "logs"), exist_ok=True)
with open(os.path.join(out, "logs", "salmon_quant.log"), "w") as f:
    f.write("fake salmon stub\\ndigest: " + d + "\\n")
'''

_FAKE_STAR = _FAKE_PRELUDE + '''
argv = sys.argv[1:]
record(argv)
fail = [s for s in os.environ.get("FAIL_SAMPLES", "").split(",") if s]
prefix = getflag(argv, "--outFileNamePrefix")
reads = getflag(argv, "--readFilesIn")
f1 = reads
import itertools
i = argv.index("--readFilesIn")
f1 = argv[i + 1]; f2 = argv[i + 2]
sid = os.path.basename(prefix).rstrip("_")
if sid in fail:
    sys.stderr.write("fake STAR: forced failure for " + sid + "\\n")
    sys.exit(3)
d = digest(f1, f2)
gz_out(prefix + "Aligned.sortedByCoord.out.bam", b"BAMSTUB:" + d.encode())
with open(prefix + "ReadsPerGene.out.tab", "w") as f:
    f.write("N_unmapped\\t1\\t0\\t0\\nN_multimapping\\t2\\t0\\t0\\n")
    f.write("N_noFeature\\t3\\t0\\t0\\nN_ambiguous\\t4\\t0\\t0\\n")
    for g in range(5):
        f.write("GENE%d\\t%d\\t%d\\t%d\\t%d\\n" % (g, (int(d[:6], 16) + g) % 999, 0, (int(d[6:12], 16) + g) % 999, 0))
with open(prefix + "Log.out", "w") as f:
    f.write("fake STAR stub " + d + "\\n")
'''

_FAKE_TRUST4 = _FAKE_PRELUDE + '''
argv = sys.argv[1:]
record(argv)
fail = [s for s in os.environ.get("FAIL_SAMPLES", "").split(",") if s]
bam = getflag(argv, "-b"); r1 = getflag(argv, "-1"); r2 = getflag(argv, "-2")
ru = getflag(argv, "-u")
prefix = getflag(argv, "-o", "TRUST_out")
od = getflag(argv, "--od")
srcs = [p for p in (bam, r1, r2, ru) if p]
d = digest(*srcs)
base_dir = od if od else (os.path.dirname(prefix) or ".")
stem = os.path.join(base_dir, os.path.basename(prefix))
os.makedirs(base_dir, exist_ok=True)
tag = os.path.basename(stem)
if tag in fail:
    sys.stderr.write("fake trust4: forced failure\\n")
    sys.exit(4)
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


def _write_fake(dirpath, name, body, tag=None):
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, name)
    with open(p, "w") as f:
        f.write(f"#!{sys.executable}\n" + body)
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


@pytest.fixture()
def fakebins(tmp_path, monkeypatch):
    """Prepend a directory of fake tools to PATH; return the dir."""
    bindir = tmp_path / "fakebin"
    _write_fake(str(bindir), "fastp", _FAKE_FASTP)
    _write_fake(str(bindir), "multiqc", _FAKE_MULTIQC)
    _write_fake(str(bindir), "salmon", _FAKE_SALMON)
    _write_fake(str(bindir), "STAR", _FAKE_STAR)
    _write_fake(str(bindir), "run-trust4", _FAKE_TRUST4)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("TRUST4_BIN", raising=False)
    monkeypatch.delenv("FAIL_SAMPLES", raising=False)
    return bindir


def _set_calllog(monkeypatch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    monkeypatch.setenv("FAKE_CALL_LOG", path)


def _calls(logpath):
    with open(logpath) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _norm_tok(tok, roots):
    for r in roots:
        tok = tok.replace(r, "@ROOT@")
    tok = re.sub(r"/tmp/iobrpy_trust4_[^/\s]+", "@REFTMP@", tok)
    return tok


def _norm_calls(calls, roots):
    return [
        [c["bin"], tuple(_norm_tok(a, roots) for a in c["argv"])]
        for c in calls
    ]


# ---------------------------------------------------------------------------
# miniature inputs
# ---------------------------------------------------------------------------
def _mini_fastq(path, n_reads=6, seed_byte=b"A"):
    lines = []
    for i in range(n_reads):
        lines.append(f"@READ{i} len={n_reads}")
        lines.append((seed_byte.decode() * 4 + "ACGT") * 5)
        lines.append("+")
        lines.append("IIIIIIIIIIIIIIIIIIIIIIII")
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(("\n".join(lines) + "\n").encode())


def _make_pe_tree(root, samples, n_reads=6):
    """Write <sample>_1.fastq.gz / <sample>_2.fastq.gz for each sample."""
    os.makedirs(root, exist_ok=True)
    for j, s in enumerate(samples):
        _mini_fastq(os.path.join(root, f"{s}_1.fastq.gz"), n_reads, b"ACGT"[j % 4:j % 4 + 1])
        _mini_fastq(os.path.join(root, f"{s}_2.fastq.gz"), n_reads, b"TGG"[j % 3:j % 3 + 1])


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
            rel = os.path.relpath(p, root)
            # Reliability metadata is intentionally additional to upstream outputs.
            if fn.endswith(".iobrx.json") or rel.replace(os.sep, "/") == "multiqc_report/task.complete":
                continue
            out[rel] = _sha(p)
    return out


def _assert_trees_equal(a, b):
    ma, mb = _tree_manifest(a), _tree_manifest(b)
    assert set(ma) == set(mb), (
        f"file sets differ:\n only-orig: {sorted(set(ma) - set(mb))}\n "
        f"only-port: {sorted(set(mb) - set(ma))}"
    )
    diff = [k for k in ma if ma[k] != mb[k]]
    assert not diff, f"byte content differs for: {sorted(diff)}"


# ---------------------------------------------------------------------------
# fastq_qc
# ---------------------------------------------------------------------------
def test_fastq_qc_pe_parity_and_resume(tmp_path, monkeypatch, fakebins):
    fq = _fq()
    orig = _orig_fastq_qc()

    in_dir = tmp_path / "in"
    _make_pe_tree(str(in_dir), ["SRR_A", "SRR_B"])

    o_out = tmp_path / "orig_out"
    p_out = tmp_path / "port_out"
    o_log = tmp_path / "logs" / "orig.jsonl"
    p_log = tmp_path / "logs" / "port.jsonl"

    # --- original (workflow-level main via argv rewrite) ---
    _set_calllog(monkeypatch, str(o_log))
    monkeypatch.setattr(sys, "argv", [
        "fastq_qc", "--path1_fastq", str(in_dir), "--path2_fastp", str(o_out),
        "--num_threads", "4", "--batch_size", "2", "--length_required", "50"])
    orig.main()

    # --- port ---
    _set_calllog(monkeypatch, str(p_log))
    res = fq.fastq_qc(str(in_dir), str(p_out), num_threads=4, batch_size=2,
                      length_required=50)
    assert set(res) == {"rc", "results", "outputs", "multiqc_report"}
    assert res["rc"] == 0
    assert sorted(r["sample"] for r in res["results"]) == ["SRR_A", "SRR_B"]
    assert all(r["status"] == "processed" for r in res["results"])
    assert res["multiqc_report"] == str(p_out / "multiqc_report" / "multiqc_fastp_report.html")

    _assert_trees_equal(str(o_out), str(p_out))

    # command lines token-identical (multiset — shuffle/imap_unordered order)
    o_calls = _norm_calls(_calls(str(o_log)), [str(in_dir), str(o_out), str(fakebins)])
    p_calls = _norm_calls(_calls(str(p_log)), [str(in_dir), str(p_out), str(fakebins)])
    # normalize the output-root token difference
    o_calls = [[b, tuple(t.replace("@ROOT@", "@X@") for t in a)] for b, a in o_calls]
    p_calls = [[b, tuple(t.replace("@ROOT@", "@X@") for t in a)] for b, a in p_calls]
    assert sorted(map(str, o_calls)) == sorted(map(str, p_calls))
    fastp_calls = [a for b, a in p_calls if b.endswith("fastp") or "fastp" in str(b)]
    assert len(_calls(str(p_log))) == 3  # 2 fastp + 1 multiqc

    # --- resume: re-run the port, no new tool calls, tree unchanged ---
    before = _tree_manifest(str(p_out))
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "port2.jsonl"))
    res2 = fq.fastq_qc(str(in_dir), str(p_out), num_threads=4, batch_size=2)
    assert _calls(str(tmp_path / "logs" / "port2.jsonl")) == []
    assert all(r["status"] == "skipped" for r in res2["results"])
    assert _tree_manifest(str(p_out)) == before


def test_fastq_qc_se_mode_and_binary_override(tmp_path, monkeypatch, fakebins):
    fq = _fq()
    orig = _orig_fastq_qc()

    in_dir = tmp_path / "in_se"
    os.makedirs(in_dir)
    _mini_fastq(str(in_dir / "S1_1.fastq.gz"))

    o_out = tmp_path / "se_orig"
    p_out = tmp_path / "se_port"
    o_log = tmp_path / "logs" / "se_orig.jsonl"
    p_log = tmp_path / "logs" / "se_port.jsonl"

    _set_calllog(monkeypatch, str(o_log))
    monkeypatch.setattr(sys, "argv", [
        "fastq_qc", "--path1_fastq", str(in_dir), "--path2_fastp", str(o_out),
        "--num_threads", "2", "--batch_size", "1", "--se"])
    orig.main()

    # port with explicit binary override (same fake script under a copy name)
    alt = _write_fake(str(tmp_path / "altbin"), "fastp-alt", _FAKE_FASTP)
    _set_calllog(monkeypatch, str(p_log))
    res = fq.fastq_qc(str(in_dir), str(p_out), num_threads=2, batch_size=1,
                      se=True, fastp_bin=alt,
                      multiqc_bin=str(fakebins / "multiqc"))
    _assert_trees_equal(str(o_out), str(p_out))
    calls = _calls(str(p_log))
    assert calls[0]["bin"] == alt  # override reached the spawned binary
    # SE command shape: -i/-o only, no -I/-O
    assert "-I" not in calls[0]["argv"] and "-O" not in calls[0]["argv"]
    assert res["results"][0]["status"] == "processed"


# ---------------------------------------------------------------------------
# batch_salmon
# ---------------------------------------------------------------------------
def _run_orig_salmon(monkeypatch, orig, index, path_fq, path_out, **kw):
    argv = ["batch_salmon", "--index", str(index), "--path_fq", str(path_fq),
            "--path_out", str(path_out),
            "--suffix1", kw.get("suffix1", "_1.fastq.gz"),
            "--batch_size", str(kw.get("batch_size", 1)),
            "--num_threads", str(kw.get("num_threads", 4))]
    if kw.get("gtf"):
        argv += ["--gtf", str(kw["gtf"])]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        orig.main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


def test_batch_salmon_parity_happy_and_failures(tmp_path, monkeypatch, fakebins):
    bs = _bs()
    orig = _orig_batch_salmon()

    index = tmp_path / "sidx"
    os.makedirs(index)
    (index / "meta_info.json").write_text('{"indexVersion": "stub"}')

    in_dir = tmp_path / "fq"
    _make_pe_tree(str(in_dir), ["SAL_X", "SAL_Y"])

    o_out = tmp_path / "sal_orig"
    p_out = tmp_path / "sal_port"
    o_log = tmp_path / "logs" / "sal_orig.jsonl"
    p_log = tmp_path / "logs" / "sal_port.jsonl"

    _set_calllog(monkeypatch, str(o_log))
    rc_o = _run_orig_salmon(monkeypatch, orig, index, in_dir, o_out,
                            batch_size=2, num_threads=4)
    assert rc_o == 0

    _set_calllog(monkeypatch, str(p_log))
    res = bs.batch_salmon(str(index), str(in_dir), str(p_out),
                          batch_size=2, num_threads=4)
    assert res["rc"] == 0 and res["n_pairs"] == 2 and not res["failures"]

    _assert_trees_equal(str(o_out), str(p_out))
    # quant.sf + task.complete("ok\n") written by orchestrator/tool
    assert (p_out / "SAL_X" / "quant.sf").exists()
    assert (p_out / "SAL_X" / "task.complete").read_text() == "ok\n"

    o_calls = _norm_calls(_calls(str(o_log)), [str(in_dir), str(o_out), str(index), str(fakebins)])
    p_calls = _norm_calls(_calls(str(p_log)), [str(in_dir), str(p_out), str(index), str(fakebins)])
    o_calls = sorted(str([b, tuple(t.replace("@ROOT@", "@X@") for t in a)]) for b, a in o_calls)
    p_calls = sorted(str([b, tuple(t.replace("@ROOT@", "@X@") for t in a)]) for b, a in p_calls)
    assert o_calls == p_calls

    # resume: re-run port -> no salmon calls
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "sal_port2.jsonl"))
    res2 = bs.batch_salmon(str(index), str(in_dir), str(p_out),
                           batch_size=2, num_threads=4)
    assert res2["rc"] == 0
    assert _calls(str(tmp_path / "logs" / "sal_port2.jsonl")) == []

    # failure path: fake salmon fails for SAL_Y -> original SystemExit(1),
    # port rc=1 with the friendly guidance message
    f_out = tmp_path / "sal_fail"
    f_log = tmp_path / "logs" / "sal_fail.jsonl"
    _set_calllog(monkeypatch, str(f_log))
    monkeypatch.setenv("FAIL_SAMPLES", "SAL_Y")
    rc_o = _run_orig_salmon(monkeypatch, orig, index, in_dir, f_out, num_threads=2)
    assert rc_o == 1
    res3 = bs.batch_salmon(str(index), str(in_dir), str(f_out), num_threads=2)
    assert res3["rc"] == 1
    assert [s for s, _ in res3["failures"]] == ["SAL_Y"]
    assert "salmon quant failed" in res3["failures"][0][1]
    monkeypatch.delenv("FAIL_SAMPLES")


def test_batch_salmon_discovery_edges(tmp_path, monkeypatch, fakebins, capsys):
    bs = _bs()
    orig = _orig_batch_salmon()
    index = tmp_path / "sidx2"
    os.makedirs(index)

    # missing R2 -> warn + skip (both implementations), rc 0 with 1 pair
    d = tmp_path / "fq_missing_r2"
    _make_pe_tree(str(d), ["P1"])
    _mini_fastq(str(d / "P2_1.fastq.gz"))  # P2 has no R2
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "edge1.jsonl"))
    res = bs.batch_salmon(str(index), str(d), str(tmp_path / "o_edge1"))
    err = capsys.readouterr().err
    assert "Missing R2 for P2" in err
    assert res["rc"] == 0 and res["n_pairs"] == 1

    # no FASTQs at all -> rc 2 (original: SystemExit(2))
    empty = tmp_path / "fq_empty"
    os.makedirs(empty)
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "edge2.jsonl"))
    res = bs.batch_salmon(str(index), str(empty), str(tmp_path / "o_edge2"))
    assert res["rc"] == 2
    monkeypatch.setattr(sys, "argv", [
        "batch_salmon", "--index", str(index), "--path_fq", str(empty),
        "--path_out", str(tmp_path / "o_edge2b")])
    with pytest.raises(SystemExit) as e:
        orig.main()
    assert e.value.code == 2

    # suffix inference: port helper == upstream helper, incl. the error case
    up = orig._infer_suffix2_from_suffix1
    mine = bs._infer_suffix2_from_suffix1
    for s in ["_1.fastq.gz", "R1.fq.gz", "_1.fq", ".1.fastq.gz", "x-1.fq.gz"]:
        assert mine(s) == up(s), s
    for bad in ["_X.fastq.gz"]:
        with pytest.raises(ValueError):
            up(bad)
        with pytest.raises(ValueError):
            mine(bad)
    res = bs.batch_salmon(str(index), str(empty), str(tmp_path / "o_edge3"),
                          suffix1="_X.fastq.gz")
    assert res["rc"] == 2  # upstream prints to stderr + exit(2)


def test_batch_salmon_original_escape_hatch(tmp_path, monkeypatch, fakebins):
    bs = _bs()
    index = tmp_path / "sidx3"
    os.makedirs(index)
    in_dir = tmp_path / "fq3"
    _make_pe_tree(str(in_dir), ["ESC1"])
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "esc.jsonl"))
    out = bs.batch_salmon_original(str(index), str(in_dir),
                                   str(tmp_path / "esc_out"), num_threads=2)
    assert out["rc"] == 0
    assert (tmp_path / "esc_out" / "ESC1" / "quant.sf").exists()


# ---------------------------------------------------------------------------
# batch_star_count
# ---------------------------------------------------------------------------
def test_batch_star_count_parity_and_failure(tmp_path, monkeypatch, fakebins):
    bst = _bst()
    orig = _orig_batch_star()

    index = tmp_path / "stidx"
    os.makedirs(index)
    in_dir = tmp_path / "fq"
    _make_pe_tree(str(in_dir), ["ST_1S", "ST_2S"])

    o_out = tmp_path / "st_orig"
    p_out = tmp_path / "st_port"
    o_log = tmp_path / "logs" / "st_orig.jsonl"
    p_log = tmp_path / "logs" / "st_port.jsonl"

    _set_calllog(monkeypatch, str(o_log))
    monkeypatch.setattr(sys, "argv", [
        "batch_star_count", "--index", str(index), "--path_fq", str(in_dir),
        "--path_out", str(o_out), "--batch_size", "1", "--num_threads", "3"])
    orig.main()

    _set_calllog(monkeypatch, str(p_log))
    res = bst.batch_star_count(str(index), str(in_dir), str(p_out),
                               batch_size=1, num_threads=3)
    assert res["rc"] == 0 and sorted(res["samples"]) == ["ST_1S", "ST_2S"]

    _assert_trees_equal(str(o_out), str(p_out))
    # orchestrator-side artifacts
    assert (p_out / "ST_1S.task.complete").read_bytes() == b""
    assert (p_out / "ST_1S").is_dir()  # upstream creates the (empty) sample dir
    assert (p_out / "ST_1S_Aligned.sortedByCoord.out.bam").exists()
    assert (p_out / "ST_1S_ReadsPerGene.out.tab").exists()

    o_calls = _norm_calls(_calls(str(o_log)), [str(in_dir), str(o_out), str(index), str(fakebins)])
    p_calls = _norm_calls(_calls(str(p_log)), [str(in_dir), str(p_out), str(index), str(fakebins)])
    o_calls = sorted(str([b, tuple(t.replace("@ROOT@", "@X@") for t in a)]) for b, a in o_calls)
    p_calls = sorted(str([b, tuple(t.replace("@ROOT@", "@X@") for t in a)]) for b, a in p_calls)
    assert o_calls == p_calls
    # STAR command tokens (spot-check the verbatim options)
    star_argv = _calls(str(p_log))[0]["argv"]
    for tok in ["--twopassMode", "Basic", "--readFilesCommand", "zcat",
                "--outSAMtype", "BAM", "SortedByCoordinate",
                "--quantMode", "GeneCounts", "--limitBAMsortRAM", "137438953472"]:
        assert tok in star_argv

    # resume: re-run -> no STAR calls
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "st_port2.jsonl"))
    bst.batch_star_count(str(index), str(in_dir), str(p_out), num_threads=3)
    assert _calls(str(tmp_path / "logs" / "st_port2.jsonl")) == []

    # failure: RuntimeError propagates (upstream semantics), message parity
    f_out = tmp_path / "st_fail"
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "st_fail.jsonl"))
    monkeypatch.setenv("FAIL_SAMPLES", "ST_2S")
    with pytest.raises(RuntimeError) as e_port:
        bst.batch_star_count(str(index), str(in_dir), str(f_out), num_threads=2)
    assert "STAR failed for sample ST_2S with exit code 3" in str(e_port.value)
    monkeypatch.setattr(sys, "argv", [
        "batch_star_count", "--index", str(index), "--path_fq", str(in_dir),
        "--path_out", str(tmp_path / "st_fail2"), "--num_threads", "2"])
    with pytest.raises(RuntimeError) as e_orig:
        orig.main()
    assert "STAR failed for sample ST_2S with exit code 3" in str(e_orig.value)
    monkeypatch.delenv("FAIL_SAMPLES")


# ---------------------------------------------------------------------------
# trust4
# ---------------------------------------------------------------------------
def _mini_bam(path, seed=b"B"):
    with open(path, "wb") as f:
        f.write(b"BAM\1" + seed * 64)


def test_trust4_single_bam_parity(tmp_path, monkeypatch, fakebins):
    t4 = _t4()
    orig = _orig_trust4()

    bam_dir = tmp_path / "bams"
    os.makedirs(bam_dir)
    bam = bam_dir / "SRR1_Aligned.sortedByCoord.out.bam"
    _mini_bam(str(bam))

    o_root = tmp_path / "t4_orig"
    p_root = tmp_path / "t4_port"
    os.makedirs(o_root)
    os.makedirs(p_root)
    o_log = tmp_path / "logs" / "t4_orig.jsonl"
    p_log = tmp_path / "logs" / "t4_port.jsonl"

    # original: -b single BAM, -o <root>/04-trust4 (prefix), -t 4
    _set_calllog(monkeypatch, str(o_log))
    with pytest.raises(SystemExit) as e:
        orig.main(["-b", str(bam), "-t", "4", "-o", str(o_root / "04-trust4")])
    assert e.value.code == 0

    # port: identical argv
    _set_calllog(monkeypatch, str(p_log))
    with pytest.raises(SystemExit) as e:
        t4.main(["-b", str(bam), "-t", "4", "-o", str(p_root / "04-trust4")])
    assert e.value.code == 0

    _assert_trees_equal(str(o_root), str(p_root))
    # post-processing artifacts (ORIGINAL processor vs accelerated processor)
    assert (p_root / "trust4_immdata.csv").exists()
    assert (p_root / "trust4_immune_indices.csv").exists()
    assert _sha(str(o_root / "trust4_immdata.csv")) == _sha(str(p_root / "trust4_immdata.csv"))
    assert _sha(str(o_root / "trust4_immune_indices.csv")) == _sha(str(p_root / "trust4_immune_indices.csv"))

    # runner command token parity (temp ref paths normalized)
    o_calls = _norm_calls(_calls(str(o_log)), [str(o_root), str(bam_dir), str(fakebins)])
    p_calls = _norm_calls(_calls(str(p_log)), [str(p_root), str(bam_dir), str(fakebins)])
    o_n = [[b, tuple(t.replace("@ROOT@", "@X@") for t in a)] for b, a in o_calls]
    p_n = [[b, tuple(t.replace("@ROOT@", "@X@") for t in a)] for b, a in p_calls]
    assert o_n == p_n  # sequential single call: exact sequence match
    argv = p_n[0][1]
    assert os.path.basename(p_n[0][0]) == "run-trust4"
    assert "-f" in argv and "--ref" in argv and "-t" in argv


def test_trust4_bamdir_batch_parity_resume_and_rc(tmp_path, monkeypatch, fakebins):
    t4 = _t4()
    orig = _orig_trust4()

    bam_dir = tmp_path / "bamdir"
    os.makedirs(bam_dir)
    for i, seed in enumerate([b"A", b"C"]):
        _mini_bam(str(bam_dir / f"S{i}_Aligned.sortedByCoord.out.bam"), seed)

    o_root = tmp_path / "tb_orig"
    p_root = tmp_path / "tb_port"
    o_log = tmp_path / "logs" / "tb_orig.jsonl"
    p_log = tmp_path / "logs" / "tb_port.jsonl"

    _set_calllog(monkeypatch, str(o_log))
    with pytest.raises(SystemExit) as e:
        orig.main(["-b", str(bam_dir), "-o", str(o_root), "-t", "2"])
    assert e.value.code == 0

    _set_calllog(monkeypatch, str(p_log))
    with pytest.raises(SystemExit) as e:
        t4.main(["-b", str(bam_dir), "-o", str(p_root), "-t", "2"])
    assert e.value.code == 0

    _assert_trees_equal(str(o_root), str(p_root))
    # per-sample dirs, TRUST_ prefixes, done flags
    for i in (0, 1):
        sd = p_root / f"S{i}"
        assert (sd / f"S{i}.TRUST4.done").read_text() == f"SUCCESS: TRUST4 finished for sample S{i}\n"
        assert (sd / f"TRUST_S{i}_Aligned_report.tsv").exists()
    # batch-level immune post-processing over subfolders
    imm = pd.read_csv(p_root / "trust4_immdata.csv")
    assert sorted(imm["Sample"].unique()) == ["S0", "S1"]
    assert _sha(str(o_root / "trust4_immune_indices.csv")) == _sha(str(p_root / "trust4_immune_indices.csv"))

    # resume: done flags -> zero new trust4 calls, identical tree
    before = _tree_manifest(str(p_root))
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "tb_port2.jsonl"))
    with pytest.raises(SystemExit) as e:
        t4.main(["-b", str(bam_dir), "-o", str(p_root), "-t", "2"])
    assert e.value.code == 0
    assert _calls(str(tmp_path / "logs" / "tb_port2.jsonl")) == []
    assert _tree_manifest(str(p_root)) == before

    # batch rc propagation: one sample fails -> exit code = failing rc,
    # successful sibling keeps its done flag; identical on both sides
    f_root_o = tmp_path / "tb_fail_orig"
    f_root_p = tmp_path / "tb_fail_port"
    monkeypatch.setenv("FAIL_SAMPLES", "TRUST_S1_Aligned")
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "tb_fo.jsonl"))
    with pytest.raises(SystemExit) as e:
        orig.main(["-b", str(bam_dir), "-o", str(f_root_o), "-t", "2"])
    rc_o = e.value.code
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "tb_fp.jsonl"))
    with pytest.raises(SystemExit) as e:
        t4.main(["-b", str(bam_dir), "-o", str(f_root_p), "-t", "2"])
    assert e.value.code == rc_o == 4
    assert (f_root_p / "S0" / "S0.TRUST4.done").exists()
    assert not (f_root_p / "S1" / "S1.TRUST4.done").exists()
    monkeypatch.delenv("FAIL_SAMPLES")


def test_trust4_fqdir_batch_parity(tmp_path, monkeypatch, fakebins):
    t4 = _t4()
    orig = _orig_trust4()

    fqd = tmp_path / "fqdir"
    _make_pe_tree(str(fqd), ["FQ_S1", "FQ_S2"])

    o_root = tmp_path / "tf_orig"
    p_root = tmp_path / "tf_port"
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "tf_o.jsonl"))
    with pytest.raises(SystemExit) as e:
        orig.main(["--fqdir", str(fqd), "-o", str(o_root), "-t", "2",
                   "--extraToolOptX", "7"])  # unknown pass-through
    assert e.value.code == 0
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "tf_p.jsonl"))
    with pytest.raises(SystemExit) as e:
        t4.main(["--fqdir", str(fqd), "-o", str(p_root), "-t", "2",
                 "--extraToolOptX", "7"])  # unknown pass-through
    assert e.value.code == 0
    _assert_trees_equal(str(o_root), str(p_root))
    # pass-through reached the tool command
    log_calls = _calls(str(tmp_path / "logs" / "tf_p.jsonl"))
    assert any("--extraToolOptX" in c["argv"] for c in log_calls)
    o_calls = _norm_calls(_calls(str(tmp_path / "logs" / "tf_o.jsonl")),
                          [str(o_root), str(fqd), str(fakebins)])
    p_calls = _norm_calls(log_calls, [str(p_root), str(fqd), str(fakebins)])
    o_n = sorted(str([b, tuple(t.replace("@ROOT@", "@X@") for t in a)]) for b, a in o_calls)
    p_n = sorted(str([b, tuple(t.replace("@ROOT@", "@X@") for t in a)]) for b, a in p_calls)
    assert o_n == p_n


def test_trust4_error_paths_and_bin_override(tmp_path, monkeypatch, fakebins):
    t4 = _t4()
    orig = _orig_trust4()

    # runner missing -> identical message + exit 127 on both sides
    empty_bin = tmp_path / "nobin"
    os.makedirs(empty_bin)
    monkeypatch.setenv("PATH", str(empty_bin))
    bam = tmp_path / "x.bam"
    _mini_bam(str(bam))
    with pytest.raises(SystemExit) as e:
        orig.main(["-b", str(bam), "-o", str(tmp_path / "z1")])
    assert e.value.code == 127
    with pytest.raises(SystemExit) as e:
        t4.main(["-b", str(bam), "-o", str(tmp_path / "z2")])
    assert e.value.code == 127
    monkeypatch.setenv("PATH", f"{fakebins}{os.pathsep}{os.environ['PATH']}")

    # conflicting inputs -> argparse error, exit 2 on both sides
    fqd = tmp_path / "fqdir2"
    _make_pe_tree(str(fqd), ["Q1"])
    with pytest.raises(SystemExit) as e:
        orig.main(["--fqdir", str(fqd), "-b", str(bam), "-o", str(tmp_path / "z3")])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        t4.main(["--fqdir", str(fqd), "-b", str(bam), "-o", str(tmp_path / "z4")])
    assert e.value.code == 2

    # no input at all -> exit 2
    with pytest.raises(SystemExit) as e:
        t4.main(["-o", str(tmp_path / "z5")])
    assert e.value.code == 2

    # trust4_bin override (TRUST4_BIN semantics preserved when None)
    alt = _write_fake(str(tmp_path / "altbin"), "my-trust", _FAKE_TRUST4)
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "bin_ovr.jsonl"))
    o_root = tmp_path / "ovr"
    os.makedirs(o_root)
    with pytest.raises(SystemExit) as e:
        t4.main(["-b", str(bam), "-o", str(o_root / "p"), "-t", "1"],
                trust4_bin=alt)
    assert e.value.code == 0
    assert _calls(str(tmp_path / "logs" / "bin_ovr.jsonl"))[0]["bin"] == alt

    # TRUST4_BIN env honored (upstream resolution order)
    monkeypatch.setenv("TRUST4_BIN", alt)
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "bin_env.jsonl"))
    o_root2 = tmp_path / "ovr2"
    os.makedirs(o_root2)
    with pytest.raises(SystemExit) as e:
        t4.main(["-b", str(bam), "-o", str(o_root2 / "p")])
    assert e.value.code == 0
    assert _calls(str(tmp_path / "logs" / "bin_env.jsonl"))[0]["bin"] == alt


def test_trust4_original_escape_hatch(tmp_path, monkeypatch, fakebins):
    t4 = _t4()
    bam = tmp_path / "esc.bam"
    _mini_bam(str(bam))
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "esc_t4.jsonl"))
    root = tmp_path / "esc_root"
    os.makedirs(root)
    with pytest.raises(SystemExit) as e:
        t4.trust4_original(["-b", str(bam), "-o", str(root / "e"), "-t", "1"])
    assert e.value.code == 0
    assert (root / "trust4_immdata.csv").exists()


# ---------------------------------------------------------------------------
# accelerated immune post-processing: bit-identical to the ORIGINAL
# ---------------------------------------------------------------------------
def _synthetic_report(path, sample, n_clones, seed, degenerate=False,
                      zero_counts=False, all_zero=False):
    rng = __import__("numpy").random.default_rng(seed)
    header = "#count\tfrequency\tCDR3nt\tCDR3aa\tV\tD\tJ\tC\tcid\tcid_full_length\n"
    if degenerate:
        with open(path, "w") as f:
            f.write(header)
        return
    counts = rng.integers(1, 500, n_clones)
    if zero_counts:
        counts[: max(1, n_clones // 5)] = 0
    if all_zero:
        counts[:] = 0
    with open(path, "w") as f:
        f.write(header)
        for i in range(n_clones):
            nt = "TGTGCA" + "".join(rng.choice(list("ACGT"), size=3 * (5 + i % 20)))
            aa = "CAS" + "K" * (5 + i % 20)
            f.write(f"{counts[i]}\t0.01\t{nt}\t{aa}\tTRBV{i%5+1}\t*\tTRBJ{i%3+1}\tTRBC1\tcid{i}\tcid{i}\n")


def test_immune_postproc_fast_bitwise(tmp_path):
    t4 = _t4()
    pytest.importorskip("iobrpy.utils.process_immune_data_batch")
    from iobrpy.utils.process_immune_data_batch import process_immune_data_batch

    rep_dir = tmp_path / "reports"
    os.makedirs(rep_dir)
    _synthetic_report(rep_dir / "S1_report.tsv", "S1", 400, 1)
    _synthetic_report(rep_dir / "S2_report.tsv", "S2", 1, 2)            # single clone
    _synthetic_report(rep_dir / "S3_report.tsv", "S3", 250, 3, zero_counts=True)
    _synthetic_report(rep_dir / "S0_report.tsv", "S0", 120, 4, degenerate=True)  # header only
    _synthetic_report(rep_dir / "S4_report.tsv", "S4", 30, 6, all_zero=True)     # Nreads<=0
    _synthetic_report(rep_dir / "S10_report.tsv", "S10", 60, 5)        # sort-order trap

    o_imm = tmp_path / "o_immdata.csv"
    o_idx = tmp_path / "o_indices.csv"
    f_imm = tmp_path / "f_immdata.csv"
    f_idx = tmp_path / "f_indices.csv"

    df_o = process_immune_data_batch(str(rep_dir), str(o_imm), str(o_idx))
    df_f = t4.process_immune_data_batch_fast(str(rep_dir), str(f_imm), str(f_idx))

    # CSV artifacts byte-identical (incl. float tokens and row order)
    assert _sha(str(o_imm)) == _sha(str(f_imm)), "immdata.csv bytes differ"
    assert _sha(str(o_idx)) == _sha(str(f_idx)), "immune_indices.csv bytes differ"
    # returned frames identical (values, order, dtypes)
    pd.testing.assert_frame_equal(df_o, df_f, check_exact=True)
    # sanity: sorted-key group order; header-only sample contributes NO rows
    # (upstream groupby semantics); all-zero sample takes the Nreads<=0 branch
    idx = pd.read_csv(f_idx)
    assert list(idx["Sample"]) == ["S1", "S10", "S2", "S3", "S4"]
    imm = pd.read_csv(f_imm)
    assert "S0" not in set(imm["Sample"]) and "S0" not in set(idx["Sample"])
    s4 = idx[idx["Sample"] == "S4"].iloc[0]
    assert s4["Nreads"] == 0 and s4["Nclones"] == 0
    assert pd.isna(s4["Length_CDR3"]) and pd.isna(s4["Gini"])
    assert idx.loc[idx["Sample"] == "S2", "Second_top_clone"].isna().iloc[0]


def test_immune_postproc_fast_degenerate_real_shape(tmp_path):
    """The frozen real-data TRUST4 gate shape: ONE header-only report ->
    header-only immdata.csv and the 1-byte '\\n' indices file (upstream
    ``pd.DataFrame([]).to_csv`` behaviour)."""
    t4 = _t4()
    pytest.importorskip("iobrpy.utils.process_immune_data_batch")
    from iobrpy.utils.process_immune_data_batch import process_immune_data_batch

    rep = tmp_path / "rep"
    os.makedirs(rep)
    (rep / "04-trust4_report.tsv").write_text(
        "#count\tfrequency\tCDR3nt\tCDR3aa\tV\tD\tJ\tC\tcid\tcid_full_length\n")

    t4.process_immune_data_batch_fast(str(rep), str(tmp_path / "f_imm.csv"),
                                      str(tmp_path / "f_idx.csv"))
    process_immune_data_batch(str(rep), str(tmp_path / "o_imm.csv"),
                              str(tmp_path / "o_idx.csv"))
    assert (tmp_path / "f_idx.csv").read_bytes() == b"\n"
    assert _sha(str(tmp_path / "f_imm.csv")) == _sha(str(tmp_path / "o_imm.csv"))
    assert _sha(str(tmp_path / "f_idx.csv")) == _sha(str(tmp_path / "o_idx.csv"))


# ---------------------------------------------------------------------------
# *_original escape hatches for the remaining stages
# ---------------------------------------------------------------------------
def test_fastq_qc_and_star_original_escape_hatches(tmp_path, monkeypatch, fakebins):
    fq = _fq()
    bst = _bst()

    in_dir = tmp_path / "eh_in"
    _make_pe_tree(str(in_dir), ["EH1"])
    _set_calllog(monkeypatch, str(tmp_path / "logs" / "eh.jsonl"))
    assert fq.fastq_qc_original(str(in_dir), str(tmp_path / "eh_qc"),
                                num_threads=2, batch_size=1) is None
    assert (tmp_path / "eh_qc" / "EH1.task.complete").exists()

    index = tmp_path / "eh_idx"
    os.makedirs(index)
    res = bst.batch_star_count_original(str(index), str(in_dir),
                                        str(tmp_path / "eh_star"),
                                        num_threads=2)
    assert res["rc"] == 0
    assert (tmp_path / "eh_star" / "EH1.task.complete").exists()
