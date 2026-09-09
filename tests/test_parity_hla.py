"""HLA orchestration parity gates (spechla / hla_typing).

Synthetic-miniature contract tests in the same style as
``tests/test_parity_orchestration.py``: the REAL SpecHLA pipeline
(bowtie2 / bwa / freebayes / SpecHap / ExtractHAIRs ...) is replaced by
deterministic stub assets — a miniature SpecHLA root (stub
``SpecHLA_RNAseq.sh`` / ``ExtractHLAread.sh`` that RECORD their exact argv
to a JSONL-ish call log and emit fixed stub products), stub external tools
(samtools / bwa / bowtie2 / bowtie2-build / freebayes / bgzip / tabix / bam
/ bcftools / conda) prepended to PATH — and the ORIGINAL
``iobrpy.SpecHLA.SpecHLA`` / ``iobrpy.workflow.hla_typing`` flows and the
PORTED ``iobrx._fast.spechla_fast`` / ``iobrx._fast.hla_typing_fast`` flows
are run on identical inputs into separate trees and compared:

* output trees BYTE-identical (``hla.result.txt``, ``hla.result.details``,
  ``hla_result_merged.txt`` from BOTH merge variants, extracted stub
  FASTQs, ``<id>.ExtractHLAread.done`` / ``<id>.SpecHLA.done`` markers);
* recorded external-command call logs TOKEN-identical after outdir
  normalization (sequential order preserved — both arms are sequential);
* stdout protocol lines identical after outdir normalization (including
  the position of the ``[SpecHLA] Using SpecHLA root:`` line, which the
  port keeps inside ``run_spechla_phase``, exactly where upstream prints
  it);
* pure-helper parity (``infer_sample_id`` incl. the
  ``_Aligned.sortedByCoord.out.bam`` rule, ``collect_samples`` ordering,
  ``find_fastqs_for_sample`` preference, ``hla_result_has_sample``
  case/whitespace variants, ``_parse_version_prefix``,
  ``get_bcftools_version``);
* command construction (``run_spechla_rnaseq`` argv, ``run_extraction``
  argv — the port's only difference is the script-path root);
* resume/skip logic (done markers -> no new tool calls), failure paths
  (vcflib-without-conda exit 1, missing blastn exit 1, argparse exit 2)
  identical across arms, and the ``backend='python'`` escape hatch.

The heavy real-toolchain parity (official STAR BAM + cleaned FASTQ subset,
env_hla binaries, deployed SpecHap/ExtractHAIRs) runs outside pytest via
``research/build_hla`` — see the port's INTEGRATION.md.

Run with the repo source on the path::

    PYTHONPATH=<repo>/src pytest -q tests/test_parity_hla.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# module handles
# ---------------------------------------------------------------------------
def _sp():
    return pytest.importorskip(
        "iobrx._fast.spechla_fast",
        reason="needs PYTHONPATH=<repo>/src (or an installed iobrx)",
    )


def _ht():
    return pytest.importorskip("iobrx._fast.hla_typing_fast")


def _orig_spec():
    return pytest.importorskip("iobrpy.SpecHLA.SpecHLA")


def _orig_hla():
    return pytest.importorskip("iobrpy.workflow.hla_typing")


def _api():
    return pytest.importorskip("iobrx")


# ---------------------------------------------------------------------------
# stub assets
# ---------------------------------------------------------------------------
_TOOL_NAMES = [
    "samtools", "bwa", "bowtie2", "freebayes", "bgzip", "tabix", "bam",
    "perl", "novoalign",
]

_TOOL_STUB = '''#!/usr/bin/env bash
{{
  echo "CALL $(basename "$0")"
  for a in "$@"; do echo "ARG $a"; done
}} >> "${{FAKE_CALL_LOG:-/dev/null}}"
exit 0
'''

_BCFTOOLS_STUB = '''#!/usr/bin/env bash
{
  echo "CALL bcftools"
  for a in "$@"; do echo "ARG $a"; done
} >> "${FAKE_CALL_LOG:-/dev/null}"
if [[ "$1" == "--version" ]]; then
  echo "bcftools 1.21"
  echo "Using htslib 1.21"
  exit 0
fi
exit 0
'''

_BOWTIE2_BUILD_STUB = '''#!/usr/bin/env bash
{
  echo "CALL bowtie2-build"
  for a in "$@"; do echo "ARG $a"; done
} >> "${FAKE_CALL_LOG:-/dev/null}"
# last arg = index prefix; create the six .bt2 files ensure_bowtie2_index checks
prefix="${@: -1}"
for ext in 1.bt2 2.bt2 3.bt2 4.bt2 rev.1.bt2 rev.2.bt2; do
  echo stub > "${prefix}.${ext}"
done
exit 0
'''

_CONDA_STUB = '''#!/usr/bin/env bash
{
  echo "CALL conda"
  for a in "$@"; do echo "ARG $a"; done
} >> "${FAKE_CALL_LOG:-/dev/null}"
sub="$1"; shift || true
pkg="${1:-}"
if [[ "$sub" == "list" ]]; then
  if [[ "$*" == *"--json"* ]]; then
    case "$pkg" in
      vcflib)     echo '[{"name":"vcflib","version":"1.0.10","build_string":"hdcf5f25_1","channel":"bioconda"}]';;
      libdeflate) echo '[{"name":"libdeflate","version":"1.25","build_string":"x","channel":"conda-forge"}]';;
      htslib)     echo '[{"name":"htslib","version":"1.21","build_string":"x","channel":"bioconda"}]';;
      *)          echo '[]';;
    esac
    exit 0
  fi
  echo "# packages in environment:"
  echo "#"
  case "$pkg" in
    vcflib)     printf 'vcflib                   1.0.10                hdcf5f25_1    bioconda\\n';;
    libdeflate) printf 'libdeflate               1.25                  hd45a770_1    conda-forge\\n';;
    htslib)     printf 'htslib                   1.21                  h3a4d415_1    bioconda\\n';;
    bcftools)   printf 'bcftools                 1.21                  h3a4d415_1    bioconda\\n';;
  esac
  exit 0
fi
if [[ "$sub" == "install" ]]; then exit 0; fi
if [[ "$sub" == "info" ]]; then echo '{}'; exit 0; fi
exit 0
'''

_RNASEQ_STUB = '''#!/usr/bin/env bash
{
  echo "CALL SpecHLA_RNAseq.sh"
  for a in "$@"; do echo "ARG $a"; done
} >> "${FAKE_CALL_LOG:-/dev/null}"
n=""; o=""; u="1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) n="$2"; shift 2;;
    -o) o="$2"; shift 2;;
    -u) u="$2"; shift 2;;
    *) shift;;
  esac
done
mkdir -p "$o/$n"
res="$o/$n/hla.result.txt"
printf '# version: IPD-IMGT/HLA 3.38.0\\n' > "$res"
printf 'Sample\\tHLA_A_1\\tHLA_A_2\\n' >> "$res"
printf '%s\\tA*02:01\\tA*11:01\\n' "$n" >> "$res"
printf '# details stub u=%s n=%s\\n' "$u" "$n" > "$o/$n/hla.result.details.txt"
exit 0
'''

_EXTRACT_STUB = '''#!/usr/bin/env bash
{
  echo "CALL ExtractHLAread.sh"
  for a in "$@"; do echo "ARG $a"; done
} >> "${FAKE_CALL_LOG:-/dev/null}"
s=""; b=""; r=""; o=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s) s="$2"; shift 2;;
    -b) b="$2"; shift 2;;
    -r) r="$2"; shift 2;;
    -o) o="$2"; shift 2;;
    *) shift;;
  esac
done
mkdir -p "$o"
echo "stub-fastq-$s-1" | gzip -n > "$o/${s}_extract_1.fq.gz"
echo "stub-fastq-$s-2" | gzip -n > "$o/${s}_extract_2.fq.gz"
echo "stub-unpaired-$s" | gzip -n > "$o/${s}_extract.unpaired.fq.gz"
exit 0
'''

_SPECHAP_STUB = '''#!/usr/bin/env bash
{ echo "CALL SpecHap"; for a in "$@"; do echo "ARG $a"; done; } >> "${FAKE_CALL_LOG:-/dev/null}"
exit 0
'''


def _write_exec(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def make_fake_tools(tmp_path: Path) -> Path:
    """Stub external tools + conda, all argv-recording."""
    d = tmp_path / "fake_tools"
    d.mkdir(parents=True, exist_ok=True)
    for t in _TOOL_NAMES:
        _write_exec(d / t, _TOOL_STUB)
    _write_exec(d / "bcftools", _BCFTOOLS_STUB)
    _write_exec(d / "bowtie2-build", _BOWTIE2_BUILD_STUB)
    _write_exec(d / "conda", _CONDA_STUB)
    return d


def make_fake_root(tmp_path: Path, name: str = "fake_root",
                   bt2_for_exon: bool = True) -> Path:
    """Miniature SpecHLA asset tree (script/ + db/ + bin/)."""
    root = tmp_path / name
    (root / "script" / "whole").mkdir(parents=True, exist_ok=True)
    (root / "db" / "ref").mkdir(parents=True, exist_ok=True)
    (root / "bin" / "SpecHap" / "build").mkdir(parents=True, exist_ok=True)
    (root / "bin" / "extractHairs" / "build").mkdir(parents=True, exist_ok=True)
    (root / "bin" / "fermikit" / "fermi.kit").mkdir(parents=True, exist_ok=True)

    _write_exec(root / "script" / "whole" / "SpecHLA_RNAseq.sh", _RNASEQ_STUB)
    _write_exec(root / "script" / "ExtractHLAread.sh", _EXTRACT_STUB)
    _write_exec(root / "bin" / "SpecHap" / "build" / "SpecHap", _SPECHAP_STUB)
    _write_exec(root / "bin" / "extractHairs" / "build" / "ExtractHAIRs",
                _SPECHAP_STUB)
    _write_exec(root / "bin" / "bcftools", _BCFTOOLS_STUB)   # bundled, 1.21
    _write_exec(root / "bin" / "blastn", _TOOL_STUB)         # bundled blastn
    (root / "bin" / "libwfa2.so.0").write_bytes(b"")         # LD path branch

    exon_fa = root / "db" / "ref" / "hla_gen.format.filter.extend.DRB.no26789.fasta"
    exon_fa.write_text(">stub\nACGT\n")
    if bt2_for_exon:
        for ext in ("1.bt2", "2.bt2", "3.bt2", "4.bt2", "rev.1.bt2",
                    "rev.2.bt2"):
            (root / "db" / "ref" / (exon_fa.name + "." + ext)).write_text("stub\n")
    v2_fa = root / "db" / "ref" / "hla_gen.format.filter.extend.DRB.no26789.v2.fasta"
    v2_fa.write_text(">stub-v2\nACGT\n")
    (root / "install_spechap.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    return root


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Snapshot/restore the process environment and strip ambient overrides
    that would leak into the wrapper gates."""
    snapshot = dict(os.environ)
    for var in ("BCFTOOLS", "BOWTIE2_BUILD", "CONDA_EXE", "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV", "SPECHLA_ROOT", "FAKE_CALL_LOG",
                "LD_LIBRARY_PATH"):
        monkeypatch.delenv(var, raising=False)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def _wire_env(monkeypatch, tools_dir: Path, log_path: Path,
              base_path: str) -> None:
    """Reset the wrapper-visible env to a CLEAN baseline for one arm.

    The gates mutate ``os.environ`` (PATH prepends, ``BCFTOOLS``,
    ``LD_LIBRARY_PATH``); without this reset, arm 2 would inherit arm 1's
    mutations (e.g. a stale ``BCFTOOLS`` pointing into arm 1's fake root)
    and the stdout/log comparison would see cross-arm leakage that is not
    part of either implementation's behaviour.
    """
    monkeypatch.delenv("BCFTOOLS", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("PATH", f"{tools_dir}{os.pathsep}{base_path}")
    monkeypatch.setenv("FAKE_CALL_LOG", str(log_path))
    monkeypatch.setenv("CONDA_EXE", str(tools_dir / "conda"))


def _tree_hashes(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _normalize_call_log(log_path: Path, replacements: dict) -> list:
    lines = log_path.read_text().splitlines()
    norm = []
    for ln in lines:
        for a, b in replacements.items():
            ln = ln.replace(a, b)
        norm.append(ln)
    return norm


# ---------------------------------------------------------------------------
# spechla end-to-end (stubbed pipeline): original vs port
# ---------------------------------------------------------------------------
def _run_spechla_both_arms(tmp_path, monkeypatch, use_exon=1, extra_argv=()):
    tools = make_fake_tools(tmp_path)
    base_path = os.environ["PATH"]
    sp = _sp()
    orig = _orig_spec()

    results = {}
    for arm in ("orig", "port"):
        root = make_fake_root(tmp_path, name=f"root_{arm}")
        outdir = tmp_path / f"out_{arm}"
        log = tmp_path / f"calls_{arm}.log"
        _wire_env(monkeypatch, tools, log, base_path)

        argv = ["-n", "SMPL", "-1", str(tmp_path / "r1.fq.gz"),
                "-2", str(tmp_path / "r2.fq.gz"), "-o", str(outdir),
                "-j", "4", "-u", str(use_exon), *extra_argv]
        (tmp_path / "r1.fq.gz").write_bytes(b"r1")
        (tmp_path / "r2.fq.gz").write_bytes(b"r2")

        import io
        from contextlib import redirect_stdout, redirect_stderr
        so, se = io.StringIO(), io.StringIO()
        code = 0
        try:
            if arm == "orig":
                def _detect_stub(_r=str(root)):
                    # keep the upstream protocol line (the real detector
                    # prints it; the port prints it in _resolve_spec_hla_root)
                    print(f"[SpecHLA] Using SpecHLA root: {_r}")
                    return _r
                monkeypatch.setattr(orig, "detect_spec_hla_root", _detect_stub)
                with redirect_stdout(so), redirect_stderr(se):
                    orig.main(argv)
            else:
                with redirect_stdout(so), redirect_stderr(se):
                    sp.main(argv, spec_hla_root=str(root))
        except SystemExit as e:  # pragma: no cover - only failure paths
            code = e.code if isinstance(e.code, int) else 1
        results[arm] = {
            "root": root, "outdir": outdir, "log": log, "rc": code,
            "stdout": so.getvalue(), "stderr": se.getvalue(),
        }
    return results


def test_spechla_e2e_output_tree_parity(tmp_path, monkeypatch):
    res = _run_spechla_both_arms(tmp_path, monkeypatch)
    assert res["orig"]["rc"] == 0 and res["port"]["rc"] == 0
    ta = _tree_hashes(res["orig"]["outdir"])
    tb = _tree_hashes(res["port"]["outdir"])
    assert ta == tb and ta  # byte-identical AND non-empty


def test_spechla_e2e_call_log_parity(tmp_path, monkeypatch):
    res = _run_spechla_both_arms(tmp_path, monkeypatch)
    la = _normalize_call_log(res["orig"]["log"], {
        str(res["orig"]["root"]): "<ROOT>", str(res["orig"]["outdir"]): "<OUT>"})
    lb = _normalize_call_log(res["port"]["log"], {
        str(res["port"]["root"]): "<ROOT>", str(res["port"]["outdir"]): "<OUT>"})
    assert la == lb
    assert any("CALL SpecHLA_RNAseq.sh" in x for x in la)


def test_spechla_e2e_stdout_protocol_parity(tmp_path, monkeypatch):
    res = _run_spechla_both_arms(tmp_path, monkeypatch)
    def norm(arm):
        s = res[arm]["stdout"]
        s = s.replace(str(res[arm]["root"]), "<ROOT>")
        s = s.replace(str(res[arm]["outdir"]), "<OUT>")
        return [ln for ln in s.splitlines() if ln.strip()]
    assert norm("orig") == norm("port")


def test_spechla_wgs_mode_builds_v2_bowtie2_index(tmp_path, monkeypatch):
    # -u 0 -> v2 DRB fasta without prebuilt .bt2 -> bowtie2-build invoked
    # identically in both arms (and the stub index files appear).
    res = _run_spechla_both_arms(tmp_path, monkeypatch, use_exon=0)
    for arm in ("orig", "port"):
        la = _normalize_call_log(res[arm]["log"], {
            str(res[arm]["root"]): "<ROOT>", str(res[arm]["outdir"]): "<OUT>"})
        assert "CALL bowtie2-build" in la
        v2 = (res[arm]["root"] / "db" / "ref" /
              "hla_gen.format.filter.extend.DRB.no26789.v2.fasta.1.bt2")
        assert v2.is_file()
    la = _normalize_call_log(res["orig"]["log"], {
        str(res["orig"]["root"]): "<ROOT>", str(res["orig"]["outdir"]): "<OUT>"})
    lb = _normalize_call_log(res["port"]["log"], {
        str(res["port"]["root"]): "<ROOT>", str(res["port"]["outdir"]): "<OUT>"})
    assert la == lb


# ---------------------------------------------------------------------------
# hla_typing end-to-end (stubbed pipeline): original vs port
# ---------------------------------------------------------------------------
def _make_bam_dir(tmp_path: Path) -> Path:
    d = tmp_path / "bams"
    d.mkdir()
    (d / "S1_Aligned.sortedByCoord.out.bam").write_bytes(b"stub-bam-1")
    (d / "S2.bam").write_bytes(b"stub-bam-2")
    return d


def _run_hla_typing_both_arms(tmp_path, monkeypatch):
    tools = make_fake_tools(tmp_path)
    base_path = os.environ["PATH"]
    ht = _ht()
    orig = _orig_hla()
    bam_dir = _make_bam_dir(tmp_path)

    results = {}
    for arm in ("orig", "port"):
        root = make_fake_root(tmp_path, name=f"hroot_{arm}")
        outdir = tmp_path / f"hout_{arm}"
        log = tmp_path / f"hcalls_{arm}.log"
        _wire_env(monkeypatch, tools, log, base_path)

        argv = ["-b", str(bam_dir), "-r", "hg38", "-o", str(outdir),
                "-j", "4", "-u", "1"]

        if arm == "orig":
            def _detect_stub(_r=str(root)):
                print(f"[SpecHLA] Using SpecHLA root: {_r}")
                return _r
            monkeypatch.setattr(orig, "detect_spec_hla_root", _detect_stub)

            def _orig_run_extraction(sample_id, bam_path, ref, outdir,
                                     _root=root):
                # mirrors the ported run_extraction: same script, same tokens
                sh = Path(_root) / "script" / "ExtractHLAread.sh"
                outdir.mkdir(parents=True, exist_ok=True)
                cmd = ["bash", str(sh), "-s", sample_id, "-b", str(bam_path),
                       "-r", ref, "-o", str(outdir)]
                print("[extract_hla_read] Running:", " ".join(cmd),
                      file=sys.stderr)
                subprocess.run(cmd, check=True)

            monkeypatch.setattr(orig, "run_extraction", _orig_run_extraction)
            runner = lambda: orig.main(argv)
        else:
            runner = lambda: ht.main(argv, spec_hla_root=str(root))

        import io
        from contextlib import redirect_stdout, redirect_stderr
        so, se = io.StringIO(), io.StringIO()
        code = 0
        try:
            with redirect_stdout(so), redirect_stderr(se):
                runner()
        except SystemExit as e:  # pragma: no cover
            code = e.code if isinstance(e.code, int) else 1
        results[arm] = {
            "root": root, "outdir": outdir, "log": log, "rc": code,
            "stdout": so.getvalue(), "stderr": se.getvalue(),
        }
    return results


def test_hla_typing_e2e_output_tree_parity(tmp_path, monkeypatch):
    res = _run_hla_typing_both_arms(tmp_path, monkeypatch)
    assert res["orig"]["rc"] == 0 and res["port"]["rc"] == 0
    ta = _tree_hashes(res["orig"]["outdir"])
    tb = _tree_hashes(res["port"]["outdir"])
    assert ta == tb and ta
    # expected content: extracted stub FASTQs, done markers, per-sample
    # results, merged table
    names = set(ta)
    for expect in (
        "ExtractHLAread/S1/S1_extract_1.fq.gz",
        "ExtractHLAread/S1/S1_extract_2.fq.gz",
        "ExtractHLAread/S1/S1.ExtractHLAread.done",
        "ExtractHLAread/S2/S2.ExtractHLAread.done",
        "SpecHLA/S1/hla.result.txt",
        "SpecHLA/S1/S1.SpecHLA.done",
        "SpecHLA/S2/hla.result.txt",
        "hla_result_merged.txt",
    ):
        assert expect in names, expect
    merged = (res["orig"]["outdir"] / "hla_result_merged.txt").read_text()
    assert merged.splitlines()[0].startswith("# version:")
    assert merged.splitlines()[1].startswith("Sample\t")
    assert "S1\t" in merged and "S2\t" in merged


def test_hla_typing_e2e_call_log_parity(tmp_path, monkeypatch):
    res = _run_hla_typing_both_arms(tmp_path, monkeypatch)
    la = _normalize_call_log(res["orig"]["log"], {
        str(res["orig"]["root"]): "<ROOT>", str(res["orig"]["outdir"]): "<OUT>"})
    lb = _normalize_call_log(res["port"]["log"], {
        str(res["port"]["root"]): "<ROOT>", str(res["port"]["outdir"]): "<OUT>"})
    assert la == lb
    # pipeline calls only (the log also records the bcftools --version
    # probes from ensure_bcftools_121 — symmetric across arms, already
    # covered by the full-log equality above)
    calls = [x for x in la if x.startswith("CALL ") and
             ("ExtractHLAread.sh" in x or "SpecHLA_RNAseq.sh" in x)]
    # sequential schedule: extract S1, extract S2, spechla S1, spechla S2
    assert calls == ["CALL ExtractHLAread.sh", "CALL ExtractHLAread.sh",
                     "CALL SpecHLA_RNAseq.sh", "CALL SpecHLA_RNAseq.sh"]


def test_hla_typing_e2e_stdout_protocol_parity(tmp_path, monkeypatch):
    res = _run_hla_typing_both_arms(tmp_path, monkeypatch)
    def norm(arm):
        s = res[arm]["stdout"]
        s = s.replace(str(res[arm]["root"]), "<ROOT>")
        s = s.replace(str(res[arm]["outdir"]), "<OUT>")
        return [ln for ln in s.splitlines() if ln.strip()]
    a, b = norm("orig"), norm("port")
    assert a == b
    # the root line prints INSIDE the SpecHLA phase (upstream position)
    roots = [i for i, ln in enumerate(a) if "Using SpecHLA root" in ln]
    starts = [i for i, ln in enumerate(a) if "Starting SpecHLA for" in ln]
    assert roots and starts and roots[0] < starts[0]


def test_hla_typing_resume_skips_completed_samples(tmp_path, monkeypatch):
    ht = _ht()
    tools = make_fake_tools(tmp_path)
    base_path = os.environ["PATH"]
    root = make_fake_root(tmp_path, name="hroot_resume")
    bam_dir = _make_bam_dir(tmp_path)
    outdir = tmp_path / "hout_resume"
    log = tmp_path / "hcalls_resume.log"
    _wire_env(monkeypatch, tools, log, base_path)
    argv = ["-b", str(bam_dir), "-r", "hg38", "-o", str(outdir), "-j", "4",
            "-u", "1"]
    ht.main(argv, spec_hla_root=str(root))
    first = _normalize_call_log(log, {str(root): "<ROOT>", str(outdir): "<OUT>"})
    n_first = sum(1 for x in first if x.startswith("CALL ") and
                  ("ExtractHLAread.sh" in x or "SpecHLA_RNAseq.sh" in x))

    # rerun: both phases skip via done markers -> no new pipeline calls
    log2 = tmp_path / "hcalls_resume2.log"
    _wire_env(monkeypatch, tools, log2, base_path)
    ht.main(argv, spec_hla_root=str(root))
    second = _normalize_call_log(log2, {str(root): "<ROOT>",
                                        str(outdir): "<OUT>"})
    pipeline_calls = [x for x in second
                      if x.startswith("CALL ") and
                      ("SpecHLA_RNAseq.sh" in x or "ExtractHLAread.sh" in x)]
    assert n_first == 4
    assert pipeline_calls == []
    # merged table still (re)written identically
    assert (outdir / "hla_result_merged.txt").is_file()


# ---------------------------------------------------------------------------
# pure-helper parity
# ---------------------------------------------------------------------------
def test_infer_sample_id_parity(tmp_path):
    ht, orig = _ht(), _orig_hla()
    cases = [
        "S1_Aligned.sortedByCoord.out.bam",
        "S2.bam",
        "weird.name_Aligned.sortedByCoord.out.bam",
        "x.y.bam",
    ]
    for c in cases:
        p = tmp_path / c
        assert ht.infer_sample_id(p) == orig.infer_sample_id(p)
    assert ht.infer_sample_id(tmp_path / "S1_Aligned.sortedByCoord.out.bam") \
        == "S1"
    with pytest.raises(ValueError):
        ht.infer_sample_id(tmp_path / "not_a_bam.txt")


def test_collect_samples_parity(tmp_path):
    ht, orig = _ht(), _orig_hla()
    bam_dir = _make_bam_dir(tmp_path)
    (bam_dir / "ignored.txt").write_text("x")
    a = orig.collect_samples(bam_dir)
    b = ht.collect_samples(bam_dir)
    assert [(i, str(p)) for i, p in a] == [(i, str(p)) for i, p in b]
    assert [i for i, _ in b] == ["S1", "S2"]


def test_find_fastqs_for_sample_parity(tmp_path):
    ht, orig = _ht(), _orig_hla()
    d = tmp_path / "sample_dir"
    d.mkdir()
    (d / "SMPL_extract_1.fq.gz").write_bytes(b"a")
    (d / "SMPL_extract_2.fq.gz").write_bytes(b"b")
    (d / "zzz_other_1.fq.gz").write_bytes(b"c")
    (d / "zzz_other_2.fq.gz").write_bytes(b"d")
    assert [str(x) for x in orig.find_fastqs_for_sample(d, "SMPL")] == \
           [str(x) for x in ht.find_fastqs_for_sample(d, "SMPL")]
    # preference falls back to the first sorted candidate
    e = tmp_path / "sample_dir2"
    e.mkdir()
    (e / "aaa_1.fq.gz").write_bytes(b"a")
    (e / "aaa_2.fq.gz").write_bytes(b"b")
    assert [x.name for x in ht.find_fastqs_for_sample(e, "SMPL")] == \
           ["aaa_1.fq.gz", "aaa_2.fq.gz"]
    with pytest.raises(FileNotFoundError):
        ht.find_fastqs_for_sample(tmp_path / "missing", "X")


@pytest.mark.parametrize("content,expected", [
    ("# version: IPD-IMGT/HLA 3.38.0\nSample\tHLA_A_1\nSMPL\tA*02:01\n", True),
    ("# version: x\nSample\tHLA_A_1\nOTHER\tA*02:01\n", False),
    ("Sample HLA_A_1\nSMPL A*02:01\n", True),           # whitespace fallback
    ("sample\tHLA_A_1\nSMPL\tA*02:01\n", True),         # lowercase header
    ("# only comments\n", False),
    ("", False),
])
def test_hla_result_has_sample_parity(tmp_path, content, expected):
    ht, orig = _ht(), _orig_hla()
    d = tmp_path / "sdir"
    d.mkdir()
    (d / "hla.result.txt").write_text(content)
    assert orig.hla_result_has_sample(d, "SMPL") is expected
    assert ht.hla_result_has_sample(d, "SMPL") == \
           orig.hla_result_has_sample(d, "SMPL")
    # results.txt fallback name
    (d / "hla.result.txt").unlink()
    (d / "hla.results.txt").write_text(content)
    assert ht.hla_result_has_sample(d, "SMPL") is expected


def _merge_fixture(tmp_path: Path) -> Path:
    out = tmp_path / "merge_in"
    for sid, body in (
        ("A", "# version: IPD-IMGT/HLA 3.38.0\nSample\tHLA_A_1\nA\tA*01:01\n"),
        ("B", "# version: IPD-IMGT/HLA 3.38.0\nSample\tHLA_A_1\nB\tB*02:01\n"
              "B2row\tB*03:01\n"),
        ("C", "\n\n"),                                  # empty-ish -> warning
        ("D", "# version: IPD-IMGT/HLA 3.38.0\nSample\tHLA_A_1\n"
              "Sample\tDUPLICATE\nD\tD*04:01\n"),       # dup header suppressed
    ):
        d = out / sid
        d.mkdir(parents=True)
        (d / "hla.result.txt").write_text(body)
    (out / "E").mkdir()                                 # missing file -> warn
    return out


def test_spechla_merge_variant_parity(tmp_path, capsys):
    # SpecHLA.py merge_hla_results: line 0 = header, rest = data (quirk
    # preserved verbatim: the '# version' line becomes the header).
    sp, orig = _sp(), _orig_spec()
    src = _merge_fixture(tmp_path)
    oa = tmp_path / "oa"
    ob = tmp_path / "ob"
    shutil.copytree(src, oa)
    shutil.copytree(src, ob)
    orig.merge_hla_results(str(oa))
    capsys.readouterr()
    sp.merge_hla_results(str(ob))
    cap2 = capsys.readouterr()
    assert (oa / "hla_result_merged.txt").read_bytes() == \
           (ob / "hla_result_merged.txt").read_bytes()
    text = (oa / "hla_result_merged.txt").read_text()
    assert text.splitlines()[0].startswith("# version:")


def test_hla_typing_merge_variant_parity(tmp_path, capsys):
    # workflow merge: version line + single header + data lines, duplicate
    # 'Sample...' data lines suppressed, warnings on stderr.
    ht, orig = _ht(), _orig_hla()
    src = _merge_fixture(tmp_path)
    samples = [("A", None), ("B", None), ("C", None), ("D", None),
               ("E", None), ("F", None)]
    oa = tmp_path / "wa"
    ob = tmp_path / "wb"
    shutil.copytree(src, oa)
    shutil.copytree(src, ob)
    import io
    from contextlib import redirect_stderr
    ea, eb = io.StringIO(), io.StringIO()
    with redirect_stderr(ea):
        orig.merge_hla_results(samples, oa, oa)
    with redirect_stderr(eb):
        ht.merge_hla_results(samples, ob, ob)
    assert (oa / "hla_result_merged.txt").read_bytes() == \
           (ob / "hla_result_merged.txt").read_bytes()
    na = [l.replace(str(oa), "<O>") for l in ea.getvalue().splitlines() if l.strip()]
    nb = [l.replace(str(ob), "<O>") for l in eb.getvalue().splitlines() if l.strip()]
    assert na == nb                                    # same warnings
    text = (oa / "hla_result_merged.txt").read_text()
    lines = text.splitlines()
    assert lines[0].startswith("# version:")
    assert lines[1].startswith("Sample\t")
    assert not any(l.startswith("Sample\tDUPLICATE") for l in lines[2:])
    assert any(l.startswith("A\t") for l in lines)
    assert any(l.startswith("B2row\t") for l in lines)


def test_version_parsers_parity(tmp_path):
    sp, orig = _sp(), _orig_spec()
    for tok in ("1.21", "1.21+htslib-1.21", "1.21-123-gabcdef", "", None,
                "v2", "3.4.5.6"):
        assert sp._parse_version_prefix(tok) == orig._parse_version_prefix(tok)
    tools = make_fake_tools(tmp_path)
    assert sp.get_bcftools_version(str(tools / "bcftools")) == \
           orig.get_bcftools_version(str(tools / "bcftools")) == "1.21"
    assert sp.get_bcftools_version(str(tmp_path / "missing")) == \
           orig.get_bcftools_version(str(tmp_path / "missing")) is None


def test_run_spechla_rnaseq_command_parity(tmp_path, monkeypatch):
    sp, orig = _sp(), _orig_spec()
    root = make_fake_root(tmp_path)
    captured = {}
    for mod, key in ((orig, "orig"), (sp, "port")):
        def fake_run(cmd, cwd=None, _key=key, **kw):
            captured[_key] = cmd

            class R:
                returncode = 0
            return R()
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        mod.run_spechla_rnaseq(spec_hla_root=str(root), sample_name="SMPL",
                               read1="/x/r1.fq.gz", read2="/x/r2.fq.gz",
                               outdir=str(tmp_path / "o"), threads=4,
                               use_exon=1)
    assert captured["orig"] == captured["port"]
    assert captured["port"][:2] == ["bash",
                                    str(root / "script" / "whole" /
                                        "SpecHLA_RNAseq.sh")]
    assert captured["port"][2:] == ["-n", "SMPL", "-1", "/x/r1.fq.gz", "-2",
                                    "/x/r2.fq.gz", "-o",
                                    str(tmp_path / "o"), "-j", "4", "-u", "1"]


def test_run_extraction_command_parity(tmp_path, monkeypatch):
    # The ONLY allowed difference: the script path root (original derives it
    # from its own __file__ inside iobrpy; the port from spec_hla_root).
    ht = _ht()
    from iobrpy.SpecHLA import extract_hla_read as orig_ext
    root = make_fake_root(tmp_path)
    captured = {}

    def make_fake(key):
        def fake_run(cmd, **kw):
            captured[key] = cmd

            class R:
                returncode = 0
            return R()
        return fake_run

    monkeypatch.setattr(orig_ext.subprocess, "run", make_fake("orig"))
    orig_ext.run_extraction(sample_id="SMPL",
                            bam_path=Path(tmp_path / "SMPL.bam"),
                            ref="hg38", outdir=tmp_path / "e_orig")
    monkeypatch.setattr(ht.subprocess, "run", make_fake("port"))
    ht.run_extraction(sample_id="SMPL",
                      bam_path=Path(tmp_path / "SMPL.bam"),
                      ref="hg38", outdir=tmp_path / "e_port",
                      spec_hla_root=str(root))
    a, b = captured["orig"], captured["port"]
    assert a[0] == b[0] == "bash"
    assert Path(a[1]).name == Path(b[1]).name == "ExtractHLAread.sh"
    assert Path(a[1]).parent.parent.name == "SpecHLA"        # iobrpy tree
    assert Path(b[1]) == root / "script" / "ExtractHLAread.sh"  # resolved root
    bam = str((tmp_path / "SMPL.bam").resolve())
    assert a[2:8] == b[2:8] == ["-s", "SMPL", "-b", bam, "-r", "hg38"]
    assert a[8] == b[8] == "-o"
    assert Path(a[9]).name == "e_orig" and Path(b[9]).name == "e_port"


# ---------------------------------------------------------------------------
# failure paths & backend dispatch
# ---------------------------------------------------------------------------
def test_spechla_failure_paths_parity(tmp_path, monkeypatch, capsys):
    sp, orig = _sp(), _orig_spec()
    # GitHub runner images ship a working conda in /usr/bin, which would let
    # both arms auto-install tools instead of exercising the failure gates.
    monkeypatch.setattr(sp, "detect_conda_exe", lambda: None)
    monkeypatch.setattr(orig, "detect_conda_exe", lambda: None)
    argv = ["-n", "S", "-1", "r1", "-2", "r2", "-o",
            str(tmp_path / "o"), "-j", "2", "-u", "1"]

    # (a) no conda anywhere -> vcflib gate exits 1 with the upstream message
    tools_no_conda = tmp_path / "tools_nc"
    tools_no_conda.mkdir()
    for t in _TOOL_NAMES:
        _write_exec(tools_no_conda / t, _TOOL_STUB)
    _write_exec(tools_no_conda / "bcftools", _BCFTOOLS_STUB)
    _write_exec(tools_no_conda / "bowtie2-build", _BOWTIE2_BUILD_STUB)
    codes, errs = [], []
    for arm, mod in (("orig", orig), ("port", sp)):
        root = make_fake_root(tmp_path, name=f"froot_{arm}")
        monkeypatch.setenv("PATH", f"{tools_no_conda}{os.pathsep}/usr/bin"
                                   f"{os.pathsep}/bin")
        monkeypatch.delenv("CONDA_EXE", raising=False)
        if arm == "orig":
            monkeypatch.setattr(orig, "detect_spec_hla_root",
                                lambda: str(root))
            run = lambda: orig.main(argv)
        else:
            run = lambda: sp.main(argv, spec_hla_root=str(root))
        with pytest.raises(SystemExit) as ei:
            run()
        codes.append(ei.value.code)
        errs.append(capsys.readouterr().err)
    assert codes == [1, 1]
    assert "vcflib" in errs[0] and "vcflib" in errs[1]

    # (b) missing bundled blastn -> exit 1
    tools = make_fake_tools(tmp_path)
    base_path = os.environ["PATH"]
    codes = []
    for arm, mod in (("orig2", orig), ("port2", sp)):
        root = make_fake_root(tmp_path, name=f"broot_{arm}")
        (root / "bin" / "blastn").unlink()
        _wire_env(monkeypatch, tools, tmp_path / f"blog_{arm}.log", base_path)
        if arm == "orig2":
            monkeypatch.setattr(orig, "detect_spec_hla_root",
                                lambda: str(root))
            run = lambda: orig.main(argv)
        else:
            run = lambda: sp.main(argv, spec_hla_root=str(root))
        with pytest.raises(SystemExit) as ei:
            run()
        codes.append(ei.value.code)
        capsys.readouterr()
    assert codes == [1, 1]

    # (c) argparse: missing required / bad choice -> exit 2
    for mod, kwargs in ((orig, {}), (sp, {"spec_hla_root": None})):
        with pytest.raises(SystemExit) as ei:
            if mod is orig:
                orig.parse_args(["-n", "S"])
            else:
                sp.parse_args(["-n", "S"])
        assert ei.value.code == 2
        capsys.readouterr()


def test_hla_typing_argparse_errors_parity(tmp_path, monkeypatch, capsys):
    ht, orig = _ht(), _orig_hla()
    for argv in (["-b", str(tmp_path), "-r", "hg18", "-o", "x"],  # bad choice
                 ["-r", "hg38", "-o", "x"],                       # missing -b
                 ):
        codes = []
        for mod in (orig, ht):
            with pytest.raises(SystemExit) as ei:
                mod.main(argv, **({} if mod is orig else
                                  {"spec_hla_root": None}))
            codes.append(ei.value.code)
            capsys.readouterr()
        assert codes == [2, 2]


def test_public_api_backend_dispatch(tmp_path, monkeypatch):
    api = _api()
    tools = make_fake_tools(tmp_path)
    base_path = os.environ["PATH"]
    root = make_fake_root(tmp_path, name="api_root")
    log = tmp_path / "api_calls.log"
    _wire_env(monkeypatch, tools, log, base_path)
    outdir = tmp_path / "api_out"
    (tmp_path / "r1.fq.gz").write_bytes(b"r1")
    (tmp_path / "r2.fq.gz").write_bytes(b"r2")

    res = api.spechla(name="API", read1=str(tmp_path / "r1.fq.gz"),
                      read2=str(tmp_path / "r2.fq.gz"), outdir=str(outdir),
                      threads=2, use_exon=1, spec_hla_root=str(root))
    assert res == {"rc": 0}
    assert (outdir / "API" / "hla.result.txt").is_file()
    assert (outdir / "hla_result_merged.txt").is_file()

    # backend='python' runs the untouched upstream main (root resolved from
    # the iobrpy package; monkeypatch its detector to the fake root)
    orig = _orig_spec()
    monkeypatch.setattr(orig, "detect_spec_hla_root", lambda: str(root))
    outdir2 = tmp_path / "api_out_py"
    res2 = api.spechla(name="API", read1=str(tmp_path / "r1.fq.gz"),
                       read2=str(tmp_path / "r2.fq.gz"),
                       outdir=str(outdir2), threads=2, use_exon=1,
                       backend="python")
    assert res2 == {"rc": 0}
    assert _tree_hashes(outdir) == _tree_hashes(outdir2)

    # rust backend fails clearly; bad backend string raises ValueError
    with pytest.raises(RuntimeError):
        api.spechla(name="X", read1="a", read2="b", outdir="c",
                    backend="rust")
    with pytest.raises(ValueError):
        api.spechla(name="X", read1="a", read2="b", outdir="c",
                    backend="julia")
    with pytest.raises(RuntimeError):
        api.hla_typing(bam_dir="x", ref="hg38", outdir="y", backend="rust")

    # n_threads alias + omitted optionals -> upstream CLI defaults (8 / 1)
    log2 = tmp_path / "api_calls2.log"
    _wire_env(monkeypatch, tools, log2, base_path)
    outdir3 = tmp_path / "api_out3"
    res3 = api.spechla(name="API3", read1=str(tmp_path / "r1.fq.gz"),
                       read2=str(tmp_path / "r2.fq.gz"), outdir=str(outdir3),
                       n_threads=3, spec_hla_root=str(root))
    assert res3 == {"rc": 0}
    toks = log2.read_text().splitlines()
    i = toks.index("CALL SpecHLA_RNAseq.sh")
    rnaseq_args = [t[4:] for t in toks[i + 1:] if t.startswith("ARG ")]
    assert rnaseq_args[:14] == ["-n", "API3", "-1",
                                str(tmp_path / "r1.fq.gz"), "-2",
                                str(tmp_path / "r2.fq.gz"), "-o",
                                str(outdir3), "-j", "3", "-u", "1"]


def test_hla_typing_public_api_e2e(tmp_path, monkeypatch):
    api = _api()
    tools = make_fake_tools(tmp_path)
    base_path = os.environ["PATH"]
    root = make_fake_root(tmp_path, name="api_hroot")
    bam_dir = _make_bam_dir(tmp_path)
    log = tmp_path / "api_hcalls.log"
    _wire_env(monkeypatch, tools, log, base_path)
    outdir = tmp_path / "api_hout"
    res = api.hla_typing(bam_dir=str(bam_dir), ref="hg38",
                         outdir=str(outdir), threads=2, use_exon=1,
                         spec_hla_root=str(root))
    assert res == {"rc": 0}
    assert (outdir / "hla_result_merged.txt").is_file()
    assert (outdir / "ExtractHLAread" / "S1" /
            "S1.ExtractHLAread.done").is_file()
    assert (outdir / "SpecHLA" / "S2" / "S2.SpecHLA.done").is_file()


def test_spec_hla_root_resolution(tmp_path, monkeypatch):
    sp = _sp()
    root = make_fake_root(tmp_path, name="res_root")
    # 1) explicit argument wins over env
    monkeypatch.setenv("SPECHLA_ROOT", str(tmp_path / "env_root"))
    assert sp._resolve_spec_hla_root(str(root), quiet=True) == str(root)
    # 2) env var second
    monkeypatch.setenv("SPECHLA_ROOT", str(root))
    assert sp._resolve_spec_hla_root(None, quiet=True) == str(root)
    # 3) iobrpy package fallback finds a tree with script/whole/
    monkeypatch.delenv("SPECHLA_ROOT")
    resolved = sp._resolve_spec_hla_root(None, quiet=True)
    assert os.path.isdir(os.path.join(resolved, "script", "whole"))
    assert os.path.basename(resolved) == "SpecHLA"
    # missing root -> exit 1 with a clear message
    monkeypatch.setenv("SPECHLA_ROOT", str(tmp_path / "does_not_exist"))
    with pytest.raises(SystemExit) as ei:
        sp._resolve_spec_hla_root(None, quiet=True)
    assert ei.value.code == 1
