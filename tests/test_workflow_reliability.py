"""Failure, dry-run and stale-result regressions using explicitly fake tools."""
import json
from pathlib import Path

import pytest
from test_parity_orchestration import fakebins, _make_pe_tree, _set_calllog, _calls


def inputs(tmp_path, monkeypatch):
    reads, index = tmp_path / "reads", tmp_path / "index"
    _make_pe_tree(str(reads), ["sample"])
    index.mkdir()
    (index / "reference").write_text("reference A")
    log = tmp_path / "calls.jsonl"
    _set_calllog(monkeypatch, str(log))
    return reads, index, log


def test_qc_failure_stops_runall(tmp_path, monkeypatch, fakebins):
    import iobrx
    reads, index, log = inputs(tmp_path, monkeypatch)
    (fakebins / "fastp").write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(7)\n")
    out = tmp_path / "run"
    result = iobrx.runall("salmon", out, reads, threads=1, unknown=["--index", str(index)])
    assert result["rc"] != 0
    assert not (out / "01-qc/.fastq_qc.done").exists()
    assert not list(out.rglob("quant.sf"))
    state = json.loads((out / ".iobrx-run-state.json").read_text())
    assert state["status"] == "failed"


@pytest.mark.parametrize("analysis,product", [("batch_star_count", "sample_ReadsPerGene.out.tab"),
                                               ("batch_salmon", "sample/quant.sf")])
def test_resume_requires_outputs_and_matching_reference(tmp_path, monkeypatch, fakebins, analysis, product):
    import iobrx
    reads, index, log = inputs(tmp_path, monkeypatch)
    out = tmp_path / "run"
    call = getattr(iobrx, analysis)
    call(index, reads, out, n_threads=1, verbose=False)
    count = len(_calls(str(log)))
    assert count == 1
    call(index, reads, out, n_threads=1, verbose=False)
    assert len(_calls(str(log))) == count
    (out / product).unlink()
    call(index, reads, out, n_threads=1, verbose=False)
    assert (out / product).stat().st_size > 0
    assert len(_calls(str(log))) == count + 1
    (index / "reference").write_text("reference B")
    call(index, reads, out, n_threads=1, verbose=False)
    assert len(_calls(str(log))) == count + 2


def test_runall_changed_inputs_refuse_resume(tmp_path, monkeypatch):
    from iobrx._fast import runall_fast as ra
    reads, index, _ = inputs(tmp_path, monkeypatch)
    out = tmp_path / "run"
    args = ["--mode", "salmon", "--outdir", str(out), "--fastq", str(reads), "--index", str(index)]
    calls = []
    def fake_pipeline(argv):
        calls.append(argv)
        (out / "result.csv").write_text("ID,value\nsample,1\n")
        return 0
    monkeypatch.setattr(ra, "_main_impl", fake_pipeline)
    assert ra.runall_argv(args) == 0
    assert ra.runall_argv(args + ["--resume"]) == 0
    (index / "reference").write_text("changed")
    with pytest.raises(ValueError, match="signature|inputs|configuration"):
        ra.runall_argv(args + ["--resume"])
    assert len(calls) == 2


def test_merge_salmon_honors_disabled_native(monkeypatch):
    import iobrx._backend as backend
    from iobrx._fast.merge_salmon_fast import _rust_parse_fn
    monkeypatch.setattr(backend, "_native", None)
    assert _rust_parse_fn(False) is None
    with pytest.raises(RuntimeError):
        _rust_parse_fn(True)


def test_reference_directory_links_are_hashed_without_cycles(tmp_path):
    from iobrx._run_state import inventory
    reference, parts = tmp_path / "reference", tmp_path / "parts"
    reference.mkdir()
    parts.mkdir()
    (parts / "index.dat").write_text("A")
    (reference / "linked").symlink_to(parts, target_is_directory=True)
    (parts / "cycle").symlink_to(reference, target_is_directory=True)
    before = inventory([reference])
    assert any(row["path"].endswith("index.dat") for row in before[0]["files"])
    (parts / "index.dat").write_text("B")
    assert inventory([reference]) != before
