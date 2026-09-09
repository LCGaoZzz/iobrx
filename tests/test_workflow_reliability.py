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
    def fake_pipeline(argv, **kwargs):
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


def stub_pipeline(tmp_path, monkeypatch, mode="salmon"):
    """Exercise the real scheduler with controllable table-writer failures."""
    from collections import Counter
    from iobrx._fast import runall_fast as ra

    reads, index, _ = inputs(tmp_path, monkeypatch)
    out = tmp_path / "run"
    args = ["--mode", mode, "--outdir", str(out), "--fastq", str(reads),
            "--index", str(index), "--threads", "1"]
    calls = Counter()
    failure = {"step": None, "remaining": 0, "kind": "exit"}

    def fake_step(cmd, cwd=None, dry=False):
        assert not dry
        step = cmd[1]
        calls[step] += 1
        if step == "fastq_qc":
            for mate in (1, 2):
                (out / f"01-qc/sample_{mate}.fastq.gz").write_bytes(b"cleaned reads")
        if step == "batch_salmon":
            sample = out / "02-salmon/sample"
            sample.mkdir(exist_ok=True)
            (sample / "quant.sf").write_text("quantification fixture")
        if step == "batch_star_count":
            (out / "02-star/sample.bam").write_bytes(b"alignment fixture")
            (out / "02-star/sample_ReadsPerGene.out.tab").write_text("G1\t1\t1\t1\n")
        if step == "trust4":
            (out / "07-TCRBCR/sample_report.tsv").write_text("repertoire fixture")
            (out / "07-TCRBCR/trust4_immdata.csv").write_text("ID,value\nsample,1\n")
        if step == "merge_salmon":
            (out / "02-salmon/runall_salmon_tpm.tsv").write_text("ID\tsample\nG1\t1\n")
        if step == "merge_star_count":
            import gzip
            with gzip.open(out / "02-star/runall.STAR.count.tsv.gz", "wt") as handle:
                handle.write("ID\tsample\nG1\t1\n")
        output = None
        for flag in ("--output", "-o"):
            if flag in cmd and step != "trust4":
                output = Path(cmd[cmd.index(flag) + 1])
                break
        if step == failure["step"] and failure["remaining"]:
            failure["remaining"] -= 1
            if failure["kind"] == "missing":
                return 0
            output.write_text("ID,unfinished_score\nsample,")
            if failure["kind"] == "exception":
                raise OSError("simulated interrupted write")
            return 7
        if output is not None:
            output.write_text(f"ID,{step}\nsample,1\n")
        return 0

    monkeypatch.setattr(ra, "_run", fake_step)
    return ra, args, out, calls, failure


@pytest.mark.parametrize("mode,step", [
    ("salmon", "prepare_salmon"), ("salmon", "log2_eset"),
    ("salmon", "calculate_sig_score"), ("salmon", "cibersort"),
    ("salmon", "IPS"), ("salmon", "estimate"), ("salmon", "mcpcounter"),
    ("salmon", "quantiseq"), ("salmon", "epic"), ("salmon", "LR_cal"),
    ("star", "count2tpm"), ("star", "log2_eset"),
])
def test_runall_retries_partial_tables_and_reuses_successes(tmp_path, monkeypatch, mode, step):
    ra, args, out, calls, failure = stub_pipeline(tmp_path, monkeypatch, mode)
    failure.update(step=step, remaining=2)
    for attempt in (1, 2):
        assert ra.runall_argv(args + (["--resume"] if attempt == 2 else [])) == 7
        state = json.loads((out / ".iobrx-run-state.json").read_text())
        assert state["status"] == "failed"
        assert step not in state["completed_steps"]
        assert calls[step] == attempt
    assert ra.runall_argv(args + ["--resume"]) == 0
    state = json.loads((out / ".iobrx-run-state.json").read_text())
    assert state["status"] == "completed"
    assert step in state["completed_steps"]
    assert calls[step] == 3
    assert all(count == 1 for name, count in calls.items() if name != step)
    assert not any("unfinished_score" in p.read_text() for p in out.rglob("*.csv"))
    before = calls.copy()
    assert ra.runall_argv(args + ["--resume"]) == 0
    assert calls == before


@pytest.mark.parametrize("kind,error", [("exception", OSError), ("missing", RuntimeError)])
def test_runall_failed_or_missing_table_is_not_checkpointed(tmp_path, monkeypatch, kind, error):
    ra, args, out, calls, failure = stub_pipeline(tmp_path, monkeypatch)
    failure.update(step="calculate_sig_score", remaining=1, kind=kind)
    with pytest.raises(error):
        ra.runall_argv(args)
    state = json.loads((out / ".iobrx-run-state.json").read_text())
    assert state["status"] == "failed"
    assert "calculate_sig_score" not in state["completed_steps"]
    assert ra.runall_argv(args + ["--resume"]) == 0
    assert calls["calculate_sig_score"] == 2
    assert calls["prepare_salmon"] == 1


def test_runall_retries_partial_deconvolution_merge(tmp_path, monkeypatch):
    ra, args, out, calls, _ = stub_pipeline(tmp_path, monkeypatch)
    original = ra.pd.DataFrame.to_csv
    attempts = []

    def interrupted_merge(frame, path, *args, **kwargs):
        if Path(path).name == "deconvo_merged.csv":
            attempts.append(path)
            if len(attempts) == 1:
                Path(path).write_text("ID,unfinished_score\nsample,")
                raise OSError("simulated interrupted merge")
        return original(frame, path, *args, **kwargs)

    monkeypatch.setattr(ra.pd.DataFrame, "to_csv", interrupted_merge)
    with pytest.raises(OSError):
        ra.runall_argv(args)
    assert ra.runall_argv(args + ["--resume"]) == 0
    assert len(attempts) == 2
    assert all(count == 1 for count in calls.values())
    merged = ra.pd.read_csv(out / "05-tme/deconvo_merged.csv")
    assert len(merged) == 1 and "unfinished_score" not in merged.columns


def test_runall_fresh_attempt_invalidates_old_successes(tmp_path, monkeypatch):
    ra, args, out, calls, failure = stub_pipeline(tmp_path, monkeypatch)
    assert ra.runall_argv(args) == 0
    failure.update(step="calculate_sig_score", remaining=1)
    assert ra.runall_argv(args) == 7
    state = json.loads((out / ".iobrx-run-state.json").read_text())
    assert "cibersort" not in state["completed_steps"]
    assert ra.runall_argv(args + ["--resume"]) == 0
    assert calls["cibersort"] == 2
    assert calls["fastq_qc"] == 2


def test_runall_rejects_legacy_state_without_step_records(tmp_path, monkeypatch):
    ra, args, out, calls, _ = stub_pipeline(tmp_path, monkeypatch)
    assert ra.runall_argv(args) == 0
    path = out / ".iobrx-run-state.json"
    state = json.loads(path.read_text())
    del state["schema_version"]
    del state["completed_steps"]
    path.write_text(json.dumps(state))
    before = calls.copy()
    with pytest.raises(ValueError, match="per-step completion records"):
        ra.runall_argv(args + ["--resume"])
    assert calls == before


@pytest.mark.parametrize("mode", ["salmon", "star"])
def test_runall_notes_and_figures_do_not_affect_resume(tmp_path, monkeypatch, mode):
    ra, args, out, calls, _ = stub_pipeline(tmp_path, monkeypatch, mode)
    assert ra.runall_argv(args) == 0
    before = calls.copy()
    extras = [out / "notes.md", out / "05-tme/heatmap.svg", out / "04-signatures/plot-data.csv"]
    original_hash = ra.file_hash
    def checked_hash(path):
        assert Path(path) not in extras, "Unrelated artifacts must not be hashed"
        return original_hash(path)
    monkeypatch.setattr(ra, "file_hash", checked_hash)
    for contents in ("first analysis notes", "updated interpretation"):
        for path in extras:
            path.write_text(contents)
        assert ra.runall_argv(args + ["--resume"]) == 0
        assert calls == before
    for path in extras:
        path.unlink()
    assert ra.runall_argv(args + ["--resume"]) == 0
    assert calls == before


def test_runall_reads_old_schema2_without_tracking_old_notes(tmp_path, monkeypatch):
    ra, args, out, calls, _ = stub_pipeline(tmp_path, monkeypatch)
    assert ra.runall_argv(args) == 0
    notes = out / "notes.md"
    notes.write_text("old notes")
    path = out / ".iobrx-run-state.json"
    state = json.loads(path.read_text())
    state["outputs"]["notes.md"] = ra.file_hash(notes)  # Previous whole-tree format.
    path.write_text(json.dumps(state))
    notes.write_text("new notes")
    before = calls.copy()
    assert ra.runall_argv(args + ["--resume"]) == 0
    assert calls == before
    assert "notes.md" not in json.loads(path.read_text())["outputs"]


@pytest.mark.parametrize("mode,relative", [
    ("salmon", "01-qc/sample_1.fastq.gz"),
    ("salmon", "02-salmon/sample/quant.sf"),
    ("salmon", "04-signatures/calculate_sig_score.csv"),
    ("salmon", "05-tme/deconvo_merged.csv"),
    ("salmon", "07-TCRBCR/trust4_immdata.csv"),
    ("salmon", "01-qc/.fastq_qc.done"),
    ("star", "02-star/sample.bam"),
    ("star", "02-star/sample_ReadsPerGene.out.tab"),
])
@pytest.mark.parametrize("change", ["edit", "delete"])
def test_runall_still_protects_recorded_products(tmp_path, monkeypatch, mode, relative, change):
    ra, args, out, calls, _ = stub_pipeline(tmp_path, monkeypatch, mode)
    assert ra.runall_argv(args) == 0
    product = out / relative
    if change == "edit":
        product.write_text("altered calculation product")
    else:
        product.unlink()
    before = calls.copy()
    with pytest.raises(ValueError, match="outputs changed"):
        ra.runall_argv(args + ["--resume"])
    assert calls == before


def test_runall_additional_quantification_cannot_change_sample_set(tmp_path, monkeypatch):
    ra, args, out, calls, _ = stub_pipeline(tmp_path, monkeypatch)
    assert ra.runall_argv(args) == 0
    extra = out / "02-salmon/unrequested-sample"
    extra.mkdir()
    (extra / "quant.sf").write_text("new sample")
    before = calls.copy()
    with pytest.raises(ValueError, match="outputs changed"):
        ra.runall_argv(args + ["--resume"])
    assert calls == before


def test_runall_tracks_custom_read_names_from_tool_records(tmp_path, monkeypatch):
    ra, args, out, calls, _ = stub_pipeline(tmp_path, monkeypatch)
    original = ra._run
    def with_custom_output(cmd, **kwargs):
        rc = original(cmd, **kwargs)
        if cmd[1] == "fastq_qc":
            custom = out / "01-qc/sample.custom"
            custom.write_bytes(b"custom-suffix reads")
            record = out / "01-qc/sample.task.complete.iobrx.json"
            record.write_text(json.dumps({"outputs": [{"path": str(custom), "sha256": ra.file_hash(custom)}]}))
        return rc
    monkeypatch.setattr(ra, "_run", with_custom_output)
    assert ra.runall_argv(args) == 0
    (out / "01-qc/sample.custom").unlink()
    with pytest.raises(ValueError, match="outputs changed"):
        ra.runall_argv(args + ["--resume"])
