"""New adapters: numerical calls, typed inputs, isolation and tool failures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iobrx_harness import runtime
from iobrx_harness.catalog import CATALOG


def request(analysis, path, output, **fields):
    return {"schema_version": "1.0", "analysis": analysis, "threads": 1,
            "input": {"path": str(path), **fields}, "output_dir": str(output), "parameters": {}}


@pytest.mark.parametrize("analysis", ["ips", "lr_cal", "log2_eset"])
def test_matrix_adapters_match_api(analysis, fixtures, tmp_path):
    import iobrx
    source = fixtures / "tpm.parquet"
    req = request(analysis, source, tmp_path / "run", orientation="genes_by_samples", scale="tpm", gene_id="symbol", organism="hsa")
    manifest = runtime.run(req, tmp_path)
    assert manifest["status"] == "completed", manifest
    actual = pd.read_parquet(tmp_path / "run/result.parquet")
    data = pd.read_parquet(source)
    if analysis == "log2_eset":
        canonical = tmp_path / "input.csv"
        data.to_csv(canonical)
        expected = iobrx.log2_eset(canonical, tmp_path / "expected.csv")
    elif analysis == "lr_cal":
        expected = iobrx.lr_cal(data, data_type="tpm", id_type="symbol")
    else:
        expected = iobrx.ips(data)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert runtime.status(tmp_path / "run", tmp_path)["integrity"]["mismatches"] == []


@pytest.mark.parametrize("analysis", ["nmf", "tme_cluster"])
def test_feature_table_adapters(analysis, tmp_path):
    import iobrx
    source = tmp_path / "features.parquet"
    data = pd.DataFrame(np.random.default_rng(17).gamma(2, 3, (16, 8)),
                        index=[f"S{i}" for i in range(16)], columns=[f"F{i}" for i in range(8)])
    data.to_parquet(source)
    req = request(analysis, source, tmp_path / "run", kind="feature_table", orientation="samples_by_features", scale="linear")
    req["parameters"] = {"kmin": 2, "kmax": 3} if analysis == "nmf" else {"min_nc": 2, "max_nc": 3}
    manifest = runtime.run(req, tmp_path)
    assert manifest["status"] == "completed", manifest
    if analysis == "nmf":
        expected = iobrx.nmf(data, kmin=2, kmax=3, n_threads=1)
        # Parquet/numeric validation can change array memory order and BLAS rounding.
        np.testing.assert_allclose(pd.read_parquet(tmp_path / "run/W.parquet").to_numpy(), expected["W"], rtol=1e-12, atol=1e-14)
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "run/top_features.parquet"), expected["top_features"])
    else:
        expected = iobrx.tme_cluster(data.rename_axis("ID").reset_index(), id="ID", min_nc=2, max_nc=3, n_threads=1)
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "run/result.parquet"), expected, check_exact=True)


def test_salmon_merge_stages_inputs(tmp_path):
    source = tmp_path / "salmon"
    for sample, tpm in [("S1", 4), ("S2", 6)]:
        directory = source / sample
        directory.mkdir(parents=True)
        (directory / "quant.sf").write_text(f"Name\tLength\tEffectiveLength\tTPM\tNumReads\nTX1\t100\t80\t{tpm}\t8\n")
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    manifest = runtime.run(request("merge_salmon", source, tmp_path / "run", kind="salmon_directory"), tmp_path)
    assert manifest["status"] == "completed", manifest
    assert {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    result = pd.read_parquet(tmp_path / "run/tpm.parquet")
    assert sorted(result.iloc[0].tolist()) == [4, 6]


def test_nested_symlink_cannot_escape_workspace(tmp_path):
    scope = tmp_path / "workspace"
    source = scope / "reads"
    source.mkdir(parents=True)
    outside = tmp_path / "outside.fastq.gz"
    outside.write_bytes(b"private input")
    (source / "S_1.fastq.gz").symlink_to(outside)
    with pytest.raises(runtime.HarnessError, match="outside"):
        runtime.normalize_request(request("fastq_qc", source, scope / "run", kind="fastq_directory"), scope, scope)


def test_directory_symlink_chain_cannot_escape_workspace(tmp_path):
    scope, outside = tmp_path / "workspace", tmp_path / "outside"
    source, intermediate = scope / "reads", scope / "linked"
    source.mkdir(parents=True)
    intermediate.mkdir()
    outside.mkdir()
    (source / "first").symlink_to(intermediate, target_is_directory=True)
    (intermediate / "second").symlink_to(outside, target_is_directory=True)
    with pytest.raises(runtime.HarnessError, match="outside"):
        runtime.normalize_request(request("fastq_qc", source, scope / "run", kind="fastq_directory"), scope, scope)


def test_outputs_cannot_be_inside_input(tmp_path):
    source = tmp_path / "reads"
    source.mkdir()
    with pytest.raises(runtime.HarnessError, match="outside every input"):
        runtime.normalize_request(request("fastq_qc", source, source / "run", kind="fastq_directory"), tmp_path)


def test_missing_pair_fails_validation(tmp_path):
    source = tmp_path / "reads"
    source.mkdir()
    (source / "S_1.fastq.gz").write_bytes(b"read")
    with pytest.raises(runtime.HarnessError, match="mate"):
        runtime.validate(request("fastq_qc", source, tmp_path / "run", kind="fastq_directory"), tmp_path)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("rc,produce", [(7, True), (0, False)])
def test_external_failure_or_missing_output_not_completed(tmp_path, monkeypatch, rc, produce):
    import iobrx
    from iobrx_harness import extended_runtime
    source, index = tmp_path / "reads", tmp_path / "index"
    source.mkdir()
    index.mkdir()
    for mate in (1, 2):
        (source / f"S_{mate}.fastq.gz").write_bytes(b"read")
    executable = tmp_path / "salmon"
    executable.write_text("fake salmon for adapter contract only")
    monkeypatch.setattr(extended_runtime.shutil, "which", lambda name: str(executable))
    def fake(index, path, out, **kwargs):
        if produce:
            (Path(out) / "quant.sf").write_text("partial output")
        return {"rc": rc}
    monkeypatch.setattr(iobrx, "batch_salmon", fake)
    manifest = runtime.run(request("batch_salmon", source, tmp_path / "run", kind="fastq_directory", index=str(index)), tmp_path)
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] in {"tool_failed", "invalid_result"}


def test_bayesprism_counts_and_orientation(tmp_path, monkeypatch):
    import iobrx
    source = tmp_path / "counts.parquet"
    frame = pd.DataFrame([[1, 2], [3, 4]], index=["G1", "G2"], columns=["S1", "S2"])
    frame.to_parquet(source)
    req = request("bayesprism", source, tmp_path / "run", orientation="genes_by_samples", scale="counts", gene_id="symbol", organism="hsa")
    def fake(bulk, **kwargs):
        pd.testing.assert_frame_equal(bulk, frame.astype(float))
        return {"theta": pd.DataFrame({"type": [0.4, 0.6]}, index=frame.columns)}
    monkeypatch.setattr(iobrx, "bayesprism", fake)
    assert runtime.run(req, tmp_path)["status"] == "completed"
    frame.iloc[0, 0] = 1.5
    frame.to_parquet(source)
    with pytest.raises(runtime.HarnessError, match="integer"):
        runtime.validate(req, tmp_path)


def test_raw_command_passthrough_rejected(tmp_path):
    req = request("runall", tmp_path, tmp_path / "run", kind="fastq_directory", index="idx", f="f.fa", ref="r.fa")
    req["parameters"]["unknown"] = ["--outdir", "/other"]
    with pytest.raises(runtime.HarnessError, match="unknown"):
        runtime.normalize_request(req, tmp_path)


def test_profile_rnaseq_defaults_and_result_files(tmp_path, fixtures, monkeypatch):
    import iobrx
    def fake(path, output, **kwargs):
        assert kwargs["QN"] is False and kwargs["arrays"] is False
        assert kwargs["platform"] == "rnaseq"
        frame = pd.read_csv(path, index_col=0)
        assert frame.shape[1] == 3
        target = Path(output)
        target.mkdir()
        result = target / "cibersort.csv"
        pd.DataFrame({"fraction": [0.4, 0.6, 0.5]}, index=frame.columns).to_csv(result)
        return {"cibersort": str(result)}
    monkeypatch.setattr(iobrx, "tme_profile", fake)
    req = request("tme_profile", fixtures / "tpm.parquet", tmp_path / "run", orientation="genes_by_samples", scale="tpm", gene_id="symbol", organism="hsa")
    manifest = runtime.run(req, tmp_path)
    assert manifest["status"] == "completed", manifest
    assert (tmp_path / "run/cibersort.parquet").exists()


def test_zero_feature_and_changed_source(tmp_path, monkeypatch):
    import iobrx
    data = pd.DataFrame(np.arange(32).reshape(8, 4) + 1., index=[f"S{i}" for i in range(8)], columns=list("ABCD"))
    data["D"] = 0.
    source = tmp_path / "features.parquet"
    data.to_parquet(source)
    req = request("tme_cluster", source, tmp_path / "run", kind="feature_table", orientation="samples_by_features", scale="linear")
    req["parameters"] = {"min_nc": 2, "max_nc": 3}
    assert runtime.validate(req, tmp_path)["status"] == "validated"
    def fake(frame, **kwargs):
        data.assign(A=999.).to_parquet(source)
        return frame
    monkeypatch.setattr(iobrx, "tme_cluster", fake)
    manifest = runtime.run(req, tmp_path)
    assert manifest["status"] == "failed" and manifest["error"]["code"] == "input_changed"
