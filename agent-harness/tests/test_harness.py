from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from iobrx_harness import runtime
from iobrx_harness.catalog import CATALOG, request_schema
from conftest import ROOT, request_for


def cli(*args, request=None, cwd=None, env=None):
    process = subprocess.run([sys.executable, "-m", "iobrx_harness", *map(str, args)],
        input=None if request is None else json.dumps(request), cwd=cwd, env=env,
        capture_output=True, text=True, timeout=120)
    return process, json.loads(process.stdout)


def direct(analysis, matrix):
    import iobrx

    if analysis == "anno_eset":
        return iobrx.anno_eset(matrix)
    if analysis == "count2tpm":
        return iobrx.count2tpm(matrix, check_data=True, remove_version=True)
    if analysis.startswith("signature_"):
        return iobrx.calculate_sig_score(matrix, ["signature_collection"], method=analysis[10:], n_threads=2)
    if analysis == "cibersort":
        return iobrx.cibersort(matrix, perm=0, QN=False, n_threads=2)
    if analysis == "epic":
        return iobrx.epic(matrix)
    if analysis == "quantiseq":
        return iobrx.quantiseq(matrix)
    if analysis == "mcpcounter":
        return iobrx.mcpcounter(matrix)
    return iobrx.estimate_score(matrix, platform="rnaseq")


@pytest.mark.parametrize("analysis", list(CATALOG))
def test_all_analyses_match_public_api(analysis, fixtures, tmp_path):
    request = request_for(analysis, fixtures, tmp_path / "result")
    process, manifest = cli("run", "--request", "-", request=request)
    assert process.returncode == 0, (manifest, process.stderr)
    assert manifest["status"] == "completed"
    assert manifest["analysis_seconds"] > 0
    assert manifest["elapsed_seconds"] >= manifest["analysis_seconds"]
    assert len(manifest["input"]["sha256"]) == 64
    assert manifest["environment"]["backend"]["requires_avx512"] is False
    matrix, _ = runtime.load_matrix(request["input"])
    expected = direct(analysis, matrix)
    expected = expected if isinstance(expected, dict) else {"result": expected}
    for name, frame in expected.items():
        observed = pd.read_parquet(tmp_path / f"result/{name}.parquet")
        pd.testing.assert_frame_equal(observed, frame, check_exact=True)
    assert {item["table"] for item in manifest["artifacts"]} == set(expected)
    assert runtime.status(tmp_path / "result", tmp_path)["integrity"]["mismatches"] == []


def test_schema_and_capabilities():
    Draft202012Validator.check_schema(request_schema())
    process, data = cli("capabilities")
    assert process.returncode == 0 and len(data["analyses"]) == 11
    assert data["request_schema"] == request_schema()


@pytest.mark.parametrize("mutation", ["unknown", "missing_scale", "counts_cib", "mouse_signature", "negative_threads", "string_bool", "unsupported_parameter"])
def test_invalid_request_prevents_output(mutation, fixtures, tmp_path):
    request = request_for("cibersort", fixtures, tmp_path / "result")
    if mutation == "unknown":
        request["shell_command"] = "echo unexpected"
    elif mutation == "missing_scale":
        del request["input"]["scale"]
    elif mutation == "counts_cib":
        request["input"]["scale"] = "counts"
    elif mutation == "mouse_signature":
        request["input"]["organism"] = "mmus"
    elif mutation == "negative_threads":
        request["threads"] = -1
    elif mutation == "string_bool":
        request["parameters"]["QN"] = "false"
    else:
        request["parameters"]["seed"] = 10
    process, data = cli("run", "--request", "-", request=request)
    assert process.returncode == 2 and data["status"] == "failed"
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("contents", [
    "gene,A,A\nG1,1,2\n", "gene,A,B\nG1,1,2\nG1,3,4\n", "gene,A,B\nG1,NaN,2\n",
    "gene,A,B\nG1,inf,2\n", "gene,A,B\nG1,-1,2\n", "gene,A,B\nG1,0,2\n",
    "gene,A,B\nG1,hello,2\n", "gene,A,B\n,1,2\n", "gene,A,B\n G1,1,2\n",
])
def test_bad_matrix_has_failed_manifest(contents, fixtures, tmp_path):
    path = tmp_path / "input.csv"
    path.write_text(contents)
    request = request_for("signature_pca", fixtures, tmp_path / "run")
    request["input"]["path"] = str(path)
    request["input"]["scale"] = "linear"
    process, data = cli("run", "--request", "-", request=request)
    assert process.returncode == 2 and data["status"] == "failed"
    assert runtime.read_json(tmp_path / "run/results_manifest.json")["status"] == "failed"
    assert data["artifacts"] == []


def test_orientation_and_literal_ids(tmp_path):
    path = tmp_path / "matrix.tsv"
    path.write_text("sample\t001\tNA\nS1\t1\t2\nS2\t3\t4\n")
    matrix, _ = runtime.load_matrix({"path": str(path), "orientation": "samples_by_genes"})
    assert list(matrix.index) == ["001", "NA"]
    assert list(matrix.columns) == ["S1", "S2"]
    assert matrix.loc["001", "S2"] == 3
    # The index column's title may equal a real sample name; do not let
    # pandas rename the sample to S1.1 while parsing that header.
    path.write_text("S1\tS1\tS2\n001\t1\t2\nNA\t3\t4\n")
    matrix, _ = runtime.load_matrix({"path": str(path), "orientation": "genes_by_samples"})
    assert list(matrix.columns) == ["S1", "S2"]
    assert list(matrix.index) == ["001", "NA"]


def test_cli_errors_are_json():
    process, data = cli("run", "--typo")
    assert process.returncode == 2 and data["status"] == "failed"
    process = subprocess.run([sys.executable, "-m", "iobrx_harness", "run", "--request", "-"],
                             input="{malformed}", text=True, capture_output=True)
    assert process.returncode == 2 and json.loads(process.stdout)["status"] == "failed"


def test_relative_paths_validation_no_writes_and_collision(fixtures, tmp_path):
    request = request_for("signature_pca", fixtures, tmp_path / "result")
    shutil.copyfile(fixtures / "signature.parquet", tmp_path / "input.parquet")
    request["input"]["path"] = "input.parquet"
    request["output_dir"] = "result"
    runtime.write_json(tmp_path / "request.json", request)
    process, data = cli("validate", "--request", tmp_path / "request.json", cwd=fixtures)
    assert process.returncode == 0 and data["status"] == "validated"
    assert not (tmp_path / "result").exists()
    (tmp_path / "result").mkdir()
    (tmp_path / "result/keep.txt").write_text("keep")
    process, data = cli("run", "--request", tmp_path / "request.json", cwd=fixtures)
    assert process.returncode == 4 and data["error"]["code"] == "output_exists"
    assert (tmp_path / "result/keep.txt").read_text() == "keep"


def test_tamper_and_execution_failure(fixtures, tmp_path):
    request = request_for("signature_pca", fixtures, tmp_path / "result")
    _, manifest = cli("run", "--request", "-", request=request)
    (tmp_path / "result/result.csv").write_text("altered")
    process, data = cli("status", tmp_path / "result")
    assert process.returncode == 3 and data["integrity"]["mismatches"] == ["result.csv"]
    request["output_dir"] = str(tmp_path / "failed")
    request["parameters"] = {"signature": ["unknown_signature_group"]}
    process, data = cli("run", "--request", "-", request=request)
    assert process.returncode != 0 and data["status"] == "failed"
    assert data["artifacts"] == []


def test_fallback_and_explicit_rust(fixtures, tmp_path):
    env = {**os.environ, "IOBRX_DISABLE_RUST": "1"}
    request = request_for("signature_pca", fixtures, tmp_path / "fallback")
    process, manifest = cli("run", "--request", "-", request=request, env=env)
    assert process.returncode == 0 and manifest["backend_used"] == "python"
    request["output_dir"] = str(tmp_path / "rust")
    request["parameters"]["backend"] = "rust"
    process, manifest = cli("run", "--request", "-", request=request, env=env)
    assert process.returncode == 3 and manifest["status"] == "failed"


def test_workspace_boundary(fixtures, tmp_path):
    request = request_for("signature_pca", fixtures, tmp_path / "run")
    process, data = cli("validate", "--request", "-", "--workspace", tmp_path, request=request)
    assert process.returncode == 2 and "outside" in data["error"]["message"]
    link = tmp_path / "outside"
    link.symlink_to(fixtures, target_is_directory=True)
    request["input"]["path"] = "outside/signature.parquet"
    process, data = cli("run", "--request", "-", "--workspace", tmp_path, request=request)
    assert process.returncode == 2 and not (tmp_path / "run").exists()


def test_catalog_bundle_relocation_and_installer(fixtures, tmp_path):
    import yaml

    spec = importlib.util.spec_from_file_location("install_omicos", ROOT / "agent-harness/install_omicos.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    destination = tmp_path / "relocated workspace"
    installer.install(destination, dry_run=True)
    assert not destination.exists()
    installer.install(destination)
    skill = destination / "skills/iobrx"
    front = yaml.safe_load((skill / "SKILL.md").read_text().split("---")[1])
    agent = yaml.safe_load((destination / "agents/iobrx_analyst.md").read_text().split("---")[1])
    assert front["id"] == front["name"] == "iobrx"
    assert agent["skills"] == ["iobrx"]
    with pytest.raises(FileExistsError):
        installer.install(destination)
    request = request_for("count2tpm", fixtures, tmp_path / "result")
    process = subprocess.run([sys.executable, str(skill / front["runtime_entrypoint"]), "run", "--request", "-"],
        input=json.dumps(request), cwd=tmp_path, capture_output=True, text=True, timeout=60)
    data = json.loads(process.stdout)
    assert process.returncode == 0 and data["status"] == "completed", (data, process.stderr)
    installer.install(tmp_path / "catalog", layout="catalog")
    assert (tmp_path / "catalog/domains/biology/skills/iobrx/SKILL.md").is_file()
