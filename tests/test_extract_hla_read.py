"""Standalone extraction API contracts; external HLA scripts are test stubs."""
from pathlib import Path
import os

import pytest

import iobrx
from test_parity_hla import make_fake_root, make_fake_tools, _wire_env, _tree_hashes, _normalize_call_log


@pytest.mark.parametrize("reference", ["hg19", "hg38"])
@pytest.mark.parametrize("backend", ["auto", "python"])
def test_extraction_matches_upstream_without_typing(tmp_path, monkeypatch, reference, backend):
    from iobrpy.SpecHLA import extract_hla_read as upstream

    tools = make_fake_tools(tmp_path)
    base_path = os.environ["PATH"]
    bam = tmp_path / "sample with spaces.bam"
    bam.write_bytes(b"stub alignment")
    results = {}
    for arm in ("original", "iobrx"):
        root = make_fake_root(tmp_path, name=f"assets_{arm}")
        output, log = tmp_path / f"out {arm}", tmp_path / f"{arm}.log"
        _wire_env(monkeypatch, tools, log, base_path)
        monkeypatch.setattr(upstream, "__file__", str(root / "extract_hla_read.py"))
        if arm == "original":
            upstream.main(["-s", "S1", "-b", str(bam), "-r", reference, "-o", str(output), "--no-auto-install"])
        else:
            result = iobrx.extract_hla_read("S1", bam, reference, output, spec_hla_root=root, backend=backend)
            assert result == {"rc": 0}
        results[arm] = (_tree_hashes(output), _normalize_call_log(log, {str(root): "<ROOT>", str(output): "<OUT>"}))
    assert results["original"] == results["iobrx"]
    files, commands = results["iobrx"]
    assert files and any(name.endswith(".fq.gz") for name in files)
    assert any("ExtractHLAread.sh" in line for line in commands)
    assert not any("SpecHLA_RNAseq.sh" in line for line in commands)


@pytest.mark.parametrize("backend", ["auto", "python"])
@pytest.mark.parametrize("allow_install", [False, True])
def test_dependency_installation_is_explicit(tmp_path, monkeypatch, backend, allow_install):
    from iobrx._fast import hla_typing_fast as local
    from iobrpy.SpecHLA import extract_hla_read as upstream

    module = local if backend == "auto" else upstream
    seen = []
    monkeypatch.setattr(module, "ensure_dependencies", lambda auto_install: seen.append(auto_install))
    monkeypatch.setattr(module, "run_extraction", lambda *args, **kwargs: None)
    options = {"auto_install": True} if allow_install else {}
    assert iobrx.extract_hla_read("S1", tmp_path / "input.bam", "hg38", tmp_path / "out", backend=backend, **options) == {"rc": 0}
    assert seen == [allow_install]


@pytest.mark.parametrize("backend", ["auto", "python"])
def test_invalid_arguments_and_missing_dependencies_do_not_write(tmp_path, monkeypatch, backend):
    from iobrx._fast import hla_typing_fast as local
    from iobrpy.SpecHLA import extract_hla_read as upstream

    output = tmp_path / "out"
    assert iobrx.extract_hla_read(outdir=output, backend=backend)["rc"] == 2
    assert iobrx.extract_hla_read("S1", "input.bam", "unknown", output, backend=backend)["rc"] == 2
    def missing(auto_install):
        assert auto_install is False
        raise RuntimeError("samtools is unavailable")
    monkeypatch.setattr(local if backend == "auto" else upstream, "ensure_dependencies", missing)
    assert iobrx.extract_hla_read("S1", "input.bam", "hg38", output, backend=backend)["rc"] == 2
    assert not output.exists()


@pytest.mark.parametrize("backend", ["auto", "python"])
def test_extraction_failure_returns_nonzero(tmp_path, monkeypatch, backend):
    from iobrpy.SpecHLA import extract_hla_read as upstream

    tools = make_fake_tools(tmp_path)
    root = make_fake_root(tmp_path)
    _wire_env(monkeypatch, tools, tmp_path / "calls.log", os.environ["PATH"])
    monkeypatch.setattr(upstream, "__file__", str(root / "extract_hla_read.py"))
    (root / "script/ExtractHLAread.sh").write_text("#!/usr/bin/env bash\nexit 7\n")
    assert iobrx.extract_hla_read("S1", "input.bam", "hg38", tmp_path / "out", spec_hla_root=root, backend=backend) == {"rc": 7}


def test_extraction_export_and_invalid_backend():
    assert "extract_hla_read" in iobrx.__all__
    with pytest.raises(RuntimeError, match="no Rust kernel"):
        iobrx.extract_hla_read(backend="rust")
    with pytest.raises(ValueError, match="backend"):
        iobrx.extract_hla_read(backend="typo")
