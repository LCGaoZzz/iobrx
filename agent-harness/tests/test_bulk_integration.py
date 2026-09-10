"""Bulk IO and cohort table contracts; no solver mocks presented as benchmarks."""
import importlib.util
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from iobrx_harness import runtime
from iobrx_harness.catalog import request_schema
from iobrx_harness.cohort_tables import align_samples, combine_cohorts
from iobrx_harness.h5ad import _local_node


def spec(path, matrix="X", **options):
    return {"path": str(path), "orientation": "samples_by_genes", "scale": "tpm",
            "gene_id": "symbol", "organism": "hsa", "h5ad": {"matrix": matrix, **options}}


def test_h5ad_schema_and_extended_expression_adapter():
    for analysis in ("cibersort", "epic", "estimate_score", "ips"):
        req = {"schema_version": "1.0", "analysis": analysis, "threads": 1,
               "input": spec("bulk.h5ad", "layers/tpm"), "output_dir": "result"}
        if analysis == "estimate_score":
            req["input"]["scale"] = "log2p1"
        validator = Draft202012Validator(request_schema(analysis))
        assert not list(validator.iter_errors(req))
        for options in ({}, {"matrix": "raw"}, {"matrix": "layers/a/b"},
                        {"matrix": "X", "max_dense_bytes": 0}, {"matrix": "X", "unknown": True}):
            bad = deepcopy(req)
            bad["input"]["h5ad"] = options
            assert list(validator.iter_errors(bad))


def test_defaults_reuse_library_thread_policy(tmp_path, monkeypatch):
    import iobrx
    monkeypatch.setattr(iobrx, "get_threads", lambda: 2)
    req = {"schema_version": "1.0", "analysis": "epic", "input": spec("bulk.h5ad"),
           "output_dir": "run"}
    assert runtime.normalize_request(req, tmp_path)["threads"] == 2
    req["threads"] = 16
    assert runtime.normalize_request(req, tmp_path)["threads"] == 16


@pytest.mark.parametrize("options,orientation,message", [
    (None, "samples_by_genes", "requires input.h5ad.matrix"),
    ({"matrix": "X"}, "genes_by_samples", "obs x var"),
    ({"matrix": "layers/a/b"}, "samples_by_genes", "must be X"),
    ({"matrix": "X", "max_dense_bytes": False}, "samples_by_genes", "positive integer"),
])
def test_semantic_contract_before_optional_import(tmp_path, options, orientation, message):
    path = tmp_path / "bulk.h5ad"
    path.touch()
    inp = spec(path)
    inp.update(h5ad=options, orientation=orientation)
    with pytest.raises(runtime.HarnessError, match=message):
        runtime.load_matrix(inp)


def test_missing_anndata_is_actionable_and_tables_remain_independent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "anndata.io", None)
    path = tmp_path / "bulk.h5ad"
    path.touch()
    with pytest.raises(runtime.HarnessError, match="optional anndata") as error:
        runtime.load_matrix(spec(path))
    assert error.value.code == "environment_error" and error.value.exit_code == 3
    table = tmp_path / "bulk.tsv"
    table.write_text("sample\tG1\tG2\n001\t1\t2\nNA\t3\t4\n")
    inp = spec(table)
    del inp["h5ad"]
    result, _ = runtime.load_matrix(inp)
    assert result.columns.tolist() == ["001", "NA"]
    with pytest.raises(runtime.HarnessError, match="require a .h5ad"):
        runtime.load_matrix(spec(table))


@pytest.mark.parametrize("link_type", ["soft", "external", "virtual", "external_storage", "nested"])
def test_hdf5_cannot_reach_outside_file(tmp_path, link_type):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "links.h5ad", "w") as handle:
        if link_type == "soft":
            handle.create_dataset("source", data=np.ones((2, 2)))
            handle["X"] = h5py.SoftLink("/source")
        elif link_type == "external":
            handle["X"] = h5py.ExternalLink("outside.h5ad", "/X")
        elif link_type == "virtual":
            layout = h5py.VirtualLayout(shape=(2, 2), dtype="f8")
            layout[:] = h5py.VirtualSource("outside.h5ad", "X", shape=(2, 2))
            handle.create_virtual_dataset("X", layout)
        elif link_type == "external_storage":
            handle.create_dataset("X", shape=(2, 2), dtype="f8", external=[("outside.bin", 0, 32)])
        else:
            group = handle.create_group("X")
            group["data"] = h5py.ExternalLink("outside.h5ad", "/X")
        with pytest.raises(runtime.HarnessError, match="self-contained"):
            _local_node(handle, "X", h5py)


@pytest.fixture
def adata():
    if importlib.util.find_spec("anndata") is None:
        pytest.skip("optional anndata is not installed")
    import anndata as ad
    full = ad.AnnData(np.arange(1., 9.).reshape(2, 4),
                     obs=pd.DataFrame({"sample_id": ["S1", "S2"]}, index=["001", "NA"]),
                     var=pd.DataFrame({"symbol": ["A", "B", "C", "D"]},
                                      index=["g1", "g2", "g3", "g4"]))
    data = full[:, [0, 2]].copy()
    data.raw = full
    data.layers["counts"] = data.X * 10
    return data


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
@pytest.mark.parametrize("slot", ["X", "layers/counts", "raw.X"])
def test_real_h5ad_values_ids_and_raw_axis(tmp_path, adata, storage, slot):
    from scipy import sparse
    values = adata.raw.X if slot == "raw.X" else adata.layers["counts"] if slot.startswith("layers/") else adata.X
    genes = adata.raw.var_names if slot == "raw.X" else adata.var_names
    expected = pd.DataFrame(np.asarray(values).T, index=genes, columns=adata.obs_names)
    if storage != "dense":
        convert = sparse.csr_matrix if storage == "csr" else sparse.csc_matrix
        if slot == "raw.X":
            raw = adata.raw.to_adata()
            raw.X = convert(raw.X)
            adata.raw = raw
        elif slot.startswith("layers/"):
            adata.layers["counts"] = convert(values)
        else:
            adata.X = convert(values)
    path = tmp_path / "bulk.h5ad"
    adata.write_h5ad(path)
    before = path.read_bytes()
    actual, info = runtime.load_matrix(spec(path, slot), "sha256")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert info["h5ad"]["dense_bytes"] == expected.size * 8
    assert info["h5ad"]["var"] == ("raw/var" if slot == "raw.X" else "var")
    assert path.read_bytes() == before
    csv = tmp_path / "same.csv"
    expected.to_csv(csv)
    table_input = {k: v for k, v in spec(csv).items() if k != "h5ad"}
    table_input["orientation"] = "genes_by_samples"
    roundtrip, _ = runtime.load_matrix(table_input)
    pd.testing.assert_frame_equal(actual, roundtrip, check_exact=True)


def test_real_h5ad_metadata_ids_and_no_unused_slot_reads(tmp_path, adata, monkeypatch):
    import anndata.io
    seen = []
    original = anndata.io.read_elem
    def read(node):
        seen.append(node.name)
        return original(node)
    monkeypatch.setattr(anndata.io, "read_elem", read)
    path = tmp_path / "bulk.h5ad"
    adata.write_h5ad(path)
    table, _ = runtime.load_matrix(spec(path, "raw.X", gene_column="symbol", sample_column="sample_id"))
    assert table.index.tolist() == ["A", "B", "C", "D"]
    assert table.columns.tolist() == ["S1", "S2"]
    assert set(seen) == {"/obs", "/raw/var", "/raw/X"}


@pytest.mark.parametrize("problem,message", [
    ("budget", "above max_dense_bytes"), ("missing_layer", "Missing h5ad element"),
    ("missing_column", "gene_column not found"), ("duplicate", "Duplicate sample"),
    ("nan", "NaN or infinity"), ("negative", "nonnegative"),
])
def test_real_h5ad_rejections(tmp_path, adata, problem, message):
    path = tmp_path / "bulk.h5ad"
    inp = spec(path)
    if problem == "budget":
        inp["h5ad"]["max_dense_bytes"] = 1
    elif problem == "missing_layer":
        inp["h5ad"]["matrix"] = "layers/missing"
    elif problem == "missing_column":
        inp["h5ad"]["gene_column"] = "missing"
    elif problem == "duplicate":
        adata.obs["sample_id"] = ["same", "same"]
        inp["h5ad"]["sample_column"] = "sample_id"
    elif problem == "nan":
        adata.X[0, 0] = np.nan
    else:
        adata.X[0, 0] = -1
    adata.write_h5ad(path)
    with pytest.raises(runtime.HarnessError, match=message):
        runtime.load_matrix(inp)


def test_alignment_reorders_by_id_without_mutating_or_merging_patients():
    table = pd.DataFrame({"ID": ["NA", "001"], "score": [2., 1.]})
    original = table.copy(deep=True)
    result = align_samples(table, ["001", "NA"], sample_column="ID")
    assert result.index.tolist() == ["001", "NA"] and result.score.tolist() == [1., 2.]
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("actual,expected", [
    (["S1"], ["S1", "S2"]), (["S1", "S2"], ["S1"]),
    (["S1", "S1"], ["S1"]), (["S1"], ["S1", "S1"]),
    ([" S1"], ["S1"]), ([None], ["S1"]), (["S1"], []),
    (["S1"], ["patient1"]),
])
def test_alignment_rejects_ambiguous_or_changed_sample_sets(actual, expected):
    with pytest.raises(ValueError):
        align_samples(pd.DataFrame({"score": range(len(actual))}, index=actual), expected)


def test_same_method_cohorts_keep_colliding_ids_separate():
    a = pd.DataFrame({"score": [1., 2.]}, index=["001", "NA"])
    b = pd.DataFrame({"score": [3., 4.]}, index=["001", "NA"])
    merged = combine_cohorts({"cohort-A": a, "cohort-B": b})
    assert merged.shape == (4, 1) and merged.index.is_unique
    assert merged.loc[("cohort-A", "001"), "score"] == 1.
    assert merged.loc[("cohort-B", "001"), "score"] == 3.
    with pytest.raises(ValueError, match="columns differ"):
        combine_cohorts({"A": a, "B": b.rename(columns={"score": "different"})})


@pytest.mark.parametrize("table", [pd.DataFrame(), pd.DataFrame(index=["S1"]), [1, 2]])
def test_alignment_rejects_empty_or_nontabular_results(table):
    with pytest.raises(ValueError, match="nonempty DataFrame"):
        align_samples(table, ["S1"])


def test_alignment_rejects_hierarchical_ids():
    table = pd.DataFrame({"score": [1.]}, index=["S1"])
    with pytest.raises(ValueError, match="MultiIndex"):
        align_samples(table, [("S1", "case1")])
