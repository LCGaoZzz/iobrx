"""LR_cal parity: synthetic-input gates against the ORIGINAL iobrpy workflow.

Fast gates (no official data download needed): the accelerated
``iobrx.lr_cal`` path must be bit-identical to ``iobrpy.workflow.LR_cal`` on
small synthetic matrices that exercise every branch:

* the feature_manipulation filters — NaN rows, ±inf rows, zero-variance rows
  (all-zero AND constant-nonzero, whose ``std() == 0`` decision hinges on the
  exact pandas two-pass/numpy-pairwise summation order replicated by the Rust
  kernel ``_rust.lr_gene_valid_mask``);
* skipna pair-min with double-logged NaNs (values < -1 on log2-scale input);
* absent-gene all-NaN pair columns and the final any-NaN column drop;
* group_lrpairs merging (combo columns replace their member pairs);
* single-sample input (the ORIGINAL skips the zero-variance filter when the
  transposed frame has <= 1 row);
* int64 frames (the ORIGINAL casts to float64 inside nanvar);
* guard fallbacks: object dtype columns and duplicate gene labels must route
  to the ORIGINAL compute_LR_pairs and still agree;
* the no-.so numpy fallback engine;
* the bug-compatible ``data_type='count'`` TypeError of iobrpy 0.2.0
  (count2tpm called without its two required positional annotations);
* an invalid cancer_type raising KeyError, as upstream.

The official-data byte gate (stad TPM 54,658x10 -> 10x773, output CSV sha256
``a49ddff5...`` and the three cross inputs) lives in the campaign scratch
(exp4_e2e.py) and the blind benchmark pass; it needs the frozen IOBRpy
baseline outputs, so it is not part of this fast file.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from iobrx._fast import lr_cal_fast as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bundle():
    data = F._lr_data()
    return data["intercell_networks"], data["group_lrpairs"]


def _gene_panel(n_pairs: int = 40):
    """A deterministic gene panel drawn from across the pancan pair list, so
    the tests exercise real ligand/receptor names (incl. grouped families)
    without hardcoding the network contents."""
    nets, _ = _bundle()
    net = nets["pancan"]
    pairs = list(dict.fromkeys(f"{str(a)}_{str(b)}"
                               for a, b in zip(net["ligands"], net["receptors"])))
    idx = np.linspace(0, len(pairs) - 1, n_pairs).astype(int)
    genes: list[str] = []
    for i in idx:
        for g in pairs[i].split("_"):
            if g not in genes:
                genes.append(g)
    return genes, pairs


def _matrix(genes, n_samples=6, seed=0):
    """Random TPM-like matrix with deliberate filter-trigger rows."""
    rng = np.random.default_rng(seed)
    vals = rng.random((len(genes), n_samples)) * 500.0
    df = pd.DataFrame(vals, index=list(genes),
                      columns=[f"S{i}" for i in range(n_samples)])
    if len(genes) >= 6:
        df.iloc[0, :] = 0.0                  # all-zero  -> zero variance
        df.iloc[1, :] = 0.1                  # constant nonzero (mean-rounding
        df.iloc[2, :] = 1e8 + 0.1            #   boundary: std()==0 must match
        df.iloc[3, 0] = np.nan               # NA row    -> dropna(how='any')
        df.iloc[4, 1] = np.inf               # +inf row  -> isfinite filter
        df.iloc[5, :] = -5.0                 # < -1      -> log2 gives NaN
    return df


def _bitwise(a: pd.DataFrame, b: pd.DataFrame, ctx: str = ""):
    assert list(a.columns) == list(b.columns), f"{ctx}: columns differ"
    assert list(a.index) == list(b.index), f"{ctx}: index differs"
    num = list(a.select_dtypes(include=[np.number]).columns)
    for c in a.columns:                      # object cols (e.g. ID) verbatim
        if c not in num:
            assert a[c].astype(str).equals(b[c].astype(str)), f"{ctx}: col {c}"
    va, vb = a[num].to_numpy(dtype="float64"), b[num].to_numpy(dtype="float64")
    assert va.shape == vb.shape, f"{ctx}: shape differs"
    if va.size:
        same = (va.view(np.uint64) == vb.view(np.uint64)).all()
        assert same, f"{ctx}: values not bit-identical"


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _orig_compute(df, cancer_type="pancan"):
    from iobrpy.workflow.LR_cal import compute_LR_pairs as orig
    nets, groups = _bundle()
    return orig(df, cancer_type, nets, groups, False)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def test_lr_data_cache_is_singleton():
    assert F._lr_data() is F._lr_data()


def test_compute_LR_pairs_bitwise_random_panel():
    genes, _ = _gene_panel()
    df = _matrix(genes)
    fast = F.compute_LR_pairs(df, "pancan", *_bundle())
    _bitwise(fast, _orig_compute(df), "random panel")
    # the panel is drawn from real pairs: at least one pair column survives
    assert fast.shape[1] >= 1


def test_file_roundtrip_csv_bytes_identical(tmp_path):
    """Full ORIGINAL-file semantics: read_csv -> compute -> to_csv text."""
    from iobrpy.workflow.LR_cal import LR_cal as orig_LR_cal

    genes, _ = _gene_panel()
    df = _matrix(genes)
    inp = tmp_path / "eset.csv"
    df.to_csv(inp)
    out_fast = tmp_path / "fast.csv"
    out_orig = tmp_path / "orig.csv"
    F.LR_cal(str(inp), str(out_fast), data_type="tpm")
    orig_LR_cal(str(inp), str(out_orig), "tpm")
    assert _sha(str(out_fast)) == _sha(str(out_orig))


def test_tsv_separator_detection(tmp_path):
    genes, _ = _gene_panel(n_pairs=12)
    df = _matrix(genes, n_samples=3)
    inp = tmp_path / "eset.tsv"
    df.to_csv(inp, sep="\t")
    out = tmp_path / "res.tsv"
    res = F.LR_cal(str(inp), str(out), data_type="tpm")
    assert F.detect_sep(str(inp)) == "\t" and F.detect_sep(str(out)) == "\t"
    text = out.read_text().splitlines()
    assert "\t" in text[0]
    assert len(text) == res.shape[0] + 1


def test_dataframe_input_matches_original_core(tmp_path):
    """DataFrame input (iobrx extension) must equal the ORIGINAL core run on
    the SAME in-memory frame. NOTE: comparing df-input against path-input of
    a to_csv'd copy is NOT bitwise-possible in general: pandas' CSV float
    parse (precise_xstrtod) is off by 1-2 ulp on ~12% of round-tripped
    values, which can even flip a constant-row std()==0 filter decision —
    an input-source difference, not an engine difference (the same drift is
    documented in the campaign module_spec §8-6)."""
    genes, _ = _gene_panel(n_pairs=20)
    df = _matrix(genes)
    via_df = F.LR_cal(df, None, data_type="tpm", engine="fast")
    via_orig = F.LR_cal(df, None, data_type="tpm", engine="original")
    _bitwise(via_df, via_orig, "df input vs original core")
    assert via_df.columns[0] == "ID"          # shell inserts ID like the ORIGINAL


def test_zero_variance_boundary_rows():
    """std()==0 decisions on constant-nonzero rows depend on the exact
    pandas two-pass pairwise summation order (0.1 * 10 does not sum to 1.0
    in every order); the Rust kernel must reproduce pandas bit-for-bit."""
    genes, _ = _gene_panel(n_pairs=8)
    df = pd.DataFrame({
        "S0": [0.1] * len(genes),
        "S1": [0.1] * len(genes),
    }, index=genes)
    df.iloc[:, 0] = 1e8 + 0.1
    df.iloc[:, 1] = 1e8 + 0.1
    fast_mask = F.feature_manipulation(df, n_threads=2)
    from iobrpy.workflow.LR_cal import feature_manipulation as orig_fm
    valid_orig = orig_fm(df, is_matrix=True)
    assert fast_mask is not None
    assert fast_mask[0] == valid_orig            # identical keep lists
    # and the per-row std bit pattern itself
    vals = df.to_numpy(dtype="float64")
    for i in range(vals.shape[0]):
        assert F._std_zero_two_pass(vals[i]) == (df.iloc[i].astype("float64").std() == 0)


def test_single_sample_skips_zero_variance_filter():
    genes, _ = _gene_panel(n_pairs=8)
    df = pd.DataFrame({"S0": np.zeros(len(genes))}, index=genes)  # all zero-var
    fast = F.compute_LR_pairs(df, "pancan", *_bundle())
    _bitwise(fast, _orig_compute(df), "single sample")
    # ORIGINAL keeps zero-variance genes when samples <= 1, so pairs survive
    assert fast.shape[1] >= 1


def test_int64_frame():
    genes, _ = _gene_panel(n_pairs=12)
    rng = np.random.default_rng(3)
    df = pd.DataFrame((rng.random((len(genes), 4)) * 1000).astype(np.int64),
                      index=genes, columns=list("abcd"))
    fast = F.compute_LR_pairs(df, "pancan", *_bundle())
    _bitwise(fast, _orig_compute(df), "int64")


def test_double_log_nan_skipna_min():
    """log2-scale input (< -1 -> NaN): skipna min keeps a pair column alive
    only where at least one gene is non-NaN in EVERY sample — the ORIGINAL
    imvigor210 behaviour on a synthetic frame."""
    genes, pairs = _gene_panel(n_pairs=6)
    both_genes = pairs[0].split("_")
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.random((len(genes), 5)) * 10, index=genes,
                      columns=[f"S{i}" for i in range(5)])
    df.loc[:] = np.log2(df + 1) - 5.0            # mostly < -1 -> NaN after log2
    df.loc[both_genes[0], :] = [1.0, 2.0, np.nan, 4.0, -9.0]   # mixed NaN
    df.loc[both_genes[1], :] = [5.0, np.nan, 3.0, -9.0, 2.0]
    fast = F.compute_LR_pairs(df, "pancan", *_bundle())
    _bitwise(fast, _orig_compute(df), "double-log NaN")


def test_guard_fallback_object_and_duplicate():
    from iobrpy.workflow.LR_cal import compute_LR_pairs as orig

    genes, _ = _gene_panel(n_pairs=6)
    rng = np.random.default_rng(5)
    obj = pd.DataFrame(rng.random((len(genes), 3)) * 10, index=genes,
                       columns=list("abc"))
    obj.iloc[0, 1] = "x"                         # forces object dtype
    assert F.feature_manipulation(obj) is None   # guard rejects
    nets, groups = _bundle()
    _bitwise(F.compute_LR_pairs(obj, "pancan", nets, groups),
             orig(obj, "pancan", nets, groups, False), "object fallback")

    dup = pd.DataFrame(rng.random((len(genes) + 1, 3)) * 10,
                       index=[genes[0]] + list(genes), columns=list("abc"))
    assert F.feature_manipulation(dup) is None   # duplicate index rejected
    _bitwise(F.compute_LR_pairs(dup, "pancan", nets, groups),
             orig(dup, "pancan", nets, groups, False), "dup fallback")


def test_numpy_fallback_engine_matches_rust(monkeypatch):
    genes, _ = _gene_panel()
    df = _matrix(genes)
    fast_rust = F.compute_LR_pairs(df, "pancan", *_bundle())
    monkeypatch.setattr(F, "_RUST_READY", False)
    fast_np = F.compute_LR_pairs(df, "pancan", *_bundle())
    _bitwise(fast_np, fast_rust, "numpy fallback vs rust")


def test_engine_original_matches_fast():
    genes, _ = _gene_panel()
    df = _matrix(genes)
    a = F.LR_cal(df, None, data_type="tpm", engine="fast")
    b = F.LR_cal(df, None, data_type="tpm", engine="original")
    _bitwise(a, b, "engine original vs fast")


def test_non_default_cancer_type():
    genes, _ = _gene_panel(n_pairs=25)
    df = _matrix(genes, n_samples=4, seed=9)
    for ct in ("STAD", "GBM"):
        _bitwise(F.compute_LR_pairs(df, ct, *_bundle()),
                 _orig_compute(df, ct), f"cancer_type={ct}")


def test_count_branch_bug_compatible_typeerror():
    """iobrpy 0.2.0 LR_cal(data_type='count') calls count2tpm without its two
    required positional annotations and dies with TypeError. iobrx must raise
    the IDENTICAL error (bug-compatibility, deliberately not fixed)."""
    genes, _ = _gene_panel(n_pairs=4)
    df = _matrix(genes, n_samples=2)
    with pytest.raises(TypeError) as exc:
        F.LR_cal(df, None, data_type="count", id_type="ensembl")
    assert "missing 2 required positional arguments" in str(exc.value)
    assert "anno_grch38" in str(exc.value) and "anno_gc_vm32" in str(exc.value)


def test_invalid_cancer_type_keyerror():
    genes, _ = _gene_panel(n_pairs=4)
    df = _matrix(genes, n_samples=2)
    with pytest.raises(KeyError):
        F.LR_cal(df, None, data_type="tpm", cancer_type="NOT_A_CANCER")


def test_verbose_lines_match_original(capsys):
    from iobrpy.workflow.LR_cal import LR_cal as orig_LR_cal

    genes, _ = _gene_panel(n_pairs=10)
    df = _matrix(genes, n_samples=3)
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "eset.csv")
        df.to_csv(inp)
        # SAME input source (path) on both sides — a df-vs-csv comparison
        # would include the CSV round-trip parse drift (see above).
        F.LR_cal(inp, None, data_type="tpm", verbose=True)
        fast_lines = [l for l in capsys.readouterr().out.splitlines()
                      if l.startswith(("After ", "[LOG]"))]
        orig_LR_cal(inp, os.path.join(td, "o.csv"), "tpm", "ensembl",
                    "pancan", True)
    orig_lines = [l for l in capsys.readouterr().out.splitlines()
                  if l.startswith(("After ", "[LOG]"))]
    assert fast_lines == orig_lines


def test_public_api_wrapper_when_wired():
    """Runs once the strategy layer wires iobrx.lr_cal (INTEGRATION.md)."""
    import iobrx

    if not hasattr(iobrx, "lr_cal"):
        pytest.skip("iobrx.lr_cal not wired yet")
    genes, _ = _gene_panel(n_pairs=15)
    df = _matrix(genes, n_samples=4)
    api = iobrx.lr_cal(df, data_type="tpm", n_threads=2)
    core = F.LR_cal(df, None, data_type="tpm")
    _bitwise(api, core, "api vs fast core")
    py = iobrx.lr_cal(df, data_type="tpm", backend="python")
    _bitwise(py, core, "backend=python vs fast core")
