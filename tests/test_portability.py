"""Parity on any CPU, including NumPy's different SIMD/tie-order paths."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import iobrx


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_quantile_normalization_matches_local_numpy(layout):
    native = pytest.importorskip("iobrx._rust")
    from iobrpy.workflow.cibersort import quantile_normalize_fast
    # Many ties are essential: a generic stable sort is NOT equivalent.
    matrix = np.random.default_rng(9).integers(0, 8, (401, 6)).astype(float)
    matrix[-1, :] = np.nan
    matrix = matrix[::2] if layout == "strided" else np.array(matrix, order=layout)
    np.testing.assert_array_equal(native.quantile_normalize_np(matrix),
                                  quantile_normalize_fast(matrix), strict=True)


def test_argsort_matches_local_numpy_with_ties_and_nan():
    native = pytest.importorskip("iobrx._rust")
    values = np.random.default_rng(4).integers(0, 7, 1001).astype(float)
    values[30:33] = np.nan
    for sample in (values, values[::2]):
        np.testing.assert_array_equal(native.argsort_f64_np(sample), np.argsort(sample))


@pytest.fixture(scope="module")
def mixture():
    from importlib.resources import files
    lm22 = pd.read_csv(files("iobrpy.resources").joinpath("lm22.txt"),
                       sep=r"\s+", engine="python", index_col=0).iloc[:400]
    rng = np.random.default_rng(12)
    weights = rng.dirichlet(np.ones(lm22.shape[1]), size=2)
    return pd.DataFrame(lm22.to_numpy() @ weights.T + rng.uniform(0, 0.1, (400, 2)),
                        index=lm22.index, columns=["A", "B"])


@pytest.mark.parametrize("qn,absolute", [(True, False), (False, False), (False, True)])
def test_portable_cibersort_matches_reference(mixture, qn, absolute):
    if not iobrx.backend_info()["native_available"]:
        pytest.skip("native extension not built")
    native = iobrx.cibersort(mixture, perm=2, QN=qn, absolute=absolute,
                            n_threads=2, backend="rust")
    python = iobrx.cibersort(mixture, perm=2, QN=qn, absolute=absolute,
                            n_threads=1, backend="python")
    columns = [c for c in python if c != "P-value"]
    pd.testing.assert_frame_equal(native[columns], python[columns], check_exact=True)


def test_unavailable_blas_uses_reference(mixture, monkeypatch):
    pytest.importorskip("iobrx._rust")
    from iobrx._fast import cibersort_fast
    def unavailable():
        raise RuntimeError("non-wheel BLAS layout")
    monkeypatch.setattr(cibersort_fast, "_init_blas", unavailable)
    auto = iobrx.cibersort(mixture, perm=0, n_threads=1)
    python = iobrx.cibersort(mixture, perm=0, n_threads=1, backend="python")
    pd.testing.assert_frame_equal(auto, python, check_exact=True)
    with pytest.raises(RuntimeError, match="compatible bundled OpenBLAS"):
        iobrx.cibersort(mixture, backend="rust")


def test_import_and_calculation_with_native_disabled():
    code = '''
import json, iobrx, numpy as np, pandas as pd
from importlib.resources import files
genes = pd.read_pickle(str(files("iobrpy.resources").joinpath("calculate_data.pkl")))["signature_collection"]
index = sorted({g for group in genes.values() for g in group})[:1500]
data = pd.DataFrame(np.random.default_rng(7).lognormal(size=(len(index), 5)), index=index, columns=list("ABCDE"))
out = iobrx.calculate_sig_score(data, "signature_collection", "ssgsea", n_threads=1)
assert out.shape[0] == 5 and out.shape[1] > 1
print(json.dumps(iobrx.backend_info()))
'''
    env = dict(os.environ, IOBRX_DISABLE_RUST="1")
    proc = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                          capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout.splitlines()[-1])
    assert info["native_available"] is False


def test_invalid_backend_fails_explicitly():
    with pytest.raises(ValueError, match="backend"):
        iobrx.calculate_sig_score(None, "signature_collection", "pca", backend="typo")
