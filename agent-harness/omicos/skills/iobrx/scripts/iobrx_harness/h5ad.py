"""Read only a selected AnnData expression slot; no normalization or aggregation."""
from __future__ import annotations

import math
import re

from .runtime import HarnessError, reject


def _local_node(root, path, h5py):
    """Do not let HDF5 links bypass the containing file/workspace boundary."""
    node = root
    for name in path.split("/"):
        if not isinstance(node, h5py.Group) or name not in node:
            reject(f"Missing h5ad element: {path}")
        if not isinstance(node.get(name, getlink=True), h5py.HardLink):
            reject("h5ad soft/external links are unsupported; use a self-contained file")
        node = node[name]

    def check(value):
        if isinstance(value, h5py.Group):
            for key in value:
                if not isinstance(value.get(key, getlink=True), h5py.HardLink):
                    reject("h5ad soft/external links are unsupported; use a self-contained file")
        elif value.is_virtual or value.external:
            reject("h5ad virtual/external datasets are unsupported; use a self-contained file")

    check(node)
    if isinstance(node, h5py.Group):
        node.visititems(lambda _name, value: check(value))
    return node


def read_matrix(spec):
    """Return samples x genes plus slot provenance for the common matrix loader.

    ``h5ad.matrix`` is explicitly X, raw.X or layers/<name>. obs must already
    represent bulk/pseudobulk samples. Only the selected expression, obs and
    corresponding var are read; unrelated layers/embeddings/uns are not loaded.
    The allocation check bounds the final float64 matrix, NOT peak memory.
    """
    options = spec.get("h5ad")
    if not isinstance(options, dict) or not isinstance(options.get("matrix"), str):
        reject(".h5ad requires input.h5ad.matrix: X, raw.X or layers/<name>")
    slot = options["matrix"]
    if re.fullmatch(r"X|raw\.X|layers/[^/]+", slot) is None:
        reject("input.h5ad.matrix must be X, raw.X or layers/<name>")
    if spec.get("orientation") != "samples_by_genes":
        reject("AnnData is obs x var: .h5ad requires orientation=samples_by_genes")
    budget = options.get("max_dense_bytes", 536870912)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        reject("input.h5ad.max_dense_bytes must be a positive integer")
    try:
        import h5py
        import numpy as np
        import pandas as pd
        from anndata.io import read_elem
        from scipy import sparse
    except ImportError as exc:
        raise HarnessError(
            "h5ad input needs the optional anndata>=0.11 dependency in this interpreter "
            "(or the iobrx-harness[h5ad] extra); table inputs do not need it",
            "environment_error", 3) from exc

    matrix_path = "raw/X" if slot == "raw.X" else slot
    var_path = "raw/var" if slot == "raw.X" else "var"
    with h5py.File(spec["path"], "r") as handle:
        node = _local_node(handle, matrix_path, h5py)
        if isinstance(node, h5py.Dataset):
            shape, dtype = node.shape, node.dtype
            storage = "dense"
        else:
            storage = node.attrs.get("encoding-type", "")
            if storage not in {"csr_matrix", "csc_matrix"}:
                reject("Selected h5ad expression must be a dense, CSR or CSC matrix")
            shape = node.attrs.get("shape", ())
            dtype = node["data"].dtype
        if len(shape) != 2 or any(int(n) <= 0 for n in shape):
            reject("Selected h5ad expression must be a nonempty two-dimensional matrix")
        if dtype.kind not in "iuf":
            reject("Selected h5ad expression must contain real numeric values")
        dense_bytes = math.prod(int(n) for n in shape) * 8
        if dense_bytes > budget:
            reject(f"h5ad needs {dense_bytes} bytes for its float64 matrix, above "
                   f"max_dense_bytes={budget}; prepare a scoped bulk input or explicitly "
                   "increase the budget with sufficient memory (peak usage is higher)")
        obs = read_elem(_local_node(handle, "obs", h5py))
        var = read_elem(_local_node(handle, var_path, h5py))
        if not isinstance(obs, pd.DataFrame) or not isinstance(var, pd.DataFrame):
            reject("h5ad obs and var must be encoded dataframes")
        if tuple(shape) != (len(obs), len(var)):
            reject("Selected h5ad expression shape disagrees with obs/var")
        for key, frame in (("sample_column", obs), ("gene_column", var)):
            if key in options and options[key] not in frame.columns:
                reject(f"h5ad {key} not found: {options[key]}")
        sample_ids = obs[options["sample_column"]] if "sample_column" in options else obs.index
        gene_ids = var[options["gene_column"]] if "gene_column" in options else var.index
        values = read_elem(node)
        values = values.toarray() if sparse.issparse(values) else values
        matrix = pd.DataFrame(np.asarray(values, dtype="float64"),
                              index=pd.Index(sample_ids), columns=pd.Index(gene_ids))
    return matrix, {**options, "matrix": slot, "var": var_path, "storage": storage,
                    "dense_bytes": dense_bytes, "max_dense_bytes": budget}
