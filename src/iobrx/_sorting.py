"""NumPy owns CPU dispatch and quicksort tie ordering on every platform.

The solver must see the same permutation as the user's installed NumPy.
Hard-coding either an AVX-512 or a scalar sort cannot provide that contract.
"""
import numpy as np


def argsort(values):
    """Return the local NumPy quicksort permutation, including NaN ordering."""
    return np.asarray(np.argsort(values, kind="quicksort"), dtype=np.uint64)


def quantile_normalize(values):
    """Replay IOBRpy's normalization with its exact reduction and tie order."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("expected a genes-by-samples matrix")
    if 0 in values.shape:
        return np.array(values, order="C", copy=True)
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    mean_sorted = sorted_values.mean(axis=1, keepdims=True)
    inverse = np.empty_like(order)
    np.put_along_axis(inverse, order, np.arange(values.shape[0])[:, None], axis=0)
    return np.ascontiguousarray(np.take_along_axis(mean_sorted, inverse, axis=0))
