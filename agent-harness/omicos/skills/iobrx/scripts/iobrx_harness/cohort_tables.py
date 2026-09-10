"""Small table helpers for cohort recipes, not a batch runner or state machine."""
from __future__ import annotations


def align_samples(table, sample_ids, *, sample_column=None):
    """Align a samples-by-results table by exact IDs; never drop or invent rows.

    Transpose results-by-samples outputs explicitly before calling. A case ID
    shared by multiple samples is not a unique sample key and must be resolved
    according to the study design, not by truncating IDs or keeping a first row.
    """
    import pandas as pd

    def ids(values, label):
        if isinstance(values, pd.MultiIndex):
            raise ValueError(f"{label} must not be a MultiIndex")
        index = pd.Index(values)
        if isinstance(index, pd.MultiIndex):
            raise ValueError(f"{label} must not be a MultiIndex")
        if index.empty or index.hasnans:
            raise ValueError(f"{label} must be nonempty and nonmissing")
        index = index.map(str)
        if index.has_duplicates or any(not x or x != x.strip() for x in index):
            raise ValueError(f"{label} must be unique, nonempty, unpadded identifiers")
        return index

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Result must be a nonempty DataFrame")
    expected = ids(sample_ids, "sample_ids")
    if not table.columns.is_unique:
        raise ValueError("Result columns must be unique")
    result = table.set_index(sample_column) if sample_column is not None else table.copy()
    result.index = ids(result.index, "result sample IDs")
    missing = expected.difference(result.index, sort=False)
    extra = result.index.difference(expected, sort=False)
    if len(missing) or len(extra):
        raise ValueError(f"Sample mismatch: missing={missing[:5].tolist()}, extra={extra[:5].tolist()}")
    return result.loc[expected].copy()


def combine_cohorts(tables):
    """Stack same-method cohort tables with a (cohort, sample_id) index.

    The feature columns must agree exactly, including order. Methods with
    distinct biological meanings should not be combined with this helper.
    """
    import pandas as pd

    if not tables:
        raise ValueError("At least one cohort table is required")
    columns = None
    aligned = {}
    for cohort, table in tables.items():
        if not isinstance(cohort, str) or not cohort or cohort != cohort.strip():
            raise ValueError("Cohort names must be nonempty, unpadded strings")
        frame = align_samples(table, table.index)
        if columns is not None and not frame.columns.equals(columns):
            raise ValueError(f"Result columns differ for cohort {cohort}; reconcile explicitly")
        columns = frame.columns
        aligned[cohort] = frame
    return pd.concat(aligned, names=["cohort", "sample_id"], verify_integrity=True)
