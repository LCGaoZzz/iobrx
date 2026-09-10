"""estimate_fast: vectorized drop-in for iobrpy.workflow.estimate.estimate_score().

Same signature: estimate_score(input_df, platform)

Bit-exactness strategy (verified against the reference pipeline on the
official eset_stad_symbol, platform='affy'):
  1. filter_common_genes' pd.merge(common_genes, input_df, left_on=
     'GeneSymbol', right_index=True, how='inner') is exactly a hash lookup
     of common_genes.GeneSymbol against input_df's index, keeping
     common-genes order: verified gene order and expression values
     bit-equal via Index.get_indexer + numpy gather.
  2. The per-column pd.Series(m[:, j]).rank(method='average') loop is
     replaced by one DataFrame.rank(axis=0, method='average') call:
     verified bit-equal for every element, and 10000 * ranks / Ng is an
     elementwise op (layout-independent).
  3. set_indices = [gene_names.index(g) for g in overlap] (O(Ng) list scan
     per gene, FIRST occurrence) is replaced by a first-occurrence dict built
     in one pass; membership in the gene set is precomputed as a boolean mask
     over genes, and TAG = member_mask[order] is exactly np.isin(order, set_indices)
     (verified element-identical for both gene sets).
  4. The per-sample enrichment arithmetic is copied verbatim from the
     original (same numpy calls in the same order: argsort(-col),
     correl, masked sum, P0/F0/Pn/Fn cumsums, RES.sum()), so every float
     operation and summation order is unchanged.
  5. ESTIMATEScore = score_df.sum(axis=0) and the platform=='affymetrix'
     TumorPurity conversion are copied verbatim.

Resource loading (common_genes.txt + estimate_data.pkl) is cached at module
level after the first call; the upstream function re-reads them from disk
on every call.
"""
import math
import pickle

import numpy as np
import pandas as pd
from iobrx._resources import resource_path

__all__ = ["estimate_score"]

_cache = {}


def _load_resources():
    if not _cache:
        txt_path = resource_path("common_genes.txt")
        common_genes = pd.read_csv(txt_path, sep="\t", header=0, dtype=str)
        pkl_path = resource_path("estimate_data.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        SI_geneset = data["SI_geneset"]
        _cache["common_genes"] = common_genes
        _cache["SI_geneset"] = SI_geneset
        _cache["PurityDataAffy"] = data["PurityDataAffy"]
        # constant per resource: gene-set rows/names, pre-built frozensets
        gs = SI_geneset.iloc[:, 1:].values.tolist()
        _cache["gs"] = gs
        _cache["gs_names"] = SI_geneset.index.tolist()
        _cache["gs_sets"] = [frozenset(gene_set) for gene_set in gs]
    return _cache["common_genes"], _cache["SI_geneset"], _cache["PurityDataAffy"]


def _filter_common_genes_fast(input_df: pd.DataFrame, common_genes: pd.DataFrame):
    """Vectorized equivalent of filter_common_genes + merge (inner, keeps
    common-genes order). Returns (merged_values, gene_names, sample_names)."""
    gene_syms = common_genes["GeneSymbol"].to_numpy()
    pos = pd.Index(input_df.index).get_indexer(gene_syms)
    ok = pos >= 0
    gene_names = [g for g, k in zip(gene_syms, ok) if k]
    m = input_df.to_numpy(dtype=None, copy=False)[pos[ok]]
    return m, gene_names, list(input_df.columns)


def _filter_common_genes_ref(input_df: pd.DataFrame, common_genes: pd.DataFrame):
    """Original merge path (used when input index is non-unique etc.)."""
    merged = pd.merge(common_genes, input_df, left_on="GeneSymbol", right_index=True, how="inner")
    merged.index = merged["GeneSymbol"]
    merged = merged.drop(columns=common_genes.columns)
    return merged.to_numpy(dtype=None, copy=False), merged.index.to_list(), merged.columns.to_list()


def estimate_score(input_df: pd.DataFrame, platform: str):
    print(" Starting estimate_score")
    common_genes, SI_geneset, PurityDataAffy = _load_resources()

    if input_df.index.is_unique:
        m, gene_names, sample_names = _filter_common_genes_fast(input_df, common_genes)
    else:
        m, gene_names, sample_names = _filter_common_genes_ref(input_df, common_genes)
    print(f"Merged dataset includes {m.shape[0]} genes.")

    Ns = m.shape[1]
    Ng = m.shape[0]

    # Rank normalization. Without NaN, one DataFrame.rank(axis=0) call is
    # bit-equal to the upstream per-column pd.Series.rank loop (verified);
    # with NaN we reproduce the upstream loop verbatim (pandas NaN rank
    # semantics are not trivially vectorizable bit-exactly).
    if np.isnan(m).any():
        ranks = np.zeros_like(m)
        for j in range(Ns):
            ranks[:, j] = pd.Series(m[:, j]).rank(method="average").values
    else:
        ranks = pd.DataFrame(m).rank(axis=0, method="average").to_numpy()
    m_ranked = 10000 * ranks / Ng

    # Gene set enrichment (arithmetic copied verbatim from upstream)
    gs = _cache["gs"]
    gs_names = _cache["gs_names"]
    gs_sets = _cache["gs_sets"]
    # FIRST occurrence per label, matching upstream gene_names.index(g)
    # (a plain {g: i for i, g in enumerate(...)} dict would keep the LAST
    # occurrence and pick different signature members on duplicated labels)
    gene_pos = {}
    for i, g in enumerate(gene_names):
        if g not in gene_pos:
            gene_pos[g] = i
    gene_names_set = set(gene_names)
    score_matrix = np.zeros((len(gs), Ns))
    for i, gene_set_set in enumerate(gs_sets):
        overlap = gene_set_set & gene_names_set
        print(f" Gene set '{gs_names[i]}' overlap: {len(overlap)} genes")
        if not overlap:
            score_matrix[i, :] = np.nan
            continue
        set_indices = [gene_pos[g] for g in overlap]
        member = np.zeros(Ng, dtype=bool)
        member[set_indices] = True
        ES_vec = []
        for j in range(Ns):
            col = m_ranked[:, j]
            order = np.argsort(-col)
            correl = np.abs(col[order]) ** 0.25
            TAG = member[order].astype(int)
            no_TAG = 1 - TAG
            sum_correl = correl[TAG == 1].sum()
            P0 = no_TAG / (len(order) - len(set_indices))
            F0 = np.cumsum(P0)
            Pn = TAG * correl / sum_correl
            Fn = np.cumsum(Pn)
            RES = Fn - F0
            ES_vec.append(RES.sum())
        score_matrix[i, :] = ES_vec

    score_df = pd.DataFrame(score_matrix, index=gs_names, columns=sample_names)
    est_score = score_df.sum(axis=0)
    score_df.loc["ESTIMATEScore"] = est_score
    if platform == "affymetrix":
        def convert(x): return math.cos(0.6049872018 + 0.0001467884 * x)
        purity = est_score.apply(convert).where(lambda x: x >= 0, other=np.nan)
        score_df.loc["TumorPurity"] = purity
    return score_df
