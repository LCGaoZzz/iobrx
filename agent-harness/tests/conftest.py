from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory):
    import iobrx

    root = tmp_path_factory.mktemp("harness-data")
    counts = pd.read_parquet(ROOT / "tutorials/data/eset_stad.parquet").iloc[:, :3]
    counts = counts.astype("float64")
    tpm = iobrx.count2tpm(counts, check_data=True, remove_version=True)
    log = np.log2(tpm + 1)
    signature = pd.read_parquet(ROOT / "tutorials/data/imvigor210_eset.parquet").iloc[:, :12].astype("float64")
    for name, matrix in (("counts", counts), ("tpm", tpm), ("log", log), ("signature", signature)):
        matrix.to_parquet(root / f"{name}.parquet")
    return root


def request_for(analysis, fixtures, output):
    if analysis in {"anno_eset", "count2tpm"}:
        filename, scale, gene_id = "counts", "counts", "ensembl"
    elif analysis in {"mcpcounter", "estimate_score"}:
        filename, scale, gene_id = "log", "log2p1", "symbol"
    elif analysis.startswith("signature_"):
        filename, scale, gene_id = "signature", "preprocessed", "symbol"
    else:
        filename, scale, gene_id = "tpm", "tpm", "symbol"
    params = {}
    if analysis == "count2tpm":
        params = {"check_data": True, "remove_version": True}
    elif analysis == "cibersort":
        params = {"perm": 0, "QN": False, "backend": "auto"}
    return {"schema_version": "1.0", "analysis": analysis,
            "input": {"path": str(fixtures / f"{filename}.parquet"), "orientation": "genes_by_samples",
                      "scale": scale, "gene_id": gene_id, "organism": "hsa"},
            "parameters": params, "threads": 2, "output_dir": str(output)}
