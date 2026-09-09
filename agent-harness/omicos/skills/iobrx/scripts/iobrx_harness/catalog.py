"""Machine-readable capabilities and strict request contracts (no numerics)."""
from copy import deepcopy


def choice(values, default=None):
    result = {"type": "string", "enum": values}
    if default is not None:
        result["default"] = default
    return result


def boolean(default):
    return {"type": "boolean", "default": default}


def integer(default, minimum=1):
    return {"type": "integer", "minimum": minimum, "default": default}


def operation(function, scales, ids, parameters, layout, meaning, species=("hsa",)):
    return dict(function=function, scales=scales, gene_ids=ids, organisms=list(species),
                parameters={"type": "object", "additionalProperties": False,
                            "properties": parameters}, output_layout=layout, meaning=meaning)


CATALOG = {
    "anno_eset": operation("anno_eset", ["counts", "tpm", "linear", "log2p1"],
        ["ensembl", "probe"], {
            "annotation": choice(["anno_grch38", "anno_rnaseq", "anno_hug133plus2", "anno_illumina"], "anno_grch38"),
            "method": choice(["mean", "sd", "sum"], "mean"),
        }, "genes_by_samples", "Symbol mapping and upstream probe selection; preserves input scale."),
    "count2tpm": operation("count2tpm", ["counts"], ["ensembl", "entrez", "symbol", "mgi"], {
        "check_data": boolean(False), "remove_version": boolean(False),
    }, "genes_by_samples", "TPM with bundled effective gene lengths; not raw counts.", ("hsa", "mmus")),
    "cibersort": operation("cibersort", ["tpm", "linear"], ["symbol"], {
        "perm": integer(100, 0), "QN": boolean(False), "absolute": boolean(False),
        "abs_method": choice(["sig.score", "no.sumto1"], "sig.score"),
        "backend": choice(["auto", "rust", "python"], "auto"),
    }, "samples_by_results", "LM22 weights plus P-value/Correlation/RMSE; absolute mode adds a score."),
    "epic": operation("epic", ["tpm"], ["symbol"], {
        "reference": choice(["TRef", "BRef"], "TRef"),
        "with_other_cells": boolean(True), "constrained_sum": boolean(True),
    }, "samples_by_results", "Separate cellFractions, mRNAProportions and fit_gof tables."),
    "quantiseq": operation("quantiseq", ["tpm"], ["symbol"], {
        "tumor": boolean(False), "mRNAscale": boolean(True),
        "rmgenes": choice(["unassigned", "default", "none"], "unassigned"),
    }, "samples_by_results", "TIL10 cell fractions; retain the selected tumor/mRNA settings."),
    "mcpcounter": operation("mcpcounter", ["linear", "log2p1"], ["symbol", "entrez", "probe"], {},
        "results_by_samples", "Marker abundance scores; not cell fractions or comparable across cell types."),
    "estimate_score": operation("estimate_score", ["linear", "log2p1"], ["symbol"], {
        "platform": choice(["rnaseq", "affy", "affymetrix"], "rnaseq"),
    }, "results_by_samples", "Stromal/immune/ESTIMATE scores; only affymetrix adds the upstream purity estimate."),
}
for method in ("pca", "zscore", "ssgsea", "integration"):
    CATALOG[f"signature_{method}"] = operation("calculate_sig_score", ["tpm", "linear", "log2p1", "preprocessed"], ["symbol"], {
        "signature": {"type": "array", "items": {"type": "string", "minLength": 1},
                      "minItems": 1, "uniqueItems": True, "default": ["signature_collection"]},
        "mini_gene_count": integer(3), "adjust_eset": boolean(True),
        "backend": choice(["auto", "rust", "python"], "auto"),
    }, "samples_by_results_with_ID", "Signature scores, not fractions. zscore retains IOBRpy's mean-based semantics.")

INPUT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["path", "orientation", "scale", "gene_id", "organism"],
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "orientation": choice(["genes_by_samples", "samples_by_genes"]),
        "scale": choice(["counts", "tpm", "linear", "log2p1", "preprocessed"]),
        "gene_id": choice(["ensembl", "symbol", "entrez", "probe", "mgi"]),
        "organism": choice(["hsa", "mmus"]),
    },
}


LEGACY_ANALYSES = tuple(CATALOG)
from .extended_catalog import extend
extend(CATALOG, INPUT_SCHEMA, choice, boolean, integer)


def request_schema():
    base = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "iobrx harness request v1", "type": "object", "additionalProperties": False,
        "required": ["schema_version", "analysis", "input", "output_dir"],
        "properties": {
            "schema_version": {"const": "1.0"}, "analysis": choice(list(CATALOG)),
            "input": {"type": "object"}, "output_dir": {"type": "string", "minLength": 1},
            "threads": {"type": "integer", "minimum": 1, "maximum": 1024},
            "parameters": {"type": "object"},
        },
        "allOf": [],
    }
    for name, spec in CATALOG.items():
        schema = deepcopy(spec.get("input_schema", INPUT_SCHEMA))
        if name in LEGACY_ANALYSES:
            schema["properties"].update(scale={"enum": spec["scales"]}, gene_id={"enum": spec["gene_ids"]},
                                        organism={"enum": spec["organisms"]})
        base["allOf"].append({"if": {"properties": {"analysis": {"const": name}}},
            "then": {"properties": {"parameters": deepcopy(spec["parameters"]), "input": schema}}})
    return base


def capabilities():
    return {"schema_version": "1.0", "skill_id": "iobrx", "status": "completed",
            "analyses": deepcopy(CATALOG), "input_formats": ["csv", "tsv", "parquet"],
            "request_schema": request_schema()}
