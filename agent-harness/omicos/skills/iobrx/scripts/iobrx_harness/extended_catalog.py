"""Typed adapters added in iobrx 0.3; no arbitrary shell arguments or executables."""
from copy import deepcopy


def extend(catalog, matrix_schema, choice, boolean, integer):
    path = {"type": "string", "minLength": 1}
    project = {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]*$", "default": "iobrx"}
    suffix = {"type": "string", "pattern": "^[^/\\\\]+$", "default": "_1.fastq.gz"}

    def files(kind, extra=None, required=()):
        return {"type": "object", "additionalProperties": False,
                "required": ["kind", "path", *required],
                "properties": {"kind": {"const": kind}, "path": path, **(extra or {})}}

    def matrix(scales, ids=("symbol",), organism="hsa"):
        schema = deepcopy(matrix_schema)
        schema["properties"].update(scale=choice(list(scales)), gene_id=choice(list(ids)), organism={"const": organism})
        return schema

    def add(name, schema, params=None, tools=(), meaning="", function=None):
        catalog[name] = {"function": function or name, "input_schema": schema,
                         "parameters": {"type": "object", "additionalProperties": False, "properties": params or {}},
                         "output_layout": "method_specific", "required_tools": list(tools), "meaning": meaning}

    add("ips", matrix(["tpm", "linear", "log2p1"]), meaning="Immunophenoscore; retain upstream scale semantics.")
    add("lr_cal", matrix(["counts", "tpm"], ["symbol", "ensembl"]),
        {"cancer_type": {"type": "string", "minLength": 1, "default": "pancan"}},
        meaning="Bulk ligand-receptor scores; these do not establish cell-to-cell communication.")
    add("log2_eset", matrix(["tpm", "linear", "log2p1"]), meaning="Upstream conditional transform; inspect whether log2(x+1) was applied.")
    add("mouse2human", matrix(["counts", "tpm", "linear", "log2p1"], organism="mmus"), meaning="Bundled mouse-human symbol mapping, matrix mode.")
    feature = files("feature_table", {"orientation": {"const": "samples_by_features"}, "scale": choice(["linear", "preprocessed"])}, ["orientation", "scale"])
    add("nmf", feature, {"kmin": integer(2, 2), "kmax": integer(6, 2), "log1p": boolean(False),
                        "normalize": boolean(False), "random_state": integer(42, 0), "max_iter": integer(1000)},
        meaning="Samples by nonnegative features; exports W/H, clusters and feature rankings without pickling models.")
    add("tme_cluster", feature, {"min_nc": integer(2, 2), "max_nc": integer(6, 2), "scale": boolean(True),
                                "nstart": integer(10), "max_iter": integer(10), "seed": integer(123, 0)},
        meaning="KL-index selection over sample features; sample count must exceed max_nc+1.")
    add("bayesprism", matrix(["counts"]), {"key": {"type": "string", "minLength": 1, "default": "Malignant_cells"},
        "state_order": choice(["sorted", "legacy"], "sorted")},
        meaning="Bundled single-cell reference only. Counts must be integers; sorted state order is reproducible across processes.")
    add("tme_profile", matrix(["tpm"]), {"signature": {"type": "string", "minLength": 1, "default": "all"},
        "sig_method": choice(["pca", "zscore", "ssgsea", "integration"], "integration"),
        "mini_gene_count": integer(2), "perm": integer(100, 0), "QN": boolean(False),
        "platform": choice(["rnaseq", "affymetrix"], "rnaseq"), "arrays": boolean(False)},
        meaning="RNA-seq defaults in this adapter; original CIBERSORT solver, sequential composition.")
    add("prepare_salmon", files("salmon_matrix"), {"return_feature": choice(["symbol", "ENST", "ENSG"], "symbol"), "remove_version": boolean(False)},
        meaning="GENCODE pipe-delimited transcript annotations in a merged Salmon TSV.")
    add("merge_salmon", files("salmon_directory"), {"project": project}, meaning="Stages quant.sf files in the run directory; source data remains read-only.")
    add("merge_star_count", files("star_directory"), {"project": project}, meaning="Stages ReadsPerGene files in the run directory; source data remains read-only.")
    add("fastq_qc", files("fastq_directory"), {"suffix1": suffix, "se": boolean(False), "batch_size": integer(1)},
        tools=["fastp", "multiqc"], meaning="fastp and MultiQC; any failed sample/report fails the run.")
    for name, tool in [("batch_salmon", "salmon"), ("batch_star_count", "STAR")]:
        extra = {"index": path, **({"gtf": path} if name == "batch_salmon" else {})}
        add(name, files("fastq_directory", extra, ["index"]), {"suffix1": suffix, "batch_size": integer(1)}, tools=[tool])
    add("trust4", files("fastq_directory", {"f": path, "ref": path}, ["f", "ref"]), tools=["run-trust4"],
        meaning="TRUST4 FASTQ-directory mode with explicit reference files; inspect clonotype support.")
    add("runall", files("fastq_directory", {"index": path, "f": path, "ref": path}, ["index", "f", "ref"]),
        {"mode": choice(["salmon", "star"], "salmon"), "project": project, "suffix1": suffix,
         "batch_size": integer(1), "perm": integer(100, 0)}, tools=["fastp", "multiqc", "run-trust4"],
        meaning="Upstream end-to-end defaults, including QN/arrays; fresh output only. References and tools must already be installed.")
