"""Generate and execute the 0.3 matrix-workflow tutorials, one figure revision at a time.

Run from an environment with iobrx, nbformat, nbclient and ipykernel installed.
Each notebook retains its public API call, inputs, timing and embedded figure.
"""
from pathlib import Path
import argparse
import json
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"

SETUP = '''from pathlib import Path
from time import perf_counter
from contextlib import redirect_stdout, redirect_stderr
import io, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iobrx
from _common import load, tpm_input, configure, runtime_info
from _extension_figures import plot_result

iobrx.set_threads(2)
slug = SLUG
work = Path("extension-work") / slug
work.mkdir(parents=True, exist_ok=True)
print(json.dumps(runtime_info(), indent=2))
groups = None
configure()
'''

CASES = [
    ("13_ips", "Immunophenoscore", "Public STAD TPM, first four samples. IPS is a score, not a cell fraction.",
     'data = tpm_input().iloc[:, :4]', 'result = iobrx.ips(data)', 'plot = result.set_index("ID").select_dtypes("number")'),
    ("14_lr_cal", "Ligand–receptor scores", "Public STAD TPM, first four samples; pancan network, gene symbols. Bulk scores do not demonstrate cell-to-cell communication.",
     'data = tpm_input().iloc[:, :4]', 'result = iobrx.lr_cal(data, data_type="tpm", id_type="symbol", cancer_type="pancan")',
     'plot = result.set_index("ID").select_dtypes("number")'),
    ("15_nmf", "NMF sample structure", "The committed CIBERSORT tutorial output provides 10 samples × 22 LM22 features. Factor selection is exploratory at this sample size.",
     'data = pd.read_csv("results/07_cibersort_result.csv", index_col=0).iloc[:, :22]',
     'result = iobrx.nmf(data, kmin=2, kmax=4, random_state=42, max_iter=1000, outdir=work)',
     'plot = pd.DataFrame(result["W"], index=data.index, columns=[f"Factor {i+1}" for i in range(result["best_k"])])\nresult["top_features"].to_csv(work / "top_features.csv")'),
    ("16_tme_cluster", "TME clustering", "The same 10-sample LM22 feature table; KL-index selection, standardized features. Clusters are exploratory labels, not validated subtypes.",
     'data = pd.read_csv("results/07_cibersort_result.csv", index_col=0).iloc[:, :22]',
     'result = iobrx.tme_cluster(data.rename_axis("ID").reset_index(), id="ID", min_nc=2, max_nc=4, seed=123)',
     'plot = result.set_index("ID").select_dtypes("number")\ngroups = result["cluster"].tolist()'),
    ("17_log2_eset", "Conditional log transform", "Public STAD TPM, first four samples. The upstream function decides whether to transform; inspect the before/after range rather than assuming a transform occurred.",
     'data = tpm_input().iloc[:, :4]\ndata.to_csv(work / "input.csv")',
     'result = iobrx.log2_eset(work / "input.csv", work / "output.csv")',
     'plot = result.select_dtypes("number").T'),
    ("18_mouse2human", "Mouse to human symbols", "Small synthetic matrix with common mouse symbols. Values are illustrative; mapping uses the bundled upstream resource and can drop unmapped genes.",
     'data = pd.DataFrame([[20, 12, 16], [4, 7, 8], [30, 22, 18], [9, 3, 6]], index=["Trp53", "Cd3d", "Actb", "Gapdh"], columns=["M1", "M2", "M3"])',
     'result = iobrx.mouse2human(data, is_matrix=True)', 'plot = result.select_dtypes("number").T'),
    ("19_merge_salmon", "Merge Salmon quantification", "Synthetic quant.sf tables demonstrate the file contract. This notebook does not run Salmon or establish alignment accuracy.",
     '''data = pd.DataFrame({"S1": [6., 12., 24.], "S2": [14., 8., 20.], "S3": [10., 16., 8.]}, index=["TX1", "TX2", "TX3"])
for sample in data:
    folder = work / "quant" / sample
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Name": data.index, "Length": 100, "EffectiveLength": 80., "TPM": data[sample].values, "NumReads": [2., 4., 8.]}).to_csv(folder / "quant.sf", sep="\\t", index=False)''',
     'result, counts = iobrx.merge_salmon(work / "quant", "demo", n_threads=2)', 'plot = result.select_dtypes("number").T'),
    ("20_prepare_salmon", "Prepare Salmon gene matrix", "Synthetic GENCODE-style annotations. Transcript-to-symbol aggregation follows the upstream mean rule; it is not a new gene-level quantification method.",
     '''names = [f"ENST{i}.1|ENSG{i}.1|OTTHUMG{i}|OTTHUMT{i}|{gene}|{gene}|100|protein_coding" for i, gene in enumerate(["TP53", "CD3D", "ACTB"], 1)]
data = pd.DataFrame({"Name": names, "S1": [6., 12., 24.], "S2": [14., 8., 20.], "S3": [10., 16., 8.]})
data.to_csv(work / "salmon.tsv", sep="\\t", index=False)''',
     'result = iobrx.prepare_salmon(work / "salmon.tsv", work / "genes.csv", return_feature="symbol", remove_version=True)',
     'plot = result.set_index("Name").select_dtypes("number").T'),
    ("21_merge_star_count", "Merge STAR gene counts", "Synthetic ReadsPerGene tables demonstrate unstranded-column merging and the upstream handling of summary rows. This notebook does not run STAR.",
     '''data = pd.DataFrame({"S1": [6, 12, 24], "S2": [14, 8, 20], "S3": [10, 16, 8]}, index=["ENSG1", "ENSG2", "ENSG3"])
for sample in data:
    rows = "N_unmapped\\t0\\t0\\t0\\nN_multimapping\\t0\\t0\\t0\\nN_noFeature\\t0\\t0\\t0\\nN_ambiguous\\t0\\t0\\t0\\n"
    rows += "".join(f"{gene}\\t{value}\\t{value}\\t{value}\\n" for gene, value in data[sample].items())
    (work / f"{sample}_ReadsPerGene.out.tab").write_text(rows)''',
     'result = iobrx.merge_star_count(work, "demo", n_threads=2)', 'plot = result.select_dtypes("number").T'),
    ("22_tme_profile", "Integrated TME profile", "Public STAD TPM, first two samples; signature_tme panel, RNA-seq settings, no CIBERSORT permutation inference. The original CIBERSORT solver is used. This is a small tutorial workload, not the campaign benchmark.",
     'data = tpm_input().iloc[:, :2]\ndata.to_csv(work / "input.csv")',
     'result = iobrx.tme_profile(work / "input.csv", work / "profile", threads=2, signature="signature_tme", perm=0, QN=False, arrays=False, platform="rnaseq")',
     'plot = pd.read_csv(result["cibersort"], index_col=0).iloc[:, :22]'),
    ("23_bayesprism", "BayesPrism reference demo", "Semi-synthetic counts derived from 128 genes in the bundled single-cell reference. The short Gibbs chain (24 iterations, burn-in 12) is an execution demonstration, NOT a converged biological estimate. Production analyses need adequate chains, convergence assessment and a relevant reference.",
     '''from importlib.resources import files
reference = files("iobrpy.bayesprism.BP_data")
columns = pd.read_csv(str(reference.joinpath("sc_dat.csv")), nrows=0).columns[:129]
sc = pd.read_csv(str(reference.joinpath("sc_dat.csv")), usecols=list(columns), index_col=0)
types = pd.read_csv(str(reference.joinpath("cell_type_labels.csv")), header=None).iloc[:, 0].tolist()
states = pd.read_csv(str(reference.joinpath("cell_state_labels.csv")), header=None).iloc[:, 0].tolist()
rng = np.random.default_rng(42)
rates = sc.mean().to_numpy() * 20 + 1
data = pd.DataFrame(rng.poisson(rates[:, None], size=(len(rates), 3)), index=sc.columns, columns=["Demo1", "Demo2", "Demo3"])''',
     'result = iobrx.bayesprism(data, sc_dat=sc, cell_type_labels=types, cell_state_labels=states, n_threads=2, state_order="sorted", gibbs_control={"chain.length": 24, "burn.in": 12, "thinning": 2})',
     'plot = result["theta"]'),
]


def build(revision, only=None):
    for slug, title, meaning, prepare, analysis, view in CASES:
        if only and slug != only:
            continue
        notebook = nbf.v4.new_notebook()
        notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
                                 iobrx={"figure_revision": revision, "scope": "tutorial; not an upstream speed benchmark"})
        run_code = f'''# Capture upstream progress messages; the computation remains the public API call below.
started = perf_counter()
with warnings.catch_warnings(record=True) as diagnostics, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    {analysis}
elapsed = perf_counter() - started
for warning in diagnostics:
    print(f"{{warning.category.__name__}}: {{warning.message}}")
{view}
tables = result if isinstance(result, dict) else {{"result": result}}
for name, table in tables.items():
    if isinstance(table, pd.DataFrame):
        table.to_csv(work / (name + ".csv"))
plot.to_csv(work / "display_source.csv")
timing = {{"analysis": slug, "input_shape": list(data.shape), "seconds": elapsed, "threads": 2, "scope": "one API call; excludes input preparation", "fixture": {meaning!r}}}
Path("results").mkdir(exist_ok=True)
Path("results", slug + "_timing.json").write_text(json.dumps(timing, indent=2) + "\\n")
print(json.dumps(timing, indent=2))
display(plot.head())'''
        notebook.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{meaning}\n\nRun from the `tutorials` directory in the prepared iobrx environment. Outputs are written to `extension-work/`; the figure uses a display-only feature z-score. No input values are normalized for the analysis unless the visible API parameters request it."),
                          nbf.v4.new_code_cell(SETUP.replace("SLUG", repr(slug))),
                          nbf.v4.new_code_cell(prepare), nbf.v4.new_code_cell(run_code),
                          nbf.v4.new_code_cell(f"configure({revision})\nfig = plot_result(plot, {title!r}, {revision}, groups=groups)\nfor extension in ('png', 'pdf', 'svg'):\n    fig.savefig(Path('figures') / (slug + '.' + extension), dpi=200, bbox_inches='tight')\nreview = Path('review/extensions')\nreview.mkdir(parents=True, exist_ok=True)\nfig.savefig(review / (slug + '-round-{revision}.png'), dpi=120, bbox_inches='tight')\nplt.show()"),
                          nbf.v4.new_markdown_cell("Interpretation: the heatmap highlights relative patterns within each displayed feature. Colors are not cell fractions, statistical significance or cross-feature effect sizes. Full numeric outputs are saved separately. Runtime depends on hardware, input size, parameters and cache state.")]
        path = TUTORIALS / f"{slug}.ipynb"
        NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(TUTORIALS)}}).execute()
        nbf.write(notebook, path)
        print(f"Executed {slug}, figure revision {revision}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--only")
    args = parser.parse_args()
    build(args.revision, args.only)
