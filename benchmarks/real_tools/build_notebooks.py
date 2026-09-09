"""Generate five executable real-data tutorials; run them with the prepared env."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
ITEMS = [
    ('24_real_fastq_qc', 'fastq_qc', 'FASTQ quality control',
     'Two real GEUVADIS samples (ERR188044 and ERR188104), first 100,000 pairs each. fastp and MultiQC must be available. Prefix sampling is for bounded execution, not a representative cohort accuracy claim.'),
    ('25_real_salmon', 'batch_salmon', 'Salmon quantification',
     'The same GEUVADIS reads use the complete GENCODE v44 transcript FASTA. This transcriptome-only index has no genome decoys; it reproduces this comparison, not a universal production reference recommendation. Four-thread Salmon varies even between original-versus-original reruns. The report preserves those differences.'),
    ('26_real_star_bam', 'batch_star_count', 'STAR alignment and BAM comparison',
     'Real yeast RNA-seq DRR392086 / DRR392094, first 30,000 pairs per sample, against the complete S. cerevisiae R64 reference. This tests actual BAM and GeneCounts products, not human TME analysis.'),
    ('27_real_hla_extraction', 'extract_hla_read', 'HLA read extraction from BAM',
     'NA06985 is the public, already HLA-enriched SpecHLA exon example (52,771 pairs). BWA alignment to GRCh38 chromosome 6 creates the real sorted/indexed BAM. Extraction uses prepared tools with auto-installation off. The fixture does not exercise all whole-genome/CRAM or alternate-contig cases.'),
    ('28_real_spechla', 'spechla', 'HLA typing on real reads',
     'NA06985 uses the pinned upstream package\'s SpecHLA assets, IPD-IMGT/HLA 3.38.0, exon mode and four threads. Compare eight loci / sixteen allele calls and FASTAs. Wrapper agreement does not establish genotype accuracy; no independent truth panel is supplied.'),
]


def main():
    for stem, function, title, scope in ITEMS:
        notebook = nbf.v4.new_notebook()
        notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.11'},
                             'iobrx': {'figure_revision': 2, 'dataset_kind': 'public real reads', 'analysis': function}}
        notebook.cells = [nbf.v4.new_markdown_cell(f'# {title}\n\n{scope}\n\n'
            'Prepare the dedicated environment and data using [the real-tool recipe](../benchmarks/real_tools/README.md). '
            'Set `IOBRX_REAL_WORK` to that work directory and launch Jupyter with the same environment. '
            'This notebook executes iobrx again in a fresh output directory. Its demonstration time is separate from the archived three-repeat benchmark. '
            'Tool logs stay in the work directory. No raw patient or private inputs are used.'),
            nbf.v4.new_code_cell('''import json, os, subprocess, sys, time, uuid
from pathlib import Path
from IPython.display import display, Image
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "benchmarks/real_tools").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "benchmarks/real_tools").is_dir(), "Open from the repository or tutorials directory"
sys.path.insert(0, str(ROOT / "benchmarks/real_tools"))
WORK = Path(os.environ.get("IOBRX_REAL_WORK", str(ROOT / "real-tools-work"))).resolve()
assert (WORK / "cases.json").is_file(), "Run benchmarks/real_tools/prepare.py first"
config = json.loads((WORK / "cases.json").read_text())
from run import expand, scientific_products
from figures import make_figure
'''),
            nbf.v4.new_code_cell(f'''case = next(c for c in config["cases"] if c["function"] == {function!r})
OUT = WORK / "notebook-runs" / ({stem!r} + "-" + uuid.uuid4().hex[:8])
OUT.mkdir(parents=True)
parameters = expand(case["parameters"], OUT)
# Public API equivalent: iobrx.{function}(**parameters)
# The tiny worker runs that call with the same interpreter; output goes to a log.
request = OUT.parent / (OUT.name + ".request.json")
request.write_text(json.dumps({{"function": case["function"], "parameters": parameters}}))
env = os.environ.copy()
env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONHASHSEED="0", MPLBACKEND="Agg")
start = time.perf_counter()
with (OUT.parent / (OUT.name + ".log")).open("w") as log:
    subprocess.run([sys.executable, str(ROOT / "benchmarks/real_tools/arm.py"), str(request)],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
demo_seconds = time.perf_counter() - start
products = scientific_products(OUT, case["function"])
print(f"Completed in {{demo_seconds:.3f}} s; inspected {{len(products['files'])}} scientific products")
'''),
            nbf.v4.new_code_cell(f'''evidence = json.loads((ROOT / "benchmarks/real_tools/results/2026-09-09/summary.json").read_text())
benchmark = next(c for c in evidence["cases"] if c["case"]["function"] == {function!r}
                 and "single-thread" not in c["evidence_directory"])
local_summary = WORK / "repetitions" / {function!r} / "summary.json"
if local_summary.is_file():
    benchmark = json.loads(local_summary.read_text())
display(pd.DataFrame([{{"arm": r["arm"], "repeat": r["repeat"], "wall_seconds": r["wall_seconds"],
                       "peak_RSS_MiB": r["peak_rss_kib"] / 1024}} for r in benchmark["runs"]]))
display(pd.DataFrame(benchmark["paired_comparisons"]))
'''),
            nbf.v4.new_markdown_cell('The left panel shows the archived, alternating three-repeat comparison, including process startup. '
                'The right panel uses this notebook\'s actual output. Setup, downloads, indexing and plotting are excluded from benchmark times. '
                'FASTQ comparison uses decompressed record bytes; STAR compares count files and sorted complete SAM records without headers; '
                'HLA typing compares the final allele report and all sixteen FASTAs. Logs and HTML timestamps are not scientific equality targets.'),
            nbf.v4.new_code_cell(f'''baseline_record = next(r for r in benchmark["runs"] if r["arm"] == "iobrpy")
baseline = Path(baseline_record["output"].replace("<WORK>", str(WORK)))
assert baseline.is_dir(), "Run the benchmark recipe to prepare original comparison outputs"
fig = make_figure(benchmark, OUT, baseline, revision=2)
# Export under a stable name; raw analysis outputs keep their unique run directory.
for extension in ("png", "pdf", "svg"):
    fig.savefig(ROOT / "tutorials/figures" / ({stem!r} + "." + extension), dpi=180, bbox_inches="tight")
display(Image(filename=str(ROOT / "tutorials/figures" / ({stem!r} + ".png"))))
import matplotlib.pyplot as plt
plt.close(fig)
'''),
            nbf.v4.new_markdown_cell('See [source accessions, tools, limitations and full repetitions](../benchmarks/real_tools/README.md). '
                'These examples verify the declared inputs and outputs. They do not imply every workflow is faster, every multithreaded result is byte-identical, or HLA calls are independently validated.')]
        nbf.write(notebook, ROOT / 'tutorials' / (stem + '.ipynb'))
    print('Generated five notebooks; execute them before committing.')


if __name__ == '__main__':
    main()
