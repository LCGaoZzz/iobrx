"""Generate the readable tutorial sources; execution is a separate step."""
from pathlib import Path
import argparse
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''
%matplotlib inline
from pathlib import Path
from time import perf_counter
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import iobrx

# Works from the repository root or from the tutorials directory.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "tutorials" / "_common.py").exists())
sys.path.insert(0, str(ROOT / "tutorials"))
from _common import (configure, load, tpm_input, runtime_info, numeric_scores,
                     short_samples, clean_labels, panel, heatmap, scatter,
                     stacked, composition_figure, fractions, save_figure, save_result, COLORS, TEAL, RUST, NAVY)
configure(REVISION)
iobrx.set_threads(8)
display(runtime_info())
'''

SPECS = [
dict(slug="01_gene_annotation", title="Gene annotation and duplicate resolution", zh="基因注释与重复条目处理",
     data='counts = load("eset_stad")\nannotation = load("anno_grch38")\ninput_shape = counts.shape\ndisplay(counts.iloc[:5, :4])',
     call='result = iobrx.anno_eset(counts, annotation, symbol="symbol", probe="id", method="mean")',
     purpose="Map Ensembl identifiers to symbols and inspect which expression rows survive. / 将 Ensembl ID 映射到基因符号，并检查保留情况。",
     caveat="`method='mean'` ranks duplicate candidates by their mean expression and retains the highest-scoring row. It does **not** average those rows into one. Rows that are all zero or all missing are removed according to IOBRpy. Raw sample totals can therefore change; annotation is not normalization.",
     plot='''
mapped = annotation.loc[annotation["id"].isin(counts.index) & annotation["symbol"].notna() & annotation["symbol"].ne("NA_NA")]
counts_summary = pd.Series({"Input features": len(counts), "Mapped symbols": mapped["symbol"].nunique(), "Retained symbols": len(result)})
fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.2), gridspec_kw={"width_ratios": [1, 1.1]}, layout="constrained")
axes[0].barh(counts_summary.index[::-1], counts_summary.values[::-1] / 1000, color=[TEAL, "#9FACA0", NAVY], height=0.55)
axes[0].set(xlabel="Features / symbols (thousands)", xlim=(0, 70))
for y, value in enumerate(counts_summary.values[::-1]):
    axes[0].text(value / 1000 + 1, y, f"{value:,}", va="center", fontsize=7)
scatter(axes[1], counts.sum().to_numpy() / 1e6, result.sum().to_numpy() / 1e6,
        "Input expression total (millions)", "Retained expression total (millions)", annotate=True)
limits = (25, 120)
axes[1].plot(limits, limits, color="#BCC3C5", linestyle="--", linewidth=0.8, zorder=0)
axes[1].set(xlim=limits, ylim=limits)
axes[1].set_aspect("equal", adjustable="box")
panel(axes[0], "A", "Identifier retention")
panel(axes[1], "B", "Sample-level expression retention")
save_figure(fig, SLUG)
'''),
dict(slug="02_counts_to_tpm", title="Count-to-TPM normalization", zh="Count 转 TPM",
     data='counts = load("eset_stad")\ninput_shape = counts.shape\ndisplay(counts.iloc[:5, :4])',
     call='result = iobrx.count2tpm(counts, idType="Ensembl", org="hsa", check_data=True, remove_version=True)',
     purpose="Convert the public STAD count matrix using packaged gene lengths; inspect library totals and expressed-gene distributions.",
     caveat="Use raw, nonnegative gene counts—not log expression or an already normalized TPM matrix. Missing lengths, feature filtering and post-normalization symbol deduplication can leave returned column sums slightly below one million. The notebook reports this explicitly instead of silently renormalizing the output.",
     plot='''
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), layout="constrained")
sample_labels = short_samples(counts.columns)
axes[0].bar(np.arange(len(counts.columns)), counts.sum() / 1e6, color=NAVY, width=0.65)
axes[0].set_xticks(range(len(sample_labels)), sample_labels, rotation=90)
axes[0].set(xlabel="Samples", ylabel="Input count total (millions)")
densities = []
for j in range(result.shape[1]):
    positive = result.iloc[:, j].to_numpy()
    values = np.log10(positive[positive > 0])
    density, edges = np.histogram(values, bins=np.linspace(-5, 6, 75), density=True)
    densities.append(density)
    axes[1].plot((edges[:-1] + edges[1:]) / 2, density, color="#A6B8BC", linewidth=0.7, alpha=0.65, label="Individual sample" if j == 0 else None)
axes[1].plot((edges[:-1] + edges[1:]) / 2, np.median(densities, axis=0), color=TEAL, linewidth=1.6, label="Median density")
axes[1].legend(loc="upper right")
axes[1].set(xlabel="log10(TPM), positive entries only", ylabel="Density", xlim=(-5, 6))
panel(axes[0], "A", "Sequencing-depth variation")
panel(axes[1], "B", "Expressed-gene TPM distributions")
save_figure(fig, SLUG)
display(pd.DataFrame({"sample": counts.columns, "returned_TPM_sum": result.sum().to_numpy(), "retained_percent_of_1e6": result.sum().to_numpy() / 10000, "zero_TPM_percent": (result == 0).mean().to_numpy() * 100}))
'''),
]

SIG_PLOTS = {
"pca": '''
scores = numeric_scores(result).astype(float)
features = scores.std().nlargest(12).index
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
heatmap(axes[0], scores[features].T)
x, y = features[:2]
scatter(axes[1], scores[x], scores[y], *clean_labels([x, y]))
axes[1].axhline(0, color="#DDE1E2", linewidth=0.6, zorder=0)
axes[1].axvline(0, color="#DDE1E2", linewidth=0.6, zorder=0)
panel(axes[0], "A", "Variation in signature-specific PC1 scores")
panel(axes[1], "B", "Two signature axes · all 348 samples")
save_figure(fig, SLUG)
''',
"zscore": '''
scores = numeric_scores(result).astype(float)
features = scores.std().nlargest(8).index
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.35, 1]}, layout="constrained")
vectors = [scores[name].dropna().to_numpy() for name in features[::-1]]
parts = axes[0].violinplot(vectors, vert=False, showextrema=False, widths=0.75)
for body in parts["bodies"]:
    body.set_facecolor(TEAL); body.set_edgecolor("none"); body.set_alpha(0.45)
axes[0].scatter([np.median(v) for v in vectors], np.arange(1, len(vectors)+1), s=13, color=NAVY, zorder=3)
quartiles = np.array([np.percentile(v, [25, 75]) for v in vectors])
axes[0].hlines(np.arange(1, len(vectors)+1), quartiles[:, 0], quartiles[:, 1], color=NAVY, linewidth=1.8)
axes[0].axvline(0, color="#DDE1E2", linewidth=0.6, zorder=0)
axes[0].set_yticks(np.arange(1, len(vectors)+1), clean_labels(features[::-1]))
axes[0].set(xlabel="IOBRpy zscore-method output", ylim=(0.3, 8.7))
x, y = features[:2]
scatter(axes[1], scores[x], scores[y], *clean_labels([x, y]))
panel(axes[0], "A", "Signature distributions · all samples")
panel(axes[1], "B", "Paired signature scores")
save_figure(fig, SLUG)
''',
"ssgsea": '''
scores = numeric_scores(result).astype(float)
features = scores.std().nlargest(12).index
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
heatmap(axes[0], scores[features].T)
selected = features[:5]
for i, name in enumerate(selected):
    values = np.sort(scores[name].to_numpy())
    axes[1].step(values, np.arange(1, len(values)+1) / len(values), where="post", color=COLORS[i], linewidth=1.3, label=clean_labels([name])[0])
axes[1].set(xlabel="Normalized enrichment score (NES)", ylabel="Cumulative fraction of samples", ylim=(0, 1))
axes[1].legend(loc="lower right", ncol=1, fontsize=6.5)
panel(axes[0], "A", "Relative enrichment across samples")
panel(axes[1], "B", "NES distributions · all 348 samples")
save_figure(fig, SLUG)
''',
"integration": '''
scores = numeric_scores(result).astype(float)
suffixes = ["_PCA", "_zscore", "_ssGSEA"]
common = sorted(set(c[:-4] for c in scores if c.endswith("_PCA")) & set(c[:-7] for c in scores if c.endswith("_zscore")) & set(c[:-7] for c in scores if c.endswith("_ssGSEA")))
selected = sorted(common, key=lambda name: scores[name + "_PCA"].std(), reverse=True)[:8]
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), layout="constrained")
for ax, suffix, letter, title in zip(axes, suffixes, "ABC", ["PCA", "zscore", "ssGSEA"]):
    matrix = scores[[name + suffix for name in selected]].T
    matrix.index = selected
    heatmap(ax, matrix, max_rows=8, max_samples=24, row_labels=(letter == "A"), colorbar=(letter == "C"))
    panel(ax, letter, title + " · aligned signatures")
save_figure(fig, SLUG)
print(f"{len(common)} signatures are available in all three methods; display scaling is applied independently to each row.")
''',
}
for number, method, title, zh, caveat in [
    (3,"pca","PCA signature scoring","PCA 基因签名评分","Each signature gets its own PC1 across samples; this is not a single global PCA embedding. PC1 direction is aligned to mean expression as in IOBRpy. Scores from different signatures can have different scales."),
    (4,"zscore","Mean-based signature scoring","zscore 方法基因签名评分","The upstream method named 'zscore' averages the preprocessed expression of each signature. It does not guarantee that each returned signature has mean zero and variance one. Do not confuse its name with the within-feature z-scores used only for heatmap display."),
    (5,"ssgsea","ssGSEA signature enrichment","ssGSEA 基因集富集","The output is NES, including IOBRpy/GSEApy normalization and tie handling. A score summarizes a gene set; it is not a gene-level fold change or a significance p-value. At least five overlapping genes are required for the standalone ssGSEA method."),
    (6,"integration","Integrated signature scoring","PCA、zscore 与 ssGSEA 联合评分","Integration concatenates method-specific scores with _PCA, _zscore and _ssGSEA suffixes. It does not average them into a new biological score. Different methods have different units and should not be compared by raw magnitude."),
]:
    SPECS.append(dict(slug=f"{number:02d}_signature_{method}",title=title,zh=zh,
        data='expression = load("imvigor210_eset")\ninput_shape = expression.shape\ndisplay(expression.iloc[:5, :4])',
        call=f'result = iobrx.calculate_sig_score(expression, "signature_collection", method="{method}", mini_gene_count=3, adjust_eset=True, n_threads=8)',
        purpose="Score the public IMvigor210 demonstration matrix (872 features × 348 samples). This is a restricted feature panel, not a complete transcriptome; only signatures with sufficient overlap are returned.",
        caveat=caveat,plot=SIG_PLOTS[method]))

SPECS += [
dict(slug="07_cibersort",title="CIBERSORT immune composition",zh="CIBERSORT 免疫细胞组成",
     data='tpm = tpm_input()\ninput_shape = tpm.shape\ndisplay(tpm.iloc[:5, :4])',
     call='result = iobrx.cibersort(tpm, perm=100, QN=False, n_threads=8)',
     purpose="Estimate LM22 immune-cell fractions from the full STAD TPM matrix. `QN=False` is explicit for this RNA-seq example; 100 permutations are illustrative.",
     caveat="Relative CIBERSORT fractions describe the modeled immune mixture, not fractions of every cell in the tumor. The permutation p-value evaluates model fit and has a resolution of 1/100 here; it is not a cell-type differential-abundance p-value. Native P-values use a fixed seed; the explicit Python fallback keeps upstream stochastic behavior. The grouped bar chart combines the remaining LM22 types for display only.",
     plot='''
cells = result.drop(columns=["P-value", "Correlation", "RMSE"])
fig, axes, legends = composition_figure(figsize=(10.8, 5.5), ratios=(1, 1.2))
stacked(axes[0], cells, max_types=8, legend_ax=legends[0])
heatmap(axes[1], cells.T, max_rows=22, max_samples=10, scale=False)
panel(axes[0], "A", "Relative immune composition")
panel(axes[1], "B", "All 22 LM22 populations")
save_figure(fig, SLUG)
display(result[["P-value", "Correlation", "RMSE"]])
'''),
dict(slug="08_epic",title="EPIC cell fractions and mRNA proportions",zh="EPIC 细胞比例与 mRNA 比例",
     data='tpm = tpm_input()\ninput_shape = tpm.shape\ndisplay(tpm.iloc[:5, :4])',
     call='result = iobrx.epic(tpm)',
     purpose="Run EPIC with the packaged TRef reference, inspect both output scales, and retain fit diagnostics.",
     caveat="Cell fractions and mRNA proportions differ because of cell-type RNA-content correction. `otherCells` is an uncharacterized remainder; it must not automatically be labeled tumor purity. This example uses a full transcriptome rather than the limited IMvigor210 panel.",
     plot='''
cells, mrna = result["cellFractions"], result["mRNAProportions"]
fig, axes, legends = composition_figure(figsize=(9.3, 4.4), ratios=(1, 1))
stacked(axes[0], cells, max_types=12, legend_ax=legends[0])
order = cells.mean().sort_values().index
y = np.arange(len(order))
axes[1].hlines(y, mrna[order].mean(), cells[order].mean(), color="#B6B8BB", linewidth=1)
axes[1].scatter(mrna[order].mean(), y, s=23, color=NAVY, label="mRNA proportion")
axes[1].scatter(cells[order].mean(), y, s=23, color=RUST, label="Cell fraction")
axes[1].set_yticks(y, clean_labels(order))
axes[1].set(xlabel="Cohort mean proportion", xlim=(-0.02, 1.02))
axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
axes[1].grid(axis="x", color="#E9ECEC", linewidth=0.5)
axes[1].set_axisbelow(True)
legends[1].legend(*axes[1].get_legend_handles_labels(), loc="upper left")
panel(axes[0], "A", "Estimated cell composition")
panel(axes[1], "B", "Effect of RNA-content correction")
save_figure(fig, SLUG)
display(result["fit_gof"].iloc[:, :5])
'''),
dict(slug="09_quantiseq",title="quanTIseq immune deconvolution",zh="quanTIseq 免疫细胞反卷积",
     data='tpm = tpm_input()\ninput_shape = tpm.shape\ndisplay(tpm.iloc[:5, :4])',
     call='result = iobrx.quantiseq(tpm, arrays=False, tumor=True, mRNAscale=True, method="lsei", rmgenes="default")',
     purpose="Use the TIL10 reference with explicit tumor RNA-seq settings and RNA-content correction.",
     caveat="Use linear-scale expression and compatible human symbols. The uncharacterized 'Other' fraction is a model remainder, not a confirmed cell identity. Cell fractions from different deconvolution methods use different reference sets and must not be compared as interchangeable ground truth.",
     plot='''
cells = fractions(result, id_column="Sample")
fig, axes, legends = composition_figure(figsize=(10.2, 4.9), ratios=(1.1, 1.2))
stacked(axes[0], cells, max_types=12, legend_ax=legends[0])
immune_cells = cells.drop(columns=[c for c in cells if c.lower() == "other"])
heatmap(axes[1], immune_cells.T, max_rows=12, max_samples=10, scale=False)
panel(axes[0], "A", "TIL10 composition and remainder")
panel(axes[1], "B", "Ten immune populations · Other excluded")
save_figure(fig, SLUG)
'''),
dict(slug="10_mcpcounter",title="MCP-counter population scores",zh="MCP-counter 细胞群丰度评分",
     data='tpm = tpm_input()\nlog_expression = np.log2(tpm + 1)\ninput_shape = log_expression.shape\ndisplay(log_expression.iloc[:5, :4])',
     call='result = iobrx.mcpcounter(log_expression, features_type="HUGO_symbols")',
     purpose="Estimate marker-based abundance scores from log2(TPM + 1), using human gene symbols.",
     caveat="MCP-counter produces abundance scores, not fractions. Compare a given population across samples, rather than interpreting scores across different populations as percentages. Heatmap row scaling is a visualization transform only.",
     plot='''
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), gridspec_kw={"width_ratios": [1.4, 1]}, layout="constrained")
heatmap(axes[0], result, max_rows=10, max_samples=10)
names = list(result.index)
xname = next((x for x in names if str(x) == "T cells"), names[0])
yname = next((x for x in names if "Cytotoxic" in str(x)), names[1])
scatter(axes[1], result.loc[xname].to_numpy(), result.loc[yname].to_numpy(), str(xname) + " score", str(yname) + " score", annotate=True)
axes[1].set_box_aspect(1)
panel(axes[0], "A", "Within-population sample variation")
panel(axes[1], "B", "Two immune abundance scores")
save_figure(fig, SLUG)
'''),
dict(slug="11_estimate",title="ESTIMATE stromal and immune scores",zh="ESTIMATE 基质与免疫评分",
     data='tpm = tpm_input()\nlog_expression = np.log2(tpm + 1)\ninput_shape = log_expression.shape\ndisplay(log_expression.iloc[:5, :4])',
     call='result = iobrx.estimate_score(log_expression, platform="rnaseq")',
     purpose="Calculate stromal, immune and combined ESTIMATE scores for the STAD RNA-seq example.",
     caveat="These are enrichment scores, not cell percentages. Only the exact upstream platform string 'affymetrix' requests its calibrated tumor-purity transformation; 'affy' does not. This RNA-seq tutorial deliberately shows scores without applying the Affymetrix purity calibration.",
     plot='''
stromal = next(name for name in result.index if "stromal" in str(name).lower())
immune = next(name for name in result.index if "immune" in str(name).lower())
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), layout="constrained")
scatter(axes[0], result.loc[stromal].to_numpy(), result.loc[immune].to_numpy(), "Stromal score", "Immune score", annotate=True)
total = result.loc["ESTIMATEScore"]
order = np.argsort(total.to_numpy())
labels = np.array(short_samples(total.index))[order]
axes[1].hlines(np.arange(len(order)), 0, total.iloc[order], color="#CBD9D7", linewidth=2)
axes[1].scatter(total.iloc[order], np.arange(len(order)), s=23, c=TEAL, zorder=3)
axes[1].set_yticks(np.arange(len(order)), labels)
axes[1].set(xlabel="Combined ESTIMATE score", ylabel="Samples (ordered by score)")
axes[1].spines["left"].set_visible(False)
axes[1].tick_params(axis="y", length=0)
axes[1].set_xticks([0, 1000, 2000, 3000])
panel(axes[0], "A", "Stromal and immune enrichment")
panel(axes[1], "B", "Combined score across samples")
save_figure(fig, SLUG)
'''),
dict(slug="12_complete_workflow",title="A complete, inspectable TME workflow",zh="完整的 TME 分析工作流",
     data='counts = load("eset_stad")\nannotation = load("anno_grch38")\ninput_shape = counts.shape\ndisplay(counts.iloc[:5, :4])',
     call='''
result, stage_seconds = {}, {}
def run_stage(name, function):
    start = perf_counter()
    output = function()
    stage_seconds[name] = perf_counter() - start
    result[name] = output
    return output

symbols = run_stage("annotation", lambda: iobrx.anno_eset(counts, annotation, method="mean"))
tpm = run_stage("TPM", lambda: iobrx.count2tpm(counts, check_data=True, remove_version=True))
log_expression = np.log2(tpm + 1)
scores = run_stage("signatures", lambda: iobrx.calculate_sig_score(log_expression, "signature_collection", "integration", n_threads=8))
cib = run_stage("CIBERSORT", lambda: iobrx.cibersort(tpm, perm=100, QN=False, n_threads=8))
epi = run_stage("EPIC", lambda: iobrx.epic(tpm))
qnt = run_stage("quanTIseq", lambda: iobrx.quantiseq(tpm, tumor=True, rmgenes="default"))
mcp = run_stage("MCP-counter", lambda: iobrx.mcpcounter(log_expression))
est = run_stage("ESTIMATE", lambda: iobrx.estimate_score(log_expression, platform="rnaseq"))
''',
     purpose="Follow counts → TPM → method-specific expression inputs. Each stage is an explicit iobrx call with its own wall-clock timer; loading and plotting are outside the analysis timer.",
     caveat="Annotation is shown as a separate workflow branch; count2tpm already performs its own identifier mapping. Do not feed log expression to methods expecting linear TPM. The overview scales features only for display, because deconvolution fractions and marker/enrichment scores have different units. No response, survival, or clinical groups are inferred from these unlabeled fixtures.",
     plot='''
cell_columns = [c for c in cib if c not in ["P-value", "Correlation", "RMSE"]]
cd8 = next(c for c in cell_columns if "CD8" in c)
epic_cd8 = next(c for c in epi["cellFractions"] if "CD8" in c)
immune = next(name for name in est.index if "immune" in str(name).lower())
summary = pd.DataFrame({"CD8 · CIBERSORT": cib[cd8], "CD8 · EPIC": epi["cellFractions"][epic_cd8],
                        "T cells · MCP-counter": mcp.loc["T cells"], "Immune · ESTIMATE": est.loc[immune]})
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1.4, 1]}, layout="constrained")
heatmap(axes[0], summary.T, max_rows=4, max_samples=10)
timing = pd.Series(stage_seconds).sort_values()
axes[1].hlines(np.arange(len(timing)), 0.001, timing, color="#CBD9D7", linewidth=2)
axes[1].scatter(timing, np.arange(len(timing)), s=23, c=TEAL, zorder=3)
axes[1].set_yticks(np.arange(len(timing)), timing.index)
axes[1].set(xlabel="Wall time (seconds, log scale)", xscale="log", xlim=(0.001, timing.max() * 4))
axes[1].set_xticks([0.001, 0.01, 0.1, 1, 10, 100], ["0.001", "0.01", "0.1", "1", "10", "100"])
axes[1].minorticks_off()
axes[1].spines["left"].set_visible(False)
axes[1].tick_params(axis="y", length=0)
for y, value in enumerate(timing):
    label = f"{value * 1000:.0f} ms" if value < 1 else f"{value:.2f} s"
    axes[1].annotate(label, (value, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7)
panel(axes[0], "A", "Aligned sample overview")
panel(axes[1], "B", "Observed stage times")
save_figure(fig, SLUG)
display(pd.Series(stage_seconds, name="seconds").to_frame())
'''),
]


def block(text):
    return textwrap.dedent(text).strip() + "\n"


def build(revision):
    output = ROOT / "tutorials"
    output.mkdir(exist_ok=True)
    for spec in SPECS:
        slug = spec["slug"]
        cells = [
            nbf.v4.new_markdown_cell(f'# {spec["title"]}\n\n**{spec["zh"]}**\n\n{spec["purpose"]}\n\nThis notebook uses public [IOBR release fixtures](https://github.com/IOBR/IOBR/releases/tag/data-v1.0). Source URLs and checksums are in [data/manifest.json](data/manifest.json). Figures are descriptive; no clinical outcome labels are supplied.\n\n**Setup:** from the repository root, install with `python -m pip install ".[tutorials]"`. Restart the notebook kernel after installation.'),
            nbf.v4.new_code_cell(block(SETUP.replace("REVISION", str(revision)))),
            nbf.v4.new_markdown_cell('## 1. Inspect the input / 检查输入\n\nExpression matrices are **genes × samples**. The analysis uses every sample; any figure subset is stated explicitly.'),
            nbf.v4.new_code_cell(block(spec["data"])),
            nbf.v4.new_markdown_cell('## 2. Run the analysis / 运行分析\n\nThe timer covers the call below, including its resource setup. Data loading above and plotting below are excluded. This is a single run; repeated benchmark medians are reported separately in the README.'),
            nbf.v4.new_code_cell('started = perf_counter()\n' + block(spec["call"]) + 'elapsed = perf_counter() - started\n' + f'display(save_result(result, "{slug}", elapsed, input_shape))'),
            nbf.v4.new_code_cell('display(result.head() if isinstance(result, pd.DataFrame) else {key: value.shape if isinstance(value, pd.DataFrame) else list(value) for key, value in result.items()})'),
            nbf.v4.new_markdown_cell('## 3. Visualize and export / 作图与导出\n\nWhite background, restrained colors, thin axes and editable vector text. Heatmap z-scores are display-only; exported result tables keep the original values. `S01`, `S02`, … follow the sample order of each displayed matrix; the mapping is printed below.\n\nFor the IMvigor210 heatmaps, only the first 30 samples are shown (24 per panel in the integration notebook); scoring and distributions use all 348.' if 'signature' in slug else '## 3. Visualize and export / 作图与导出\n\nWhite background, restrained colors, thin axes and editable vector text. Display transformations do not overwrite the analysis result. `S01`, `S02`, … follow the input sample order, as recorded in the mapping table below.'),
            nbf.v4.new_code_cell(block(spec["plot"].replace("SLUG", repr(slug)))),
            nbf.v4.new_code_cell('sample_ids = expression.columns if "expression" in globals() else counts.columns if "counts" in globals() else tpm.columns\ndisplay(pd.DataFrame({"plot_label": short_samples(sample_ids), "original_sample_id": sample_ids}).head(30))'),
            nbf.v4.new_markdown_cell(f'## 4. Interpretation and limits / 解读与边界\n\n{spec["caveat"]}\n\n**Reusing this tutorial:** replace the input-loading cell with your own `pd.read_csv(..., index_col=0)`; keep the matrix orientation, identifier type and expression scale consistent with this example. Check gene overlap before interpreting estimates.\n\nFigures are written to `tutorials/figures/{slug}.png`, `.pdf`, and `.svg`. Compact output tables and timing metadata are written to `tutorials/results/`. Large count/TPM matrices remain in `result`; save them explicitly with `result.to_parquet(...)` if needed.'),
        ]
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "iobrx": {"figure_revision": revision, "source": "scripts/build_tutorials.py"},
        })
        # Stable cell IDs keep subsequent revisions reviewable.
        for index, cell in enumerate(notebook.cells):
            cell.id = f"{slug[:24]}-{index:02d}"
        nbf.write(notebook, output / f"{slug}.ipynb")
        print(slug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, default=2, choices=[0,1,2])
    build(parser.parse_args().revision)
