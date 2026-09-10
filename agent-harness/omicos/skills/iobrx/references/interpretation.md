# Interpretation and figures

Cell fractions from different deconvolution methods are model-dependent
estimates. Keep LM22, EPIC and TIL10 results separate. EPIC's mRNA proportions
are not its cell fractions. CIBERSORT absolute scores are not interchangeable
with relative weights. Its P-values depend on permutation settings: `perm=0`
disables inference. Native permutations use seed 0; the Python fallback uses
the upstream RNG. Exact parity with IOBRpy excludes stochastic P-values.

MCP-counter produces marker abundance scores, not proportions; compare the
same population across appropriately normalized samples, not scores between
different cell types. Signature PCA, mean-based `zscore`, ssGSEA and integration
produce different score scales. Do not describe IOBRpy's mean-based zscore
entry as a standardized statistical z score. ESTIMATE adds the upstream purity
formula only for `platform=affymetrix`; that formula is not a generic purity
validation for another assay.

Check missing/non-finite outputs and gene coverage before interpretation.
Input schema validation does not prove correct assay normalization or adequate
signature coverage. Do not infer a treatment response or a clinical decision
from a deconvolution fraction alone.

The 23 executed tutorials, including the complete workflow, are at:
https://github.com/LCGaoZzz/iobrx/tree/main/tutorials

They include public data, timings, explanatory code and PNG/PDF/SVG figures
reviewed twice. For presentation, use white backgrounds, restrained colors,
clear units, thin axes and editable vector text. Choose stacked bars for
fractions, heatmaps for within-method scores and diagnostic panels for fit
metrics. Avoid pooling incomparable methods on one numerical axis. Read
results from the manifest's Parquet files; plotting need not rerun solvers.

When vision tools are unavailable for reviewing a generated figure, write
the plotted data to a CSV beside the figure (same base name) and verify every
quantitative claim against that CSV — row counts, column sums, extrema and
group orderings. This data-driven self-review is the standard fallback and
often catches axis or subset mistakes the figure itself would not show.
