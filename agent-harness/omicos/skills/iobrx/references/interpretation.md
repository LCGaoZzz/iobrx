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


When image inspection is unavailable, review the plotted source tables instead:
verify sample/order alignment, labels, units, missing values and method-specific
diagnostics. Relative fraction sums can be useful checks, but exclude diagnostic
columns and do not apply fraction rules to ESTIMATE/MCP/signature scores.
Record that data-level checks were performed; they do not establish that the
rendered figure is unclipped, legible or visually reviewed. Do not rerun a solver
just to recreate a plot from existing result tables.
