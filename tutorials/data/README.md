# Public example matrices

These fixtures come from the [IOBR data-v1.0 release](https://github.com/IOBR/IOBR/releases/tag/data-v1.0).
They were read from the upstream R data files with `pyreadr` and converted to
Parquet without changing matrix values, row identifiers, or sample identifiers.
[manifest.json](manifest.json) records the original URL, dimensions, SHA-256 of
the downloaded R file, and SHA-256 of the committed Parquet file.

| File | Rows × columns | Use |
| --- | --- | --- |
| `imvigor210_eset.parquet` | 872 × 348 | Restricted, preprocessed IMvigor210 expression panel; signature examples |
| `eset_stad.parquet` | 60,483 × 10 | Public TCGA-STAD count fixture; annotation, TPM and TME examples |
| `anno_grch38.parquet` | 66,533 × 11 | Upstream GRCh38 annotation table |
| `eset_blca.parquet` | 60,483 × 5 | Independent public BLCA validation fixture |

The restricted IMvigor210 panel is unsuitable for full-transcriptome
deconvolution references with insufficient overlap. No clinical outcome labels
are included, generated, or inferred in these tutorials.

## Attribution and terms

Credit the IOBR authors and the original datasets when using these fixtures.
IOBR declares **GPL-3** in its [DESCRIPTION](https://github.com/IOBR/IOBR/blob/master/DESCRIPTION).
The redistributed upstream fixtures remain under those upstream terms, with
the [GPL v3 text](LICENSE) included here; they are **not relicensed as MIT** by
the iobrx repository license. Consult the upstream release for dataset-specific
provenance and citation guidance. The source `.rda` files remain available at
the URLs in the manifest.
