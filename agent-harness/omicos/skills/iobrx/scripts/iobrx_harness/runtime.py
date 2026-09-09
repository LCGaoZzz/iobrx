"""Validated file adapter over the public iobrx API. No analysis reimplementation."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from . import SKILL_ID, __version__
from .catalog import CATALOG, LEGACY_ANALYSES, request_schema


class HarnessError(Exception):
    def __init__(self, message, code="invalid_request", exit_code=2):
        super().__init__(message)
        self.code, self.exit_code = code, exit_code


def failure(exc):
    return {"schema_version": "1.0", "skill_id": SKILL_ID, "status": "failed",
            "error": {"code": getattr(exc, "code", "execution_failed"),
                      "message": str(exc), "type": type(exc).__name__}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: reject(f"Non-finite JSON value: {value}"))


def reject(message):
    raise HarnessError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_path(value, base, workspace=None):
    path = (Path(base) / value).resolve()
    if workspace is not None and not path.is_relative_to(Path(workspace).resolve()):
        reject(f"Path is outside the configured workspace: {value}")
    return path


def normalize_request(request, base, workspace=None):
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(request_schema()).iter_errors(request),
                    key=lambda error: str(list(error.absolute_path)))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "request"
        reject(f"{location}: {error.message}")
    result = deepcopy(request)
    spec = CATALOG[result["analysis"]]
    params = result.setdefault("parameters", {})
    for name, field in spec["parameters"]["properties"].items():
        if name not in params and "default" in field:
            params[name] = deepcopy(field["default"])
    result.setdefault("threads", min(8, os.cpu_count() or 1))
    result["input"]["path"] = str(resolve_path(result["input"]["path"], base, workspace))
    result["output_dir"] = str(resolve_path(result["output_dir"], base, workspace))
    if result["analysis"] not in LEGACY_ANALYSES:
        from .extended_runtime import normalize
        return normalize(result, base, workspace)
    if result["input"]["gene_id"] == "mgi" and result["input"]["organism"] != "mmus":
        reject("mgi identifiers require organism=mmus")
    if result["analysis"] == "anno_eset":
        is_ensembl = params["annotation"] in {"anno_grch38", "anno_rnaseq"}
        if is_ensembl != (result["input"]["gene_id"] == "ensembl"):
            reject("Annotation resource and declared gene_id disagree")
    return result


def load_matrix(spec):
    import numpy as np
    import pandas as pd

    path = Path(spec["path"])
    if not path.is_file():
        reject(f"Input file does not exist: {path}")
    digest = sha256(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle, delimiter=delimiter), [])
        if len(header) < 2 or len(set(header[1:])) != len(header[1:]):
            reject("Matrix needs an index column and unique data-column labels")
        # Keep literal identifiers (e.g. 001, NA) instead of pandas' type/NA inference.
        raw = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, header=None, skiprows=1)
        if raw.shape[1] != len(header):
            reject("CSV/TSV row width does not match its header")
        matrix = raw.iloc[:, 1:].copy()
        matrix.columns = header[1:]
        matrix.index = raw.iloc[:, 0].to_numpy()
    elif path.suffix.lower() == ".parquet":
        matrix = pd.read_parquet(path)
        if isinstance(matrix.index, pd.RangeIndex):
            reject("Parquet must store feature/sample identifiers in its index")
    else:
        reject("Supported matrix formats are .csv, .tsv and .parquet; pickle is not accepted")
    if spec["orientation"] == "samples_by_genes":
        matrix = matrix.T
    if matrix.empty:
        reject("Expression matrix must contain features and samples")
    for axis, labels in (("gene", matrix.index), ("sample", matrix.columns)):
        if isinstance(labels, pd.MultiIndex) or any(pd.isna(x) or not str(x).strip() for x in labels):
            reject(f"Missing or hierarchical {axis} identifiers are unsupported")
        converted = pd.Index(labels.map(str))
        if converted.has_duplicates:
            reject(f"Duplicate {axis} identifiers: resolve them explicitly before analysis")
        if any(x != x.strip() for x in converted):
            reject(f"Whitespace around {axis} identifiers: clean it explicitly before analysis")
        if axis == "gene":
            matrix.index = converted
        else:
            matrix.columns = converted
    try:
        matrix = matrix.apply(pd.to_numeric, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        reject(f"Expression contains non-numeric values: {exc}")
    values = matrix.to_numpy()
    if not np.isfinite(values).all():
        reject("Expression contains NaN or infinity; resolve missing data explicitly")
    if spec.get("scale") == "preprocessed":
        if (values == 0).all(axis=0).any():
            reject("Preprocessed expression has an all-zero sample")
    elif (values < 0).any() or (values.sum(axis=0) <= 0).any():
        reject("Expression must be nonnegative and each sample must have a positive total")
    if sha256(path) != digest:
        reject("Input changed while it was being loaded")
    info = {**spec, "sha256": digest, "size_bytes": path.stat().st_size,
            "features": int(matrix.shape[0]), "samples": int(matrix.shape[1]),
            "loaded_orientation": "genes_by_samples", "minimum": float(values.min()),
            "maximum": float(values.max())}
    return matrix, info


def doctor():
    import iobrx
    import shutil
    from importlib.resources import files

    versions = {name: importlib.metadata.version(name) for name in
                ("iobrpy", "numpy", "pandas", "scipy", "scikit-learn", "gseapy", "pyarrow")}
    if not callable(getattr(iobrx, "backend_info", None)):
        raise HarnessError("This interpreter does not import a complete iobrx installation", "environment_error", 3)
    resources = files("iobrpy.resources")
    for filename in ("lm22.txt", "calculate_data.pkl", "epic_TRef_BRef.pkl", "quantiseq_data.pkl",
                     "mcp_data.pkl", "estimate_data.pkl", "anno_eset.pkl", "count2tpm_data.pkl"):
        if not resources.joinpath(filename).is_file():
            raise HarnessError(f"Missing bundled IOBRpy reference: {filename}", "environment_error", 3)
    source = Path(__file__).resolve().parent
    source_hash = hashlib.sha256()
    for path in sorted(source.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    return {"schema_version": "1.0", "skill_id": SKILL_ID, "status": "completed",
            "harness_version": __version__, "harness_source_sha256": source_hash.hexdigest(),
            "iobrx_version": iobrx.__version__, "iobrx_path": str(Path(iobrx.__file__).resolve()),
            "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(), "versions": versions,
            "backend": iobrx.backend_info(),
            "external_tools": {tool: shutil.which(tool) for tool in ("fastp", "multiqc", "salmon", "STAR", "run-trust4")},
            "checks": "imports and bundled resources; external tools are optional until requested; not a numerical parity test"}


def validate(request, base, workspace=None):
    normalized = normalize_request(request, base, workspace)
    _, info = prepare_input(normalized)
    return {"schema_version": "1.0", "skill_id": SKILL_ID, "status": "validated",
            "request": normalized, "input": info,
            "output_available": not Path(normalized["output_dir"]).exists(),
            "checks": "schema, input contracts and tool availability; no biological scale inference or solver run"}


def prepare_input(request):
    if request["analysis"] in LEGACY_ANALYSES:
        return load_matrix(request["input"])
    from .extended_runtime import prepare
    return prepare(request)


def dispatch(request, matrix):
    import iobrx

    if request["analysis"] not in LEGACY_ANALYSES:
        from .extended_runtime import dispatch as extended_dispatch
        return extended_dispatch(request, matrix)

    name = request["analysis"]
    params = deepcopy(request["parameters"])
    iobrx.set_threads(request["threads"])
    backend = "iobrx-python"
    notes = []
    if request["input"]["scale"] == "preprocessed":
        notes.append("Preprocessed signed expression was used as supplied; normalization provenance is user-declared, not inferred.")
    if name == "cibersort" or name.startswith("signature_"):
        from iobrx._backend import select_backend

        requested = params["backend"]
        native = select_backend(requested)
        if native and name == "cibersort":
            from iobrx._fast.cibersort_fast import _init_blas
            try:
                _init_blas()
            except (RuntimeError, OSError, AttributeError):
                if requested == "rust":
                    raise
                native = False
        backend = "rust" if native else "python"
        params["backend"] = backend
        params["n_threads"] = request["threads"]
        if requested == "auto" and not native:
            notes.append("Native acceleration unavailable; used the IOBRpy Python backend.")
    if name.startswith("signature_"):
        result = iobrx.calculate_sig_score(matrix, method=name.removeprefix("signature_"), **params)
    elif name == "count2tpm":
        ids = {"ensembl": "Ensembl", "entrez": "entrez", "symbol": "symbol", "mgi": "mgi"}
        result = iobrx.count2tpm(matrix, idType=ids[request["input"]["gene_id"]],
                                org=request["input"]["organism"], **params)
    elif name == "epic":
        import pandas as pd
        from importlib.resources import files

        # Only the trusted, installed IOBRpy reference is unpickled, never user input.
        reference = pd.read_pickle(str(files("iobrpy.resources").joinpath("epic_TRef_BRef.pkl")))[params.pop("reference")]
        result = iobrx.epic(matrix, reference=reference, **params)
    elif name == "mcpcounter":
        feature_types = {"symbol": "HUGO_symbols", "entrez": "ENTREZ_ID", "probe": "affy133P2_probesets"}
        result = iobrx.mcpcounter(matrix, features_type=feature_types[request["input"]["gene_id"]])
    else:
        result = getattr(iobrx, CATALOG[name]["function"])(matrix, **params)
    if name == "cibersort":
        notes.append("CIBERSORT P-values are not covered by exact parity; native permutations use seed 0, Python uses upstream RNG.")
        if params["perm"] == 0:
            notes.append("perm=0 disables permutation inference; do not interpret its P-value column.")
    return result, backend, notes


def save_results(result, output, analysis):
    import numpy as np
    import pandas as pd

    tables = result if isinstance(result, dict) else {"result": result}
    artifacts, notes = [], []
    for name, frame in tables.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise HarnessError(f"{name}: analysis returned an empty or non-tabular result", "invalid_result", 3)
        if not name.replace("_", "").isalnum():
            raise HarnessError("Unsafe output table name", "invalid_result", 3)
        numeric = frame.select_dtypes(include="number")
        finite_count = int(np.isfinite(numeric.to_numpy()).sum())
        text_table = analysis == "nmf" and name == "top_features"
        if not text_table and (numeric.empty or finite_count == 0):
            raise HarnessError(f"{name}: no finite numerical results; check gene overlap and scale", "invalid_result", 3)
        nonfinite = int(numeric.size - finite_count)
        if nonfinite:
            notes.append(f"{name} contains {nonfinite} non-finite values; inspect method coverage and diagnostics.")
        metadata = {"table": name, "rows": len(frame), "columns": len(frame.columns),
                    "layout": CATALOG[analysis]["output_layout"], "nonfinite_numeric_values": nonfinite,
                    "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()}}
        # Parquet retains dtypes/index and avoids CSV round-trip precision loss.
        for extension in ("parquet", "csv"):
            path = output / f"{name}.{extension}"
            if extension == "parquet":
                frame.to_parquet(path)
            else:
                frame.to_csv(path)
            artifacts.append({**metadata, "path": path.name, "format": extension,
                              "sha256": sha256(path), "size_bytes": path.stat().st_size})
    return artifacts, notes


def run(request, base, workspace=None):
    started = time.perf_counter()
    normalized = normalize_request(request, base, workspace)
    output = Path(normalized["output_dir"])
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise HarnessError("output_dir already exists; choose a new run directory", "output_exists", 4) from exc
    manifest_path = output / "results_manifest.json"
    manifest = {"schema_version": "1.0", "skill_id": SKILL_ID, "status": "running",
                "started_at": utcnow(), "request": normalized, "artifacts": [], "warnings": [],
                "manifest_path": str(manifest_path)}
    write_json(manifest_path, manifest)
    write_json(output / "request.json", normalized)
    try:
        environment = doctor()
        matrix, info = prepare_input(normalized)
        manifest.update(environment=environment, input=info)
        write_json(manifest_path, manifest)
        execution_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result, backend, notes = dispatch(normalized, matrix)
        notes.extend(dict.fromkeys(f"{item.category.__name__}: {item.message}" for item in captured))
        manifest["analysis_seconds"] = time.perf_counter() - execution_start
        if normalized["analysis"] not in LEGACY_ANALYSES:
            from .extended_runtime import verify_inputs
            verify_inputs(normalized, info)
        artifacts, result_notes = save_results(result, output, normalized["analysis"]) if result is not None else ([], [])
        if normalized["analysis"] not in LEGACY_ANALYSES:
            from .extended_runtime import artifacts as native_artifacts
            artifacts.extend(native_artifacts(output))
        if not artifacts:
            raise HarnessError("Analysis produced no artifacts", "invalid_result", 3)
        manifest.update(status="completed", backend_used=backend, artifacts=artifacts, warnings=notes + result_notes)
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        manifest.update(failure(exc))
        if isinstance(exc, KeyboardInterrupt):
            manifest["status"] = "interrupted"
        manifest["exit_code"] = 130 if isinstance(exc, KeyboardInterrupt) else getattr(exc, "exit_code", 3)
    manifest.update(finished_at=utcnow(), elapsed_seconds=time.perf_counter() - started)
    write_json(manifest_path, manifest)
    return manifest


def status(path, base, workspace=None):
    path = resolve_path(path, base, workspace)
    if path.is_dir():
        path = resolve_path("results_manifest.json", path, workspace)
    manifest = read_json(path)
    if manifest.get("skill_id") != SKILL_ID or manifest.get("schema_version") != "1.0":
        reject("Not an iobrx harness v1 manifest")
    mismatches = []
    for artifact in manifest.get("artifacts", []):
        target = resolve_path(artifact["path"], path.parent, path.parent)
        if not target.is_file() or sha256(target) != artifact["sha256"]:
            mismatches.append(artifact["path"])
    manifest["integrity"] = {"checked": len(manifest.get("artifacts", [])), "mismatches": mismatches}
    if mismatches:
        manifest["recorded_status"] = manifest["status"]
        manifest["status"] = "failed"
        manifest["error"] = {"code": "artifact_changed", "message": "Result artifacts no longer match the recorded hashes"}
    return manifest
