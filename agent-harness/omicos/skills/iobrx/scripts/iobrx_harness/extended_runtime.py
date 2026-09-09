"""File contracts for the additional public APIs. No scientific algorithms."""
from pathlib import Path
import shutil

from .catalog import CATALOG
from .runtime import HarnessError, file_metadata, load_matrix, reject, resolve_path, sha256


PATH_FIELDS = ("path", "index", "gtf", "f", "ref")


def normalize(request, base, workspace):
    from iobrx._run_state import walk_paths
    inp = request["input"]
    output = Path(request["output_dir"])
    for key in PATH_FIELDS:
        if key not in inp:
            continue
        path = resolve_path(inp[key], base, workspace)
        inp[key] = str(path)
        if not path.exists():
            reject(f"Missing {key}: {path}")
        if output == path or (path.is_dir() and output.is_relative_to(path)):
            reject("output_dir must be outside every input/reference directory")
        # Check children too: resolving just the root permits symlink escapes.
        for child in walk_paths(path) if path.is_dir() else ():
            resolved = resolve_path(str(child), base, workspace)
            if resolved.is_dir() and output.is_relative_to(resolved):
                reject("output_dir must be outside every input/reference directory")
    return request


def prepare(request):
    import numpy as np
    import pandas as pd
    from iobrx._run_state import inventory

    name, inp, params = request["analysis"], request["input"], request["parameters"]
    kind = inp.get("kind", "matrix")
    path = Path(inp["path"])
    matrix = None
    if kind in {"matrix", "feature_table"}:
        spec = inp if kind == "matrix" else {**inp, "orientation": "samples_by_genes", "scale": "preprocessed"}
        matrix, info = load_matrix(spec, request["provenance"])
        if kind == "feature_table":
            matrix = matrix.T
            info.update(loaded_orientation="samples_by_features", samples=len(matrix), features=len(matrix.columns))
            lower, upper = (params["kmin"], params["kmax"]) if name == "nmf" else (params["min_nc"], params["max_nc"])
            if lower > upper or upper + (1 if name == "tme_cluster" else 0) >= len(matrix):
                reject("Invalid cluster range for the number of samples")
            if name == "nmf" and (matrix.to_numpy() < 0).any():
                reject("NMF requires nonnegative features; transform them explicitly before running")
        if name == "bayesprism":
            values = matrix.to_numpy()
            if (values != np.floor(values)).any() or (values > np.iinfo(np.int32).max).any():
                reject("BayesPrism requires integer counts within int32 range")
    else:
        is_directory = kind.endswith("_directory")
        if is_directory != path.is_dir():
            reject(f"{kind} requires a {'directory' if is_directory else 'file'}")
        if kind == "salmon_directory":
            products = list(path.rglob("quant.sf"))
        elif kind == "star_directory":
            products = list(path.glob("*ReadsPerGene.out.tab"))
        elif kind == "fastq_directory":
            suffix = params.get("suffix1", "_1.fastq.gz")
            products = list(path.glob("*" + suffix))
            if not params.get("se", False):
                suffix2 = suffix.replace("1", "2")
                if suffix2 == suffix:
                    reject("Paired FASTQ suffix1 must distinguish R1 from R2 using 1/2")
                for read1 in products:
                    mate = read1.with_name(read1.name[:-len(suffix)] + suffix2)
                    if not mate.is_file() or not mate.stat().st_size:
                        reject(f"Missing or empty FASTQ mate: {mate}")
        else:
            products = [path]
        if not products or any(not item.stat().st_size for item in products):
            reject(f"No nonempty {kind} inputs found")
        if kind == "salmon_matrix":
            frame = pd.read_csv(path, sep="\t", nrows=5)
            if "Name" not in frame or not frame["Name"].astype(str).str.count(r"\|").ge(7).all():
                reject("prepare_salmon requires GENCODE pipe-delimited annotations in the Name column")
        info = {**inp, "input_products": len(products),
                "sources": {key: {"path": inp[key], "kind": "directory" if Path(inp[key]).is_dir() else "file",
                                  **file_metadata(inp[key])}
                            for key in PATH_FIELDS if key in inp}}
        if request["provenance"] == "sha256":
            info["files"] = inventory([inp[key] for key in PATH_FIELDS if key in inp])
    tools = list(CATALOG[name].get("required_tools", []))
    if name == "runall":
        tools.append("salmon" if params["mode"] == "salmon" else "STAR")
    missing = [tool for tool in tools if not shutil.which(tool)]
    if missing:
        raise HarnessError("Missing external tools on PATH: " + ", ".join(missing), "environment_error", 3)
    info["tools"] = {tool: {"path": shutil.which(tool), **file_metadata(shutil.which(tool), request["provenance"])} for tool in tools}
    return matrix, info


def verify_inputs(request, info):
    from iobrx._run_state import inventory
    if "files" in info:
        current = inventory([request["input"][key] for key in PATH_FIELDS if key in request["input"]])
        if current != info["files"]:
            raise HarnessError("Input/reference files changed during execution", "input_changed", 3)
    elif sha256(request["input"]["path"]) != info["sha256"]:
        raise HarnessError("Input changed during execution", "input_changed", 3)


def dispatch(request, matrix):
    import iobrx
    import pandas as pd

    name, inp = request["analysis"], request["input"]
    params, threads = dict(request["parameters"]), request["threads"]
    output = Path(request["output_dir"]) / "analysis"
    output.mkdir()
    path = inp["path"]
    iobrx.set_threads(threads)
    call = getattr(iobrx, name)
    common = {"n_threads": threads}
    notes = ["Public iobrx API with internal dispatch; native availability alone does not prove Rust executed."]
    if name in {"ips", "lr_cal", "mouse2human"}:
        if name == "lr_cal":
            params.update(data_type="count" if inp["scale"] == "counts" else "tpm", id_type=inp["gene_id"])
        elif name == "mouse2human":
            params["is_matrix"] = True
        result = call(matrix, **params, **common)
    elif name == "bayesprism":
        result = call(matrix, **params, **common)
    elif name == "nmf":
        raw = call(matrix, **params, **common)
        result = {"clusters": raw["clusters"], "top_features": raw["top_features"],
                  "W": pd.DataFrame(raw["W"], index=matrix.index),
                  "H": pd.DataFrame(raw["H"], columns=raw["used_features"].columns),
                  "selection": pd.DataFrame({"best_k": [raw["best_k"]], "silhouette": [raw["best_silhouette"]],
                                             "reconstruction_error": [raw["best_rec_err"]]})}
        for frame in result.values():
            frame.columns = frame.columns.map(str)
    elif name == "tme_cluster":
        result = call(matrix.rename_axis("ID").reset_index(), id="ID", **params, **common)
    elif name in {"log2_eset", "tme_profile"}:
        # Preserve the upstream first CSV parse when input is already canonical.
        if Path(path).suffix != ".csv" or inp["orientation"] != "genes_by_samples":
            path = str(output / "canonical_input.csv")
            matrix.to_csv(path)
            notes.append("Converted matrix to canonical CSV for the file-only public API; this adds a float text round trip.")
        if name == "log2_eset":
            result = call(path, str(output / "transformed.csv"), **params, **common)
        else:
            paths = call(path, str(output / "profile"), **params, **common)
            result = {key: pd.read_csv(value, index_col=0) for key, value in paths.items()}
            notes.append("RNA-seq adapter defaults: QN=False, arrays=False, platform=rnaseq; CIBERSORT uses the original solver.")
    elif name == "prepare_salmon":
        result = call(path, str(output / "prepared.csv"), **params, **common)
    elif name in {"merge_salmon", "merge_star_count"}:
        source = Path(path)
        inputs = source.rglob("quant.sf") if name == "merge_salmon" else source.glob("*ReadsPerGene.out.tab")
        for file in sorted(inputs):
            target = output / file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
        result = call(str(output), **params, **common)
        if name == "merge_salmon":
            result = dict(zip(("tpm", "counts"), result))
    else:
        if name == "fastq_qc":
            result = call(path, str(output), **params, **common)
            if not result.get("multiqc_report"):
                raise HarnessError("MultiQC produced no report", "invalid_result", 3)
        elif name in {"batch_salmon", "batch_star_count"}:
            if "gtf" in inp:
                params["gtf"] = inp["gtf"]
            result = call(inp["index"], path, str(output), **params, **common)
        elif name == "trust4":
            result = call(fqdir=path, f=inp["f"], ref=inp["ref"], o=str(output), **common)
        elif name == "runall":
            # runall requires an empty run directory; its own manifest lives inside analysis/.
            extra = ["--index", inp["index"], "--project", params.pop("project"),
                     "--suffix1", params.pop("suffix1"), "--perm", str(params.pop("perm")),
                     "trust4", "-f", inp["f"], "--ref", inp["ref"]]
            result = call(outdir=str(output), fastq=path, unknown=extra, **params, **common)
        else:
            raise HarnessError(f"No adapter for {name}")
        if isinstance(result, dict) and result.get("rc", 0) != 0:
            raise HarnessError(f"{name} failed with return code {result['rc']}", "tool_failed", 3)
        expected = {"fastq_qc": "*_fastp.json", "batch_salmon": "quant.sf",
                    "batch_star_count": "*ReadsPerGene.out.tab", "trust4": "trust4_immdata.csv",
                    "runall": "deconvo_merged.csv"}[name]
        if not any(file.stat().st_size for file in output.rglob(expected)):
            raise HarnessError(f"{name} produced no expected {expected} artifacts", "invalid_result", 3)
        return None, "external-tools", notes
    return result, "iobrx-api", notes


def artifacts(output, provenance="metadata"):
    """List native outputs; hash contents only for an explicitly requested audit."""
    root = Path(output)
    result = []
    for path in sorted((root / "analysis").rglob("*")):
        if path.is_file():
            resolved = resolve_path(str(path), root, root)
            result.append({"path": str(path.relative_to(root)), "format": "native",
                           **file_metadata(resolved, provenance)})
    return result
