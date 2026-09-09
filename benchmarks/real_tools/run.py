"""Compare actual external-tool outputs and fresh-process wall times.

Preparation/indexing are excluded. Both arms use the same input paths and
active output path; results are archived after each invocation. No scientific
file is edited to obtain agreement. Raw logs and differing files are retained.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def digest(path, compressed=False):
    with (gzip.open(path, "rb") if compressed else Path(path).open("rb")) as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fastq_records(path):
    with gzip.open(path, "rb") as handle:
        lines = sum(1 for _ in handle)
    if lines % 4:
        raise ValueError(f"Incomplete FASTQ: {path}")
    return lines // 4


def scientific_products(root, kind):
    files, metrics = {}, {}
    if kind in {"fastq_qc", "extract_hla_read"}:
        reads = sorted(p for p in root.rglob("*.gz")
                       if p.name.endswith((".fq.gz", ".fastq.gz")))
        if len(reads) < 2:
            raise ValueError(f"No paired FASTQ output in {root}")
        for p in reads:
            key = str(p.relative_to(root))
            files[key] = digest(p, compressed=True)
            metrics[key] = {"reads": fastq_records(p)}
        if not any(v["reads"] for v in metrics.values()):
            raise ValueError("Extraction produced no reads")
        if kind == "fastq_qc":
            for p in sorted(root.glob("*_fastp.json")):
                data = json.loads(p.read_text())
                data.pop("command", None)  # Contains paths, not measurements.
                files[p.name] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
                metrics[p.name] = data.get("summary", {})
            if not (root / "multiqc_report/multiqc_fastp_report.html").is_file():
                raise ValueError("MultiQC report is missing")
    elif kind == "batch_salmon":
        import pandas as pd
        for p in sorted(root.rglob("quant.sf")):
            table = pd.read_csv(p, sep="\t")
            if table.empty or not (table.NumReads > 0).any():
                raise ValueError(f"Empty quantification: {p}")
            key = str(p.relative_to(root))
            files[key] = digest(p)
            metrics[key] = {"transcripts": len(table), "positive_transcripts": int((table.NumReads > 0).sum()),
                            "assigned_fragments": float(table.NumReads.sum()), "tpm_sum": float(table.TPM.sum())}
        if not files:
            raise ValueError("No Salmon quantification files")
    elif kind == "batch_star_count":
        for p in sorted(root.glob("*ReadsPerGene.out.tab")):
            files[p.name] = digest(p)
        for p in sorted(root.glob("*.bam")):
            subprocess.run(["samtools", "quickcheck", "-v", str(p)], check=True)
            sam = subprocess.check_output(["samtools", "view", str(p)])
            records = sorted(sam.splitlines())
            files[p.name + ":sorted-SAM-records"] = hashlib.sha256(b"\n".join(records)).hexdigest()
            metrics[p.name] = {"alignment_records": len(records),
                               "mapped_records": sum(not (int(x.split(b"\t")[1]) & 4) for x in records)}
        if not files or not metrics or not all(v["mapped_records"] for v in metrics.values()):
            raise ValueError("STAR has no usable counts/alignments")
    elif kind in {"spechla", "hla_typing"}:
        reports = sorted(root.rglob("hla.result.txt"))
        if not reports:
            raise ValueError("SpecHLA allele report is missing")
        for p in reports:
            lines = [line for line in p.read_text().splitlines() if line.strip()]
            rows = [line.split("\t") for line in lines if not line.startswith("#")]
            if len(rows) < 2 or len(rows[0]) != 17 or any(
                len(row) != 17 or any("*" not in allele for allele in row[1:])
                for row in rows[1:]
            ):
                raise ValueError("SpecHLA must report two called alleles at each of eight loci")
            files[str(p.relative_to(root))] = digest(p)
            metrics[str(p.relative_to(root))] = {"lines": lines}
        alleles = sorted(root.rglob("hla.allele.*.fasta"))
        if len(alleles) != 16 * len(reports):
            raise ValueError("SpecHLA must produce 16 allele FASTAs per sample")
        for p in alleles:
            sequence = ''.join(line.strip() for line in p.read_text().splitlines() if not line.startswith('>')).upper()
            if not sequence or not set(sequence) & set('ACGT'):
                raise ValueError(f"SpecHLA allele has no called sequence: {p}")
            files[str(p.relative_to(root))] = digest(p)
    else:
        raise ValueError(f"Unknown comparison kind: {kind}")
    return {"files": files, "metrics": metrics}


def comparison(left, right):
    keys = sorted(set(left["files"]) | set(right["files"]))
    differences = [k for k in keys if left["files"].get(k) != right["files"].get(k)]
    return {"equal": not differences, "compared_products": len(keys), "different_products": differences}


def expand(value, out):
    if isinstance(value, str):
        return value.replace("{output}", str(out))
    if isinstance(value, dict):
        return {k: expand(v, out) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, out) for v in value]
    return value


def run_case(case, args, env):
    folder = args.output / case["name"]
    folder.mkdir(parents=True, exist_ok=False)
    active = folder / "active"
    records = []
    for rep in range(args.repeats):
        order = ("iobrpy", "iobrx") if rep % 2 == 0 else ("iobrx", "iobrpy")
        for arm in order:
            label = f"repeat-{rep + 1}-{arm}"
            if active.exists():
                raise FileExistsError(active)
            active.mkdir()
            request = {"function": case["function"], "parameters": expand(case["parameters"], active)}
            request_path = folder / f"{label}.request.json"
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            command = ([args.python, "-m", "iobrpy.main", *expand(case["upstream_cli"], active)]
                       if arm == "iobrpy" else [args.python, str(Path(__file__).with_name("arm.py")), str(request_path)])
            resource_path = folder / f"{label}.resources.txt"
            timed = ["/usr/bin/time", "-f", "%e\t%M", "-o", str(resource_path), *command]
            start = time.perf_counter()
            with (folder / f"{label}.log").open("w") as log:
                process = subprocess.run(timed, stdout=log, stderr=subprocess.STDOUT, env=env)
            wall = time.perf_counter() - start
            archive = folder / label
            active.rename(archive)
            record = {"arm": arm, "repeat": rep + 1, "wall_seconds": wall, "exit_code": process.returncode,
                      "command": command, "output": str(archive), "products": None}
            try:
                record["peak_rss_kib"] = int(resource_path.read_text().splitlines()[-1].split("\t")[1])
            except (IndexError, ValueError):
                record["peak_rss_kib"] = None
            if process.returncode == 0:
                try:
                    record["products"] = scientific_products(archive, case["function"])
                except Exception as exc:
                    record["product_error"] = str(exc)
            records.append(record)
            (folder / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
            print(f'{case["name"]} {label}: rc={process.returncode}, {wall:.3f}s, products={record["products"] is not None}', flush=True)
            if process.returncode or record["products"] is None:
                raise RuntimeError(f"Failed run {label}; preserved logs and outputs under {folder}")
    pairs = []
    for rep in range(1, args.repeats + 1):
        pair = {r["arm"]: r["products"] for r in records if r["repeat"] == rep}
        pairs.append({"repeat": rep, **comparison(pair["iobrpy"], pair["iobrx"])})
    repeatability = {}
    for arm in ("iobrpy", "iobrx"):
        products = [r["products"] for r in records if r["arm"] == arm]
        repeatability[arm] = [comparison(products[0], p) for p in products[1:]]
    medians = {arm: statistics.median(r["wall_seconds"] for r in records if r["arm"] == arm)
               for arm in ("iobrpy", "iobrx")}
    result = {"case": case, "repeats": args.repeats, "runs": records, "paired_comparisons": pairs,
              "within_arm_repeatability": repeatability, "median_wall_seconds": medians,
              "speedup_original_over_iobrx": medians["iobrpy"] / medians["iobrx"]}
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", action="append", help="Select named cases; default all")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(config["threads"]), "OPENBLAS_NUM_THREADS": str(config["threads"]),
                "MKL_NUM_THREADS": str(config["threads"]), "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg"})
    metadata = {"platform": platform.platform(), "python": sys.version, "threads": config["threads"],
                "versions": {k: importlib.metadata.version(k) for k in ("iobrx", "iobrpy", "numpy", "scipy", "pandas")},
                "protocol": "Fresh CLI versus fresh Python API process; identical paths/tools/parameters; alternating arm order; no cache flush; preparation, indexing and comparison excluded; setup completed before timing."}
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = [c for c in config["cases"] if not args.case or c["name"] in args.case]
    if not cases:
        parser.error("no matching cases")
    results = []
    for case in cases:
        results.append(run_case(case, args, env))
    (args.output / "summary.json").write_text(json.dumps({"environment": metadata, "cases": results}, indent=2) + "\n")
    # A mismatch is reported, not hidden by a blanket tolerance or a failed-file exclusion.
    return int(any(not p["equal"] for result in results for p in result["paired_comparisons"]))


if __name__ == "__main__":
    raise SystemExit(main())
