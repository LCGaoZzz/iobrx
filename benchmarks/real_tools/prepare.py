"""Prepare public, bounded real-read fixtures and a portable benchmark config.

Requires fastp, MultiQC, Salmon, STAR, BWA, samtools and the prepared HLA tools
on PATH. Downloads and index construction are recorded separately from timings.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

SPEC_REV = "c965423bb263eb4ead599d7cca8d103e616cf1c5"
GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.transcripts.fa.gz"
YEAST = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/"


def download(url, path, pairs=None):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=90) as response, part.open("wb") as output:
        if pairs is None:
            shutil.copyfileobj(response, output)
        else:
            with gzip.GzipFile(fileobj=response) as source, gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as target:
                lines = 0
                for line in itertools.islice(source, pairs * 4):
                    target.write(line)
                    lines += 1
                if lines != pairs * 4:
                    raise ValueError(f"Source has fewer than {pairs} reads: {url}")
    part.rename(path)


def run(command, log, records):
    start = time.perf_counter()
    with log.open("w") as handle:
        subprocess.run([str(x) for x in command], stdout=handle, stderr=subprocess.STDOUT, check=True)
    records.append({"command": [str(x) for x in command], "seconds": time.perf_counter() - start})


def write_config(work, threads=4):
    data = work / "data"
    def batch(name, inputs, index=None):
        params = {"path_fq": str(inputs), "path_out": "{output}", "num_threads": threads, "batch_size": 1}
        if name == "fastq_qc":
            params = {"path1_fastq": str(inputs), "path2_fastp": "{output}", "num_threads": threads, "batch_size": 1}
        if index is not None:
            params["index"] = str(index)
        cli = [name]
        for key, value in params.items():
            cli += ["--" + key, str(value)]
        return {"name": name, "function": name, "parameters": params, "upstream_cli": cli}
    cases = [batch("fastq_qc", data / "geuvadis"),
             batch("batch_salmon", data / "geuvadis", data / "gencode-v44-salmon"),
             batch("batch_star_count", data / "yeast/reads", data / "yeast/index")]
    bam = str(data / "NA06985.GRCh38.chr6.bam")
    cases += [{"name": "extract_hla_read", "function": "extract_hla_read",
               "parameters": {"sample_id": "NA06985", "bam_path": bam, "ref": "hg38", "outdir": "{output}", "auto_install": False},
               "upstream_cli": ["extract_hla_read", "-s", "NA06985", "-b", bam, "-r", "hg38", "-o", "{output}", "--no-auto-install"]},
              {"name": "spechla", "function": "spechla",
               "parameters": {"name": "NA06985", "read1": str(data / "NA06985_1.filter.fastq.gz"),
                              "read2": str(data / "NA06985_2.filter.fastq.gz"), "outdir": "{output}", "threads": threads, "use_exon": 1},
               "upstream_cli": ["spechla", "-n", "NA06985", "-1", str(data / "NA06985_1.filter.fastq.gz"),
                                "-2", str(data / "NA06985_2.filter.fastq.gz"), "-o", "{output}", "-j", str(threads), "-u", "1"]}]
    config = {"threads": threads, "cases": cases}
    (work / "cases.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--config-only", action="store_true", help="Use already prepared data with the documented layout")
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    data = work / "data"
    if args.config_only:
        write_config(work, args.threads)
        return
    sources, preparation = [], []
    def fetch(url, relative, reads=None):
        path = data / relative
        start = time.perf_counter()
        download(url, path, reads)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        sources.append({"file": str(path.relative_to(work)), "url": url, "read_prefix": reads,
                        "bytes": path.stat().st_size, "sha256": checksum,
                        "download_or_reuse_seconds": time.perf_counter() - start})
        if path.name.endswith(('.fastq.gz', '.fq.gz')):
            with gzip.open(path, 'rb') as stream:
                sources[-1]['decompressed_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return path
    for accession in ("ERR188044", "ERR188104"):
        for mate in (1, 2):
            filename = f"{accession}_{mate}.fastq.gz"
            fetch(f"https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR188/{accession}/{filename}", "geuvadis/" + filename, 100000)
    for sample, accession in (("sampleA", "DRR392086"), ("sampleB", "DRR392094")):
        for mate in (1, 2):
            fetch(f"https://ftp.sra.ebi.ac.uk/vol1/fastq/DRR392/{accession}/{accession}_{mate}.fastq.gz",
                  f"yeast/reads/{sample}_{mate}.fastq.gz", 30000)
    transcripts = fetch(GENCODE, "gencode.v44.transcripts.fa.gz")
    for name, suffix in (("genome.fa", "fna"), ("genes.gtf", "gtf")):
        archive = fetch(YEAST + f"GCF_000146045.2_R64_genomic.{suffix}.gz", "yeast/" + name + ".gz")
        target = data / "yeast" / name
        if not target.exists():
            with gzip.open(archive, "rb") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    for mate in (1, 2):
        fetch(f"https://raw.githubusercontent.com/deepomicslab/SpecHLA/{SPEC_REV}/example/exon/NA06985_{mate}.filter.fastq.gz",
              f"NA06985_{mate}.filter.fastq.gz")
    chr6 = fetch("https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.6.fa.gz", "GRCh38.chr6.fa.gz")
    genome = data / "GRCh38.chr6.fa"
    if not genome.exists():
        with gzip.open(chr6, "rb") as source, genome.open("wb") as output:
            shutil.copyfileobj(source, output)
    if not (data / "gencode-v44-salmon/versionInfo.json").exists():
        run(["salmon", "index", "-t", transcripts, "-i", data / "gencode-v44-salmon", "-p", args.threads, "-k", 31], work / "salmon-index.log", preparation)
    if not (data / "yeast/index/SA").exists():
        (data / "yeast/index").mkdir(exist_ok=True)
        run(["STAR", "--runMode", "genomeGenerate", "--genomeDir", data / "yeast/index", "--genomeFastaFiles", data / "yeast/genome.fa",
             "--sjdbGTFfile", data / "yeast/genes.gtf", "--sjdbOverhang", 75, "--genomeSAindexNbases", 10,
             "--runThreadN", args.threads], work / "star-index.log", preparation)
    if not genome.with_suffix(genome.suffix + ".bwt").exists():
        run(["bwa", "index", genome], work / "bwa-index.log", preparation)
    bam = data / "NA06985.GRCh38.chr6.bam"
    if not bam.exists():
        sam = data / "NA06985.GRCh38.chr6.sam"
        with sam.open("wb") as output, (work / "hla-alignment.log").open("w") as log:
            subprocess.run(["bwa", "mem", "-t", str(args.threads), str(genome), str(data / "NA06985_1.filter.fastq.gz"),
                            str(data / "NA06985_2.filter.fastq.gz")], stdout=output, stderr=log, check=True)
        run(["samtools", "sort", "-@", args.threads, "-o", bam, sam], work / "hla-sort.log", preparation)
    if not bam.with_suffix(".bam.bai").exists():
        run(["samtools", "index", bam], work / "hla-index.log", preparation)
    run(["samtools", "quickcheck", bam], work / "hla-quickcheck.log", preparation)
    (work / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (work / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    write_config(work, args.threads)
    print(f"Prepared {len(sources)} public input files; config: {work / 'cases.json'}")


if __name__ == "__main__":
    main()
