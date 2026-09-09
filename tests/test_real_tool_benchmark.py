"""The benchmark must not mistake incomplete scientific outputs for parity."""
import gzip
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('real_tool_benchmark', Path(__file__).resolve().parents[1] / 'benchmarks/real_tools/run.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_fastq_record_change_is_not_hidden_by_gzip_metadata(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    for folder, timestamp in ((left, 0), (right, 1)):
        folder.mkdir()
        for mate in (1, 2):
            with (folder / f'reads_{mate}.fq.gz').open('wb') as out:
                with gzip.GzipFile(fileobj=out, mode='wb', mtime=timestamp) as gz:
                    gz.write(b'@real-read\nACGT\n+\nIIII\n')
    a = benchmark.scientific_products(left, 'extract_hla_read')
    b = benchmark.scientific_products(right, 'extract_hla_read')
    assert benchmark.comparison(a, b)['equal']
    with gzip.open(right / 'reads_1.fq.gz', 'wb') as gz:
        gz.write(b'@real-read\nACGA\n+\nIIII\n')
    assert not benchmark.comparison(a, benchmark.scientific_products(right, 'extract_hla_read'))['equal']


def test_hla_headers_and_empty_sequences_are_not_success(tmp_path):
    report = tmp_path / 'hla.result.txt'
    genes = ['A', 'B', 'C', 'DPA1', 'DPB1', 'DQA1', 'DQB1', 'DRB1']
    header = '\t'.join(['Sample'] + [f'HLA_{g}_{i}' for g in genes for i in (1, 2)])
    report.write_text('# version: fixture\n' + header + '\n')
    with pytest.raises(ValueError, match='two called alleles'):
        benchmark.scientific_products(tmp_path, 'spechla')
    report.write_text(report.read_text() + '\t'.join(['sample'] + [f'{g}*01:01' for g in genes for _ in (1, 2)]) + '\n')
    for g in genes:
        for i in (1, 2):
            (tmp_path / f'hla.allele.{i}.HLA_{g}.fasta').write_text('>allele\nNNNN\n')
    with pytest.raises(ValueError, match='no called sequence'):
        benchmark.scientific_products(tmp_path, 'spechla')
    for p in tmp_path.glob('*.fasta'):
        p.write_text('>allele\nACGTN\n')
    assert len(benchmark.scientific_products(tmp_path, 'spechla')['files']) == 17
