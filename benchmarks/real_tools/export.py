"""Publish compact evidence, retaining raw timings and explicitly scoped checks.

Input data and large primary outputs stay in the work directory. The published
logs replace local paths with portable placeholders; scientific values and
their originally observed digests are unchanged.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    def sanitized(text):
        for relative in ('data/gencode.v44.transcripts.fa.gz', 'bin/salmon', 'bin/STAR', 'bin/fastp'):
            path = work / relative
            if path.is_symlink():
                text = text.replace(str(path.resolve()), '<WORK>/' + relative)
        text = text.replace(str(work / 'toolchain'), '<TOOLCHAIN>')
        text = text.replace(str(work / 'conda-cli'), '<CONDA_CLI>')
        text = text.replace(str(work), '<WORK>')
        text = text.replace(str(Path(__file__).resolve().parents[2]), '<REPO>')
        return text
    def copy(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sanitized(source.read_text(errors='replace')))
    cases = []
    for run_dir in args.run_dir:
        for summary in sorted(run_dir.glob('*/summary.json')):
            case = json.loads(sanitized(summary.read_text()))
            label = run_dir.name + '/' + summary.parent.name
            case['evidence_directory'] = label
            cases.append(case)
            for path in summary.parent.glob('*'):
                if path.is_file() and path.suffix in {'.json', '.log', '.txt'}:
                    copy(path, Path(label) / path.name)
            for path in summary.parent.glob('repeat-*/*'):
                if path.is_file() and (path.name.endswith('Log.final.out') or path.name == 'hla_result_merged.txt'):
                    copy(path, Path(label) / path.relative_to(summary.parent))
            for path in summary.parent.glob('repeat-*/**/hla.result.txt'):
                copy(path, Path(label) / path.relative_to(summary.parent))
    cpu = Path('/proc/cpuinfo').read_text()
    cpu_model = next(line.split(':', 1)[1].strip() for line in cpu.splitlines() if line.startswith('model name'))
    import iobrx
    metadata = {'date': '2026-09-09', 'cpu': cpu_model, 'avx512_present': 'avx512' in cpu,
                'packages': {name: importlib.metadata.version(name) for name in ('iobrx', 'iobrpy', 'numpy', 'pandas', 'scipy', 'pysam', 'multiqc')},
                'backend': iobrx.backend_info(), 'package_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                'scope': 'Real bounded public read fixtures; wrapper/output comparison, not independent HLA genotype accuracy or a full-size human cohort performance claim.'}
    metadata['benchmark_scripts_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in Path(__file__).parent.glob('*.py')}
    for path in sorted((work / 'dist').glob('*.whl')):
        with path.open('rb') as stream:
            metadata['tested_wheel'] = {'file': path.name, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    (output / 'summary.json').write_text(json.dumps({'environment': metadata, 'cases': cases}, indent=2) + '\n')
    for name in ('sources.json', 'downloads.json', 'tool-versions.json', 'prepare-hla-bam.json',
                 'bwa-index.time', 'salmon-index.time', 'hla-preflight.log'):
        path = work / name
        if path.exists():
            copy(path, Path('preparation') / name)
    print(f'Exported {len(cases)} cases')


if __name__ == '__main__':
    main()
