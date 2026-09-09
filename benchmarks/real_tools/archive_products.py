"""Archive exactly the inspected scientific products for a release attachment.

FASTQ gzip files and Salmon/HLA files are retained unchanged. STAR BAMs are
represented by the sorted header-free SAM record bytes actually compared.
fastp JSON is represented by the canonical bytes actually compared (command
string removed). The manifest specifies which digest interpretation applies.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve()
    evidence = json.loads((args.evidence / 'summary.json').read_text())
    manifest = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.output, 'w:gz', compresslevel=1) as archive:
        def add_bytes(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
        for case in evidence['cases']:
            for run in case['runs']:
                root = Path(run['output'].replace('<WORK>', str(work)))
                for name, expected in run['products']['files'].items():
                    prefix = case['evidence_directory'] + '/' + root.name + '/'
                    interpretation = 'raw file bytes'
                    if name.endswith(':sorted-SAM-records'):
                        bam = root / name.split(':')[0]
                        data = b'\n'.join(sorted(subprocess.check_output(['samtools', 'view', str(bam)]).splitlines()))
                        destination = prefix + bam.name + '.sam.records'
                        assert hashlib.sha256(data).hexdigest() == expected
                        add_bytes(destination, data)
                        interpretation = 'sorted SAM records, no header or trailing newline'
                    elif case['case']['function'] == 'fastq_qc' and name.endswith('.json'):
                        data = json.loads((root / name).read_text())
                        data.pop('command', None)
                        data = json.dumps(data, sort_keys=True).encode()
                        assert hashlib.sha256(data).hexdigest() == expected
                        destination = prefix + name
                        add_bytes(destination, data)
                        interpretation = 'canonical JSON, only command removed'
                    else:
                        destination = prefix + name
                        compressed = name.endswith(('.fq.gz', '.fastq.gz'))
                        with (gzip.open(root / name, 'rb') if compressed else (root / name).open('rb')) as stream:
                            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
                        archive.add(root / name, arcname=destination, recursive=False)
                        if compressed:
                            interpretation = 'digest of decompressed FASTQ records'
                    manifest.append({'file': destination, 'comparison_sha256': expected,
                                     'digest_interpretation': interpretation})
        add_bytes('products.json', (json.dumps(manifest, indent=2) + '\n').encode())
        archive.add(args.evidence, arcname='compact-evidence', recursive=True)
    with args.output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.output.with_name(args.output.name + '.sha256').write_text(digest + '  ' + args.output.name + '\n')
    print(f'Archived {len(manifest)} compared products; {args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
