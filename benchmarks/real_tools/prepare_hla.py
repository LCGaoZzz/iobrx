"""Prepare upstream SpecHLA dependencies outside the timed comparison.

Run only in the dedicated tool environment: upstream helpers may install
dependencies and compile its bundled SpecHap / ExtractHAIRs sources.
"""


if __name__ == '__main__':
    # A complete untimed example prepares dependencies and checks the actual
    # workflow before either benchmark arm starts.
    import argparse
    from pathlib import Path
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    work = args.workdir.resolve()
    output = work / 'hla-preflight'
    if output.exists():
        parser.error('Use a fresh preflight output directory; it already exists')
    subprocess.run([sys.executable, '-m', 'iobrpy.main', 'spechla', '-n', 'NA06985',
                    '-1', str(work / 'data/NA06985_1.filter.fastq.gz'),
                    '-2', str(work / 'data/NA06985_2.filter.fastq.gz'),
                    '-o', str(output), '-j', str(args.threads), '-u', '1'], check=True)
