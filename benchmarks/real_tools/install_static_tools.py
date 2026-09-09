"""Fetch the same official Linux x86-64 binaries used in this benchmark.

Hashes identify the measured builds. They are benchmark provenance, not an
iobrx installation policy. fastp's unversioned URL can change; on a mismatch
retain the downloaded file and use the recorded build or report a new version.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

TOOLS = {
    "fastp": ("https://opengene.org/fastp/fastp", None,
              "817df51647ecdacc1642daabe2438c744828ccda2d3634d90395ea91e3a7bc1f"),
    "STAR": ("https://github.com/alexdobin/STAR/releases/download/2.7.11b/STAR_2.7.11b.zip", "Linux_x86_64_static/STAR",
             "36e94b899a56b0ea5de5d65e722f55bc552b6713bd2d1e83a5c2abbd06c9881a"),
    "salmon": ("https://github.com/COMBINE-lab/salmon/releases/download/v1.10.0/salmon-1.10.0_linux_x86_64.tar.gz", "bin/salmon",
               "7a8e79acf686cc2c9a3dd0b260ef0a00157db1ed24da441d7e9b8a729fdbd54f"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    downloads = args.workdir / 'downloads'
    binaries = args.workdir / 'bin'
    downloads.mkdir(parents=True, exist_ok=True)
    binaries.mkdir(parents=True, exist_ok=True)
    for name, (url, member_suffix, expected) in TOOLS.items():
        target = binaries / name
        if not target.exists():
            archive = downloads / url.rsplit('/', 1)[-1]
            if not archive.exists():
                with urllib.request.urlopen(url, timeout=120) as source, archive.open('wb') as out:
                    shutil.copyfileobj(source, out)
            if member_suffix is None:
                shutil.copy2(archive, target)
            elif url.endswith('.zip'):
                with zipfile.ZipFile(archive) as handle:
                    members = [m for m in handle.namelist() if m.endswith(member_suffix)]
                    if len(members) != 1:
                        raise ValueError(f'Ambiguous binary in {archive}: {members}')
                    with handle.open(members[0]) as source, target.open('wb') as out:
                        shutil.copyfileobj(source, out)
            else:
                with tarfile.open(archive) as handle:
                    members = [m for m in handle.getmembers() if m.isfile() and m.name.endswith(member_suffix)]
                    if len(members) != 1:
                        raise ValueError(f'Ambiguous binary in {archive}: {members}')
                    with handle.extractfile(members[0]) as source, target.open('wb') as out:
                        shutil.copyfileobj(source, out)
        with target.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'{name}: downloaded build {actual} differs from the measured build {expected}')
        target.chmod(target.stat().st_mode | 0o111)
        print(f'{name}: measured build available')


if __name__ == '__main__':
    main()
