"""Check source/archive consistency and the installed release, using stdlib."""
from __future__ import annotations

import argparse
import email
import importlib.metadata
import pathlib
import tarfile
import tomllib
import zipfile


def check_source(root: pathlib.Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    cargo = tomllib.loads((root / "rust/Cargo.toml").read_text())["package"]
    assert project["version"] == cargo["version"], "Python/Rust versions differ"
    return project["version"]


def check_archives(directory: pathlib.Path) -> None:
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, (wheels, sources)
    wheel = wheels[0]
    assert "cp311-cp311" in wheel.name and "manylinux" in wheel.name, wheel.name
    assert "manylinux_2_17_x86_64" in wheel.name or "manylinux2014_x86_64" in wheel.name, wheel.name
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert any(n.startswith("iobrx/_rust") and n.endswith(".so") for n in names)
        assert "iobrx/__init__.py" in names
        metadata = email.message_from_bytes(archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))))
        assert metadata["Name"] == "iobrx"
        assert metadata["Version"] == check_source(pathlib.Path(__file__).resolve().parents[1])
        assert not any("__pycache__" in n for n in names)
    with tarfile.open(sources[0]) as archive:
        names = {n.split("/", 1)[1] for n in archive.getnames() if "/" in n}
        required = {"pyproject.toml", "rust/Cargo.toml", "rust/Cargo.lock", "rust/build.rs", "rust/wrapper.cpp", "rust/vendor/svm.cpp", "rust/src/lib.rs", "src/iobrx/__init__.py"}
        assert required <= names, required - names
    print(f"Verified {wheel.name} and {sources[0].name}")


def check_installed() -> None:
    import iobrx
    import iobrx._rust

    path = pathlib.Path(iobrx.__file__).resolve()
    assert "site-packages" in path.parts, f"Imported source checkout instead of wheel: {path}"
    assert iobrx.__version__ == importlib.metadata.version("iobrx")
    assert iobrx.backend_info()["native_available"], iobrx.backend_info()
    print(f"Installed iobrx {iobrx.__version__}: {path}")
    print(iobrx.backend_info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--dist", type=pathlib.Path)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    if args.source:
        print(check_source(pathlib.Path(__file__).resolve().parents[1]))
    if args.dist:
        check_archives(args.dist)
    if args.installed:
        check_installed()
