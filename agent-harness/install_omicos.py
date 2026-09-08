"""Copy the Agent and self-contained Skill into an explicitly selected local tree."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def install(destination, layout="workspace", dry_run=False):
    source = Path(__file__).resolve().parent / "omicos"
    destination = Path(destination).resolve()
    root = destination / "domains" / "biology" if layout == "catalog" else destination
    pairs = [(source / "agents" / "iobrx_analyst.md", root / "agents" / "iobrx_analyst.md"),
             (source / "skills" / "iobrx", root / "skills" / "iobrx")]
    for _, target in pairs:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing content: {target}")
    if not dry_run:
        for src, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
            else:
                shutil.copyfile(src, target)
    return {"status": "planned" if dry_run else "installed", "layout": layout,
            "paths": [str(dst) for _, dst in pairs], "production_deployed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--layout", choices=["workspace", "catalog"], default="workspace")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(install(args.destination, args.layout, args.dry_run)))
    except OSError as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
