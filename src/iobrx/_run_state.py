"""Content-checked execution state, separate from scientific output tables."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def walk_paths(root):
    """Follow linked reference directories without looping through cycles."""
    seen = set()
    for current, directories, files in os.walk(root, followlinks=True):
        resolved = Path(current).resolve(strict=True)
        if resolved in seen:
            directories.clear()
            continue
        seen.add(resolved)
        directories.sort()
        for name in sorted(directories + files):
            yield Path(current) / name


def inventory(paths):
    """Fingerprint every file in explicit inputs/references, in stable order."""
    records = []
    for value in paths:
        root = Path(value).resolve(strict=True)
        files = sorted(walk_paths(root)) if root.is_dir() else [root]
        records.append({"root": str(root), "files": [
            {"path": str(p.relative_to(root)) if root.is_dir() else p.name,
             "size": p.stat().st_size, "sha256": file_hash(p)}
            for p in files if p.is_file()]})
    return records


def signature(inputs, parameters, tools=()):
    binaries = []
    for tool in tools:
        path = shutil.which(str(tool))
        if path is None:
            raise FileNotFoundError(f"Required executable not found: {tool}")
        binaries.append(path)
    return {"schema_version": 1, "inputs": inventory(inputs),
            "parameters": parameters, "tools": inventory(binaries)}


def artifact_records(paths):
    records = []
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected nonempty output was not produced: {path}")
        records.append({"path": str(path), "sha256": file_hash(path)})
    return records


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def state_path(marker):
    return Path(str(marker) + ".iobrx.json")


def completed(marker, expected_signature, outputs):
    if not Path(marker).is_file():
        return False
    try:
        state = json.loads(state_path(marker).read_text(encoding="utf-8"))
        return (state["signature"] == expected_signature
                and state["outputs"] == artifact_records(outputs))
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def complete(marker, expected_signature, outputs, text="done\n"):
    # Verify products before creating either completion record.
    records = artifact_records(outputs)
    write_json(state_path(marker), {"signature": expected_signature, "outputs": records})
    Path(marker).write_text(text, encoding="utf-8")
