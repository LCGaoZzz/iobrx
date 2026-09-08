"""Execute every tutorial in a fresh kernel using the current interpreter."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
from time import perf_counter

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def execute(pattern="*.ipynb"):
    reports = []
    with tempfile.TemporaryDirectory(prefix="iobrx-kernel-") as directory:
        kernel = Path(directory) / "kernels" / "iobrx-execution"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "iobrx execution", "language": "python",
        }), encoding="utf-8")
        old_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = directory + (os.pathsep + old_path if old_path else "")
        try:
            for path in sorted((ROOT / "tutorials").glob(pattern)):
                notebook = nbformat.read(path, as_version=4)
                start = perf_counter()
                client = NotebookClient(notebook, timeout=600, kernel_name="iobrx-execution",
                                        record_timing=False,
                                        resources={"metadata": {"path": str(path.parent)}})
                try:
                    client.execute()
                finally:
                    nbformat.write(notebook, path)
                errors = [out for cell in notebook.cells for out in cell.get("outputs", [])
                          if out.output_type == "error"]
                images = sum("image/png" in out.get("data", {}) for cell in notebook.cells
                             for out in cell.get("outputs", []))
                if errors or not images:
                    raise RuntimeError(f"{path.name}: errors={len(errors)}, embedded images={images}")
                report = {"notebook": path.name, "execution_seconds": perf_counter()-start,
                          "embedded_figures": images, "errors": len(errors)}
                reports.append(report)
                print(json.dumps(report), flush=True)
        finally:
            if old_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old_path
    destination = ROOT / "tutorials" / "results" / "notebook_execution.json"
    destination.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*.ipynb")
    execute(parser.parse_args().pattern)
