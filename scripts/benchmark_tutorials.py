"""Measure the exact tutorial calls, sequentially, in fresh Python processes."""
from pathlib import Path
import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tutorials"))


def worker(slug, repeats):
    import numpy as np
    import pandas as pd
    import iobrx
    from _common import load, tpm_input, configure
    from build_tutorials import SPECS
    configure(2)
    iobrx.set_threads(8)
    spec = next(s for s in SPECS if s["slug"] == slug)
    scope = dict(np=np, pd=pd, iobrx=iobrx, load=load, tpm_input=tpm_input,
                 perf_counter=perf_counter, display=lambda *args: None)
    code = compile(spec["call"], slug, "exec")
    times = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(spec["data"], scope)
        for _ in range(repeats + 1):
            started = perf_counter()
            exec(code, scope)
            times.append(perf_counter() - started)
    print(json.dumps({"analysis": slug, "input_shape": list(scope["input_shape"]),
                      "first_call_seconds": times[0], "repeat_seconds": times[1:],
                      "median_seconds": statistics.median(times[1:]),
                      "min_seconds": min(times[1:]), "max_seconds": max(times[1:])}))


def main(repeats):
    import iobrx
    from _common import runtime_info
    from build_tutorials import SPECS
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                if line.startswith("model name")), platform.processor())
    iobrx.set_threads(8)
    report = {"method": "One fresh process per analysis; first call reported separately; median of three subsequent calls. Input loading/preparation and plotting excluded. Workflow includes its count-to-TPM stage. OS file caches may be warm. No concurrent benchmark workers.",
              "repeats": repeats, "hardware": {"cpu": cpu, "logical_cpus": os.cpu_count(),
              "avx2": "avx2" in cpuinfo, "avx512f": "avx512f" in cpuinfo},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **runtime_info()}, "analyses": []}
    destination = ROOT / "tutorials/results/benchmark.json"
    for spec in SPECS:
        proc = subprocess.run([sys.executable, __file__, "--worker", spec["slug"],
                               "--repeats", str(repeats)], capture_output=True, text=True, check=True)
        row = json.loads(proc.stdout.splitlines()[-1])
        report["analyses"].append(row)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f'{row["analysis"]}: median {row["median_seconds"]:.4f} s; first {row["first_call_seconds"]:.4f} s', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    worker(args.worker, args.repeats) if args.worker else main(args.repeats)
