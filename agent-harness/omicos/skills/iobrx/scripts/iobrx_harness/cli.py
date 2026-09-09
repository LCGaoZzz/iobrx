"""CLI stdout is one JSON document; dependency/native messages go to stderr."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import sys
from pathlib import Path

from . import runtime
from .catalog import capabilities, request_schema


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise runtime.HarnessError(message)


@contextlib.contextmanager
def diagnostic_output():
    sys.stdout.flush()
    original = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
            sys.stderr.flush()
            # Some native libraries buffer C stdout independently of Python.
            try:
                ctypes.CDLL(None).fflush(None)
            except (OSError, AttributeError):
                pass
    finally:
        os.dup2(original, 1)
        os.close(original)


def main(argv=None):
    try:
        parser = Parser(description="iobrx Agent harness: JSON stdout, diagnostics on stderr")
        commands = parser.add_subparsers(dest="command", required=True)
        sub = commands.add_parser("capabilities")
        sub.add_argument("--analysis", help="Return only this analysis and its request schema")
        for command in ("doctor", "schema"):
            commands.add_parser(command)
        for command in ("validate", "run"):
            sub = commands.add_parser(command)
            sub.add_argument("--request", required=True, help="JSON file; relative data paths resolve beside it. '-' reads stdin.")
            sub.add_argument("--workspace", type=Path, help="Optional boundary for all request paths; also the base for stdin requests")
        sub = commands.add_parser("status")
        sub.add_argument("path", help="Run directory or results_manifest.json")
        sub.add_argument("--workspace", type=Path)
        sub.add_argument("--verify-hashes", action="store_true", help="Compare artifacts with hashes recorded by provenance=sha256")
        args = parser.parse_args(argv)
        with diagnostic_output():
            if args.command == "capabilities":
                result = capabilities(args.analysis)
            elif args.command == "schema":
                result = request_schema()
            elif args.command == "doctor":
                result = runtime.doctor()
            elif args.command == "status":
                result = runtime.status(args.path, args.workspace or Path.cwd(), args.workspace, args.verify_hashes)
            else:
                if args.request == "-":
                    request = json.load(sys.stdin, parse_constant=lambda x: runtime.reject(f"Non-finite JSON: {x}"))
                    base = args.workspace or Path.cwd()
                else:
                    path = Path(args.request).resolve()
                    request = runtime.read_json(path)
                    base = path.parent
                result = getattr(runtime, args.command)(request, base, args.workspace)
        exit_code = result.get("exit_code", 3 if result.get("status") in {"failed", "interrupted"} else 0)
    except Exception as exc:
        result = runtime.failure(exc)
        exit_code = getattr(exc, "exit_code", 2 if isinstance(exc, (ValueError, OSError)) else 3)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
