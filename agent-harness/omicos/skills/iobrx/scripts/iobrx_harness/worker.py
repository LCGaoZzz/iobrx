"""Absolute-path subprocess launcher for both installed and catalog copies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iobrx_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
