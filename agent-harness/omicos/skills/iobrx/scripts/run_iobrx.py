"""Canonical Omicos entrypoint; Python resolves the sibling package here."""
from iobrx_harness import SKILL_ID
from iobrx_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
