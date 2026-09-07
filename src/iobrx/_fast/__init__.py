"""Fast-path implementations (package-internal).

These modules are drop-in accelerated replacements for the corresponding
``iobrpy.workflow`` stages. They import the ORIGINAL ``iobrpy`` package for
resource pickles and fallback code paths; no numeric logic differs from the
frozen, gate-validated versions (see BENCHMARKS.md and the repository's
provenance notes).

The stable, documented entry points live in :mod:`iobrx` itself; treat
everything under ``iobrx._fast`` as internal API.
"""
