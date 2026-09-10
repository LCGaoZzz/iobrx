"""Reference-data resolver for the accelerated workflows.

The reference data the native and vectorized kernels read is shipped inside
the wheel (``iobrx/_resources``), copied unmodified from the MIT-licensed
IOBRpy 0.2.1 distribution (see ``iobrx/_resources/NOTICE.md``). When a file
is not bundled, fall back to an installed IOBRpy — the optional ``python``
extra — so parity environments and the Python fallback backend keep working.

Every lookup that fails raises a FileNotFoundError that names the exact
install command, instead of a bare import error.
"""
from __future__ import annotations

from importlib.resources import files

__all__ = ["BUNDLED", "bundled_names", "resource_path"]

# Files shipped in iobrx/_resources/. Names are relative to that directory;
# "resources/<x>" is accepted and normalized so code that used to address
# ``iobrpy/resources/<x>`` through the package root keeps working.
BUNDLED = (
    "IPS_genes.txt",
    "anno_eset.pkl",
    "calculate_data.pkl",
    "common_genes.txt",
    "count2tpm_data.pkl",
    "epic_TRef_BRef.pkl",
    "estimate_data.pkl",
    "lm22.txt",
    "lr_data.pkl",
    "mcp_data.pkl",
    "mus_human.pkl",
    "quantiseq_data.pkl",
    "hg38_bcrtcr.fa",
    "human_IMGT+C.fa",
    "BP_data/cell_state_labels.csv",
    "BP_data/cell_type_labels.csv",
    "BP_data/sc_dat.csv",
    "txt/gencode.v22.broad.category.txt",
    "txt/genelist.hs.new.txt",
    "txt/genelist.mm.new.txt",
)


def bundled_names() -> tuple[str, ...]:
    """Names of the reference files bundled with this installation."""
    return tuple(BUNDLED)


def resource_path(name: str) -> str:
    """Return a filesystem path for a bundled reference file.

    Resolution order: bundled ``iobrx/_resources``, then an installed
    ``iobrpy.resources`` / ``iobrpy.bayesprism`` package (the ``python``
    extra). pip installs wheels as directories, so the returned string is a
    real path ``open()`` and ``pd.read_pickle`` can use directly.
    """
    if name.startswith("resources/"):
        name = name[len("resources/"):]
    candidate = files("iobrx").joinpath("_resources", name)
    if candidate.is_file():
        return str(candidate)
    for package in ("iobrpy.resources", "iobrpy.bayesprism"):
        try:
            alternative = files(package).joinpath(name)
        except ModuleNotFoundError:
            continue
        if alternative.is_file():
            return str(alternative)
    raise FileNotFoundError(
        f"Reference data '{name}' is not available in this iobrx installation. "
        "Reinstall iobrx, or add the Python fallback backend with "
        "`pip install 'iobrx[python]'`."
    )
