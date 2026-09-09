"""Offline integrity checks for the committed, executed tutorial collection."""
from pathlib import Path
import base64
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    tutorial_dir = ROOT / "tutorials"
    notebooks = sorted(tutorial_dir.glob("*.ipynb"))
    assert len(notebooks) == 28, f"Expected 28 tutorials, found {len(notebooks)}"
    for path in notebooks:
        text = path.read_text(encoding="utf-8")
        notebook = json.loads(text)
        assert notebook["metadata"]["iobrx"]["figure_revision"] == 2, path
        assert not re.search(r"/root/|/mnt/[a-z]/|C:\\\\Users|G:\\\\", text), path
        images = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is not None, f"Unexecuted cell in {path}"
            for output in cell.get("outputs", []):
                assert output["output_type"] != "error", path
                png = output.get("data", {}).get("image/png")
                if png:
                    assert base64.b64decode("".join(png)).startswith(b"\x89PNG\r\n\x1a\n")
                    images += 1
        assert images >= 1, f"Missing inline figure: {path}"
        for extension in ("png", "pdf", "svg"):
            figure = tutorial_dir / "figures" / f"{path.stem}.{extension}"
            assert figure.is_file() and figure.stat().st_size > 1000, figure
    manifest = json.loads((tutorial_dir / "data/manifest.json").read_text())
    for name, metadata in manifest.items():
        data = tutorial_dir / "data" / f"{name}.parquet"
        assert hashlib.sha256(data.read_bytes()).hexdigest() == metadata["parquet_sha256"], data
    print(f"Validated {len(notebooks)} executed notebooks, {3 * len(notebooks)} figure exports and {len(manifest)} dataset checksums.")


if __name__ == "__main__":
    main()
