"""Render README timing tables and the tutorial gallery from measured JSON."""
from pathlib import Path
import json
from build_tutorials import SPECS

ROOT = Path(__file__).resolve().parents[1]


def duration(seconds):
    return f"{seconds * 1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


def main():
    report = json.loads((ROOT / "tutorials/results/benchmark.json").read_text())
    results = {row["analysis"]: row for row in report["analyses"]}
    assert set(results) == {s["slug"] for s in SPECS}, "Run the complete benchmark first"
    for filename, zh in [("README.md", False), ("README.zh-CN.md", True)]:
        heading = "| 分析 / 教程 | 输入：特征 × 样本 | 中位耗时 | 首次调用 |" if zh else "| Analysis / notebook | Input: features × samples | Median | First call |"
        table = [heading, "| --- | --- | ---: | ---: |"]
        for spec in SPECS:
            row = results[spec["slug"]]
            title = spec["zh"] if zh else spec["title"]
            shape = " × ".join(f"{n:,}" for n in row["input_shape"])
            table.append(f'| [{title}](tutorials/{spec["slug"]}.ipynb) | {shape} | **{duration(row["median_seconds"])}** | {duration(row["first_call_seconds"])} |')
        path = ROOT / filename
        before, rest = path.read_text(encoding="utf-8").split("<!-- BENCHMARK_TABLE_START -->")
        _, after = rest.split("<!-- BENCHMARK_TABLE_END -->")
        path.write_text(before + "<!-- BENCHMARK_TABLE_START -->\n" + "\n".join(table) + "\n<!-- BENCHMARK_TABLE_END -->" + after, encoding="utf-8")
    gallery = ["# Executed tutorials / 已执行教程", "",
               "Eleven analysis tutorials plus one complete workflow. Every notebook includes actual outputs, a figure, interpretation notes and a sample-ID mapping. 每本均已在 WSL Omicos 环境执行，并附结果和图。", "",
               "Run from a clone of the repository after installing `.[tutorials]`. The data is local in `data/`; plot helpers live in `_common.py`. Open a notebook on GitHub to read it, or use `python -m jupyterlab tutorials` to run it.", "",
               "[Measurement details](BENCHMARKS.md) · [Two-round figure review](FIGURE_REVIEW.md) · [Data provenance](data/README.md)", "",
               "| Notebook | Question | Median analysis time | Figure exports |", "| --- | --- | ---: | --- |"]
    for spec in SPECS:
        slug = spec["slug"]
        links = " / ".join(f"[{ext.upper()}](figures/{slug}.{ext})" for ext in ["png", "pdf", "svg"])
        gallery.append(f'| [{spec["zh"]}]({slug}.ipynb) | {spec["title"]} | {duration(results[slug]["median_seconds"])} | {links} |')
    gallery += ["", "## Figure gallery", "", "White backgrounds, muted colors, thin axes and editable vector typography. Heatmap scaling is display-only; no outcomes or sample groups are invented.", ""]
    for spec in SPECS:
        gallery += [f'### {spec["zh"]}', "", f'[Open notebook]({spec["slug"]}.ipynb)', "", f'![{spec["title"]}](figures/{spec["slug"]}.png)', ""]
    (ROOT / "tutorials/README.md").write_text("\n".join(gallery), encoding="utf-8")
    print("Updated both READMEs and the 12-notebook gallery.")


if __name__ == "__main__":
    main()
