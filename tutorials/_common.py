"""Shared presentation helpers; all analysis calls remain visible in notebooks."""
from pathlib import Path
from time import perf_counter
import importlib.metadata
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tutorials" / "data"
FIGURES = ROOT / "tutorials" / "figures"
TABLES = ROOT / "tutorials" / "results"
COLORS = ["#267C7C", "#C97856", "#334C68", "#9FACA0", "#D5B777",
          "#92748E", "#73A8BA", "#B6B8BB", "#D3A39B", "#687654", "#A9BFD0"]
TEAL, RUST, NAVY = COLORS[:3]
CMAP = LinearSegmentedColormap.from_list("iobrx_balance", ["#315E79", "#FAF9F6", "#B96549"])
FRACTION_CMAP = LinearSegmentedColormap.from_list("iobrx_fraction", ["#FAFBFB", "#AACACA", "#477F90", "#233E58"])
REVISION = 2


def configure(revision=2):
    global REVISION
    REVISION = revision
    plt.rcdefaults()
    if revision >= 1:
        mpl.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 8,
            "axes.titlesize": 9, "axes.labelsize": 8,
            "xtick.labelsize": 7, "ytick.labelsize": 7,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 0.65, "xtick.major.width": 0.65,
            "ytick.major.width": 0.65, "xtick.major.size": 3,
            "ytick.major.size": 3, "text.color": "#252A2D",
            "axes.labelcolor": "#252A2D", "axes.edgecolor": "#40474B",
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "pdf.fonttype": 42,
            "ps.fonttype": 42, "svg.fonttype": "none",
            "legend.frameon": False, "legend.fontsize": 7,
            "axes.titlepad": 9, "axes.labelpad": 5,
        })
    warnings.formatwarning = lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def load(name):
    """Read the unchanged public fixture; fall back to its upstream release."""
    path = DATA / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import iobrx
    return iobrx.load_official(name, verbose=False)


def tpm_input():
    import iobrx
    return iobrx.count2tpm(load("eset_stad"), check_data=True, remove_version=True)


def runtime_info():
    import iobrx
    versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "iobrpy", "gseapy", "matplotlib"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source checkout"
    return {"iobrx": iobrx.__version__, "threads": iobrx.get_threads(),
            **iobrx.backend_info(), "versions": versions}


def numeric_scores(frame):
    return frame.set_index("ID") if "ID" in frame.columns else frame.copy()


def short_samples(columns):
    return [f"S{i+1:02d}" for i in range(len(columns))]


def clean_labels(labels):
    replacements = {"Natural Killer Cell Cytotoxicity": "NK cytotoxicity",
                    "Natural killer cell cytotoxicity": "NK cytotoxicity",
                    "IFNG signature Ayers et al": "IFNG signature (Ayers)",
                    "Li et al": "(Li)", "TMEscore": "TME score"}
    output = []
    for label in labels:
        label = str(label).replace("_", " ")
        for old, new in replacements.items():
            label = label.replace(old, new)
        output.append(label)
    return output


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=10)
    ax.text(-0.12 if REVISION < 2 else -0.10, 1.10, letter, transform=ax.transAxes,
            weight="bold", fontsize=11, va="bottom")


def heatmap(ax, frame, max_rows=12, max_samples=30, scale=True, row_labels=True, colorbar=True):
    """Features x samples. Row scaling changes the display only."""
    frame = frame.astype(float)
    if len(frame) > max_rows:
        selected = frame.std(axis=1).nlargest(max_rows).index
        frame = frame.loc[selected]
    total_samples = frame.shape[1]
    frame = frame.iloc[:, :max_samples]
    display = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1).replace(0, 1), axis=0) if scale else frame
    if scale:
        limit = 2.5 if REVISION >= 2 else max(1.0, np.nanmax(np.abs(display.to_numpy())))
        im = ax.imshow(display, aspect="auto", cmap=CMAP if REVISION >= 1 else "coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    else:
        im = ax.imshow(display, aspect="auto", cmap=FRACTION_CMAP if REVISION >= 2 else "YlGnBu", vmin=0, interpolation="nearest")
    ax.set_yticks(range(len(frame)), clean_labels(frame.index) if row_labels else [""] * len(frame))
    ticks = np.arange(len(frame.columns))
    if len(ticks) > 15:
        ticks = ticks[::5]
    labels = short_samples(frame.columns)
    ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=90 if len(ticks) > 10 else 0)
    ax.set_xlabel("Samples" + (f" (first {max_samples} shown)" if max_samples < total_samples else ""))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.035, shrink=0.75)
        cb.set_label("Row z-score" if scale else "Fraction")
        if scale and REVISION >= 2:
            cb.set_ticks([-2.5, 0, 2.5])
        cb.outline.set_visible(False)
    return frame


def scatter(ax, x, y, xlabel, ylabel, annotate=False):
    ax.scatter(x, y, s=28 if len(x) < 30 else 12, c=TEAL, alpha=0.85 if len(x) < 30 else 0.55,
               edgecolors="white", linewidths=0.35, rasterized=False)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.margins(0.14)
    if annotate:
        chosen = set(range(len(x))) if REVISION == 0 else {np.argmin(x), np.argmax(x), np.argmin(y), np.argmax(y)}
        for i, (label, a, b) in enumerate(zip(short_samples(x), x, y)):
            if i not in chosen:
                continue
            ax.annotate(label, (a, b), xytext=(4, 3), textcoords="offset points", fontsize=6.5)


def fractions(frame, id_column=None):
    frame = frame.copy()
    if id_column and id_column in frame:
        frame = frame.set_index(id_column)
    return frame.select_dtypes(include=[np.number])


def composition_figure(figsize=(10.2, 4.8), ratios=(1, 1.2)):
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.23], width_ratios=ratios)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    legends = [fig.add_subplot(grid[1, i]) for i in range(2)]
    for ax in legends:
        ax.set_axis_off()
    return fig, axes, legends


def stacked(ax, frame, max_types=8, legend_ax=None):
    frame = frame.astype(float)
    if frame.shape[1] > max_types:
        top = frame.mean().nlargest(max_types).index
        reduced = frame[top].copy()
        reduced["Remaining types"] = frame.drop(columns=top).sum(axis=1)
        frame = reduced
    bottom = np.zeros(len(frame))
    for i, name in enumerate(frame):
        values = frame[name].to_numpy()
        ax.bar(np.arange(len(frame)), values, bottom=bottom, width=0.72,
               color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.35,
               label=clean_labels([name])[0])
        bottom += values
    ax.set(ylim=(0, 1), ylabel="Estimated fraction", xlabel="Samples")
    ax.set_xticks(range(len(frame)), short_samples(frame.index), rotation=0 if REVISION >= 2 else 90)
    ax.set_yticks([0, 0.5, 1])
    handles, labels = ax.get_legend_handles_labels()
    target = legend_ax if legend_ax is not None else ax
    target.legend(handles, labels, loc="upper left", ncol=2 if len(labels) > 8 else 3,
                  columnspacing=1.0, handlelength=1.1, fontsize=6.5,
                  **({"bbox_to_anchor": (0, -0.28)} if legend_ax is None else {}))


def save_figure(fig, slug):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"{slug}.{suffix}", dpi=220 if suffix == "png" else 300,
                    bbox_inches="tight", pad_inches=0.10, metadata={"Creator": "iobrx tutorials"} if suffix != "png" else None)
        if suffix == "svg":
            path = FIGURES / f"{slug}.svg"
            # Matplotlib adds trailing spaces to path coordinates; normalize
            # text formatting without changing SVG geometry or labels.
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    review = os.environ.get("IOBRX_REVIEW_DIR")
    if review:
        directory = Path(review) / f"round-{REVISION}"
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{slug}.png", dpi=160, bbox_inches="tight")
    plt.show()


def save_result(result, slug, seconds, input_shape):
    TABLES.mkdir(parents=True, exist_ok=True)
    frames = result if isinstance(result, dict) else {"result": result}
    shapes = {}
    for key, frame in frames.items():
        if isinstance(frame, pd.DataFrame):
            shapes[key] = list(frame.shape)
            # Huge count matrices are returned by the notebook, not duplicated.
            if frame.size <= 100000:
                frame.to_csv(TABLES / f"{slug}_{key}.csv", index=True)
    report = {"analysis": slug, "seconds": seconds, "input_shape": list(input_shape),
              "output_shapes": shapes, "threads": 8}
    (TABLES / f"{slug}_timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed in {seconds:.3f} s; input {input_shape[0]:,} features × {input_shape[1]:,} samples.")
    return pd.DataFrame([report])
