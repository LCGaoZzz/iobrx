"""Display-only styling for the additional tutorial figures."""
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_result(frame, title, revision=2, groups=None):
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    variance = frame.std()
    if revision:
        variance = variance[variance > 0]
    features = variance.nlargest(min(12, len(variance))).index
    view = frame.loc[:, features].iloc[:10]
    view = view.sub(view.mean()).div(view.std().replace(0, 1)).fillna(0)
    width = 6.4 if revision < 2 else 7.2
    fig, ax = plt.subplots(figsize=(width, max(2.8, len(features) * .27 + 1.1)), layout="constrained")
    cmap = "coolwarm" if revision == 0 else LinearSegmentedColormap.from_list("balanced", ["#355F78", "#FAF9F5", "#B66A51"])
    im = ax.imshow(view.T, aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
    labels = [str(x).replace("_", " ") for x in features]
    if revision:
        labels = [x.removesuffix(" CIBERSORT").removesuffix(" IPS").removesuffix(" BayesPrism") for x in labels]
    if revision >= 2:
        labels = [textwrap.fill(x, 34) for x in labels]
    ax.set_yticks(range(len(features)), labels, fontsize=7 if revision else 9)
    sample_labels = [f"S{i+1:02d}" for i in range(len(view))]
    if groups is not None and revision:
        sample_labels = [f"{label}\n{group}" for label, group in zip(sample_labels, groups)]
    ax.set_xticks(range(len(view)), sample_labels, fontsize=7 if revision else 9)
    ax.set_xlabel("Samples · identifiers retained in the saved table" if revision >= 2 else "Samples")
    ax.set_title(title, loc="left", weight="medium", pad=12)
    ax.tick_params(length=0, pad=5)
    if revision:
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, shrink=.65, fraction=.035, pad=.04)
    bar.set_label("Feature z-score\n(display only)")
    if revision >= 2:
        bar.set_ticks([-2.5, 0, 2.5])
    bar.outline.set_visible(revision == 0)
    return fig
