import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# same lists as clustering_baseline (duplicated to avoid circular import)
NUMERIC = ["LeadTime", "StaysInWeekNights", "PreviousCancellations",
           "PreviousBookingsNotCanceled", "DaysInWaitingList",
           "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
           "StaysInWeekendNights", "TotalGuests"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
DPI = 150


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_plots(df, Xt, labels, km, out):
    k = km.n_clusters
    unique = np.unique(labels)
    n = len(labels)

    # PCA 2D scatter (10k random sample)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, 10000, replace=False)
    pca = PCA(2, random_state=42)
    xy = pca.fit_transform(Xt[idx])
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in unique:
        m = labels[idx] == c
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.4, color=COLORS[c], label=f"C{c}")
    cxy = pca.transform(km.cluster_centers_)
    ax.scatter(cxy[:, 0], cxy[:, 1], s=200, marker="*", c=COLORS[:k], edgecolors="k", zorder=5)
    ax.set(title=f"PCA 2D  PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  (10k sample)",
           xlabel="PC1", ylabel="PC2")
    ax.legend(markerscale=2)
    _save(fig, out / "pca_2d_clusters.png")

    # heatmap of z-scored per-cluster means of numeric features
    tmp = df[NUMERIC].assign(cluster=labels)
    means = tmp.groupby("cluster")[NUMERIC].mean()
    pop_m, pop_s = df[NUMERIC].mean(), df[NUMERIC].std().replace(0, 1)
    z = (means - pop_m) / pop_s
    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(NUMERIC)))
    ax.set_xticklabels(NUMERIC, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{c}" for c in unique])
    for i in range(k):
        for j in range(len(NUMERIC)):
            ax.text(j, i, f"{z.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="z-score")
    ax.set_title("Cluster profile - numeric features (z-score vs population)")
    _save(fig, out / "cluster_profile_heatmap.png")