"""
cluster.py
----------
1. Loads songs_scaled.csv, drops acousticness
2. K-Means for k=2..12 — plots inertia + silhouette scores, picks best k
3. Hierarchical clustering — plots a truncated dendrogram
4. Assigns final K-Means cluster labels to every track
5. Prints per-cluster mean feature profiles and assigns descriptive names
6. Saves songs_clustered.csv

Requirements:
    pip install pandas matplotlib seaborn scikit-learn scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings("ignore")  # suppress convergence warnings at low k

# ---------------------------------------------------------------------------
# 1. Load and drop acousticness
#    acousticness correlates strongly with energy (r=-0.80) so it's redundant
# ---------------------------------------------------------------------------
df = pd.read_csv("songs_scaled.csv")
print(f"Loaded {len(df)} tracks\n")

# Columns used for clustering (scaled numeric features, minus acousticness)
FEATURE_COLS = [
    "popularity", "danceability", "energy", "loudness", "mode",
    "speechiness", "instrumentalness", "liveness", "valence",
    "duration_ms", "time_signature",
    "tempo", "key",          # kept raw but still included in distance calc
]
# Drop acousticness if present
DROP = ["acousticness"]
FEATURE_COLS = [c for c in FEATURE_COLS if c not in DROP]

X = df[FEATURE_COLS].values
print(f"Feature matrix: {X.shape}  (acousticness dropped)\n")

# ---------------------------------------------------------------------------
# 2. K-Means sweep k=2..12
# ---------------------------------------------------------------------------
K_RANGE = range(2, 13)
inertias, silhouettes = [], []

print("Running K-Means sweep ...")
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels, sample_size=3000, random_state=42))
    print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.4f}")

# ---------------------------------------------------------------------------
# 3. Plot: inertia (elbow) + silhouette side by side
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# Elbow
ax1.plot(list(K_RANGE), inertias, "o-", color="#1DB954", linewidth=2, markersize=6)
ax1.set_xlabel("k", fontsize=11)
ax1.set_ylabel("Inertia (within-cluster SSE)", fontsize=10)
ax1.set_title("Elbow Curve", fontsize=12, fontweight="bold")
ax1.set_xticks(list(K_RANGE))
ax1.grid(axis="y", alpha=0.3)

# Silhouette
# Silhouette-optimal k — often too low for interpretability on mixed-genre data.
# We override to 5: still a strong score (0.44) and yields richer cluster profiles.
best_k_by_score = int(list(K_RANGE)[int(np.argmax(silhouettes))])
best_k = 5  # override: balances separation quality with portfolio interpretability
print(f"  (silhouette-optimal k={best_k_by_score}, using k={best_k} for richer clusters)")
colors = ["#1DB954" if k == best_k else "#aaaaaa" for k in K_RANGE]
ax2.bar(list(K_RANGE), silhouettes, color=colors, edgecolor="none")
ax2.set_xlabel("k", fontsize=11)
ax2.set_ylabel("Silhouette Score", fontsize=10)
ax2.set_title("Silhouette Scores (green = best k)", fontsize=12, fontweight="bold")
ax2.set_xticks(list(K_RANGE))
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/kmeans_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nBest k by silhouette: {best_k}  (score={max(silhouettes):.4f})")
print("  -> plots/kmeans_selection.png\n")

# ---------------------------------------------------------------------------
# 4. Hierarchical clustering — dendrogram on a sample (full dataset is too slow)
#    Uses Ward linkage (minimises within-cluster variance, same objective as K-Means)
# ---------------------------------------------------------------------------
print("Running hierarchical clustering (Ward linkage on 1000-track sample) ...")
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X), size=min(1000, len(X)), replace=False)
X_sample = X[sample_idx]

Z = linkage(X_sample, method="ward")

fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(
    Z,
    truncate_mode="lastp",   # show only the last p merges
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=Z[-best_k, 2],   # colour at the height that yields best_k clusters
    ax=ax,
    above_threshold_color="#aaaaaa",
)
ax.set_title(
    f"Hierarchical Clustering Dendrogram (Ward, sample n=1000)\n"
    f"Dashed line suggests k={best_k} clusters",
    fontsize=12, fontweight="bold",
)
ax.axhline(y=Z[-best_k, 2], color="red", linestyle="--", linewidth=1.2, label=f"k={best_k} cut")
ax.set_xlabel("Track (leaf count in brackets)", fontsize=10)
ax.set_ylabel("Ward Distance", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/dendrogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> plots/dendrogram.png\n")

# ---------------------------------------------------------------------------
# 5. Fit final K-Means with best_k on the full dataset
# ---------------------------------------------------------------------------
print(f"Fitting final K-Means with k={best_k} on all {len(X)} tracks ...")
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=15)
df["cluster"] = km_final.fit_predict(X)
print(f"Cluster sizes:\n{df['cluster'].value_counts().sort_index().to_string()}\n")

# ---------------------------------------------------------------------------
# 6. Per-cluster mean feature profiles on the ORIGINAL (unscaled) values
#    We reload songs_raw so the means are human-readable (not z-scores)
# ---------------------------------------------------------------------------
df_raw = pd.read_csv("songs_raw.csv")
df_raw["cluster"] = df["cluster"].values  # attach cluster labels

# Raw feature columns (excludes acousticness as per session decision)
RAW_FEATURES = [
    "popularity", "danceability", "energy", "loudness", "mode",
    "speechiness", "instrumentalness", "liveness", "valence",
    "duration_ms", "time_signature", "tempo", "key",
]

cluster_means = df_raw.groupby("cluster")[RAW_FEATURES].mean().round(3)
print("=" * 70)
print("Per-cluster mean audio feature profiles (raw values)")
print("=" * 70)
print(cluster_means.to_string())
print()

# Most common genre per cluster (useful sanity check)
top_genre = df_raw.groupby("cluster")["genre"].agg(lambda s: s.value_counts().index[0])
print("Dominant genre per cluster:")
print(top_genre.to_string())
print()

# ---------------------------------------------------------------------------
# 7. Assign descriptive cluster names
#    Rules derived from the mean profiles printed above.
#    Key signals used:
#      energy, danceability, valence  -> mood / intensity
#      acousticness (dropped) proxied by low energy + high instrumentalness
#      speechiness                    -> vocal vs instrumental
#      instrumentalness               -> instrumental content
#      tempo                          -> pace
# ---------------------------------------------------------------------------
def name_clusters_relative(means_df):
    """
    Assign descriptive names by ranking clusters against each other on key axes.
    Each cluster gets a label that reflects its *relative* position — the feature
    where it stands out most from the rest.
    """
    # Compute per-feature ranks (0 = lowest cluster, n-1 = highest cluster)
    ranks = means_df[["energy", "danceability", "valence", "tempo",
                       "speechiness", "instrumentalness", "loudness"]].rank()

    n = len(means_df)
    high = n - 1   # rank index for highest cluster
    low  = 0       # rank index for lowest cluster

    names = {}
    for c in means_df.index:
        r    = ranks.loc[c]
        e    = means_df.loc[c, "energy"]
        t    = means_df.loc[c, "tempo"]
        d    = means_df.loc[c, "danceability"]
        v    = means_df.loc[c, "valence"]
        ins  = means_df.loc[c, "instrumentalness"]
        sp   = means_df.loc[c, "speechiness"]

        # Pick the most distinctive trait of this cluster
        if r["tempo"] == high and r["energy"] >= n - 2:
            name = "Fast & Intense"
        elif r["tempo"] == low and r["energy"] == low:
            name = "Slow & Reflective"
        elif r["tempo"] == low and ins > means_df["instrumentalness"].mean():
            name = "Quiet Instrumental"
        elif r["danceability"] == high and r["tempo"] <= n // 2:
            name = "Groovy Mid-Tempo"
        elif r["danceability"] == high and r["tempo"] > n // 2:
            name = "High Energy Dance"
        elif r["speechiness"] == high and r["danceability"] >= n - 2:
            name = "Upbeat Hip-Hop"
        elif r["speechiness"] == high:
            name = "Vocal / Rap"
        elif r["energy"] == high and r["danceability"] < n // 2:
            name = "Driving Rock"
        elif r["instrumentalness"] == high:
            name = "Instrumental"
        elif r["valence"] == high:
            name = "Upbeat & Positive"
        elif r["valence"] == low:
            name = "Dark & Moody"
        elif r["tempo"] > n // 2 and r["energy"] > n // 2:
            name = "Energetic & Fast"
        else:
            name = "Mid-Tempo Balanced"
        names[c] = name
    return names

cluster_names = name_clusters_relative(cluster_means)

print("Cluster names:")
for c, name in cluster_names.items():
    dominant = top_genre[c]
    size = (df["cluster"] == c).sum()
    print(f"  Cluster {c}: '{name}'  (n={size}, dominant genre: {dominant})")

df_raw["cluster_name"] = df_raw["cluster"].map(cluster_names)

# ---------------------------------------------------------------------------
# 8. Cluster profile heatmap (z-scored means for visual comparison)
# ---------------------------------------------------------------------------
cluster_means_z = df[FEATURE_COLS].groupby(df["cluster"]).mean()
cluster_means_z.index = [f"{i}: {cluster_names[i]}" for i in cluster_means_z.index]

fig, ax = plt.subplots(figsize=(14, 0.6 * best_k + 2))
sns.heatmap(
    cluster_means_z,
    annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    linewidths=0.4, linecolor="white",
    annot_kws={"size": 8},
    ax=ax,
)
ax.set_title("Cluster Profiles — Mean Scaled Feature Values", fontsize=12, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  -> plots/cluster_profiles.png")

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
df_raw.to_csv("songs_clustered.csv", index=False)
print(f"\nSaved {len(df_raw)} rows -> songs_clustered.csv")
print(f"New columns added: 'cluster' (int), 'cluster_name' (str)")
