"""
eda_and_scale.py
----------------
1. Loads songs_raw.csv
2. Plots the distribution of each audio feature  -> plots/feature_distributions.png
3. Plots a Pearson correlation heatmap           -> plots/correlation_heatmap.png
4. Reports highly correlated feature pairs
5. Standardizes numerical features (excludes tempo and key)
6. Saves scaled data                             -> songs_scaled.csv

Requirements:
    pip install pandas matplotlib seaborn scikit-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv("songs_raw.csv")
print(f"Loaded {len(df)} tracks, {df.shape[1]} columns\n")

# ---------------------------------------------------------------------------
# 2. Define feature groups
#    - AUDIO_FEATURES: the 13 Spotify features we model on
#    - SCALE_FEATURES: features passed to StandardScaler (tempo + key excluded)
#    - NO_SCALE:       kept as-is (tempo has natural BPM units; key is categorical 0-11)
# ---------------------------------------------------------------------------
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature",
]

NO_SCALE    = ["tempo", "key"]
SCALE_FEATURES = [f for f in AUDIO_FEATURES if f not in NO_SCALE]

# Also include popularity in scaling — it's a useful signal
ALL_NUMERIC = ["popularity"] + AUDIO_FEATURES
SCALE_ALL   = ["popularity"] + SCALE_FEATURES

# ---------------------------------------------------------------------------
# 3. Feature distribution plots
#    One subplot per feature, histogram + KDE, coloured by genre overlay removed
#    for clarity (too many genres). Genre breakdown comes from the heatmap.
# ---------------------------------------------------------------------------
print("Plotting feature distributions ...")

n_cols = 4
n_rows = -(-len(ALL_NUMERIC) // n_cols)   # ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
axes = axes.flatten()

for i, feat in enumerate(ALL_NUMERIC):
    ax = axes[i]
    ax.hist(df[feat].dropna(), bins=50, color="#1DB954", edgecolor="none", alpha=0.85, density=True)
    # KDE overlay
    df[feat].dropna().plot.kde(ax=ax, color="#191414", linewidth=1.5)
    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=8)
    # Annotate mean and std
    mu, sigma = df[feat].mean(), df[feat].std()
    ax.axvline(mu, color="red", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(0.97, 0.93, f"μ={mu:.2f}\nσ={sigma:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f"Audio Feature Distributions (n={len(df):,})", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> plots/feature_distributions.png\n")

# ---------------------------------------------------------------------------
# 4. Pearson correlation heatmap
# ---------------------------------------------------------------------------
print("Plotting correlation heatmap ...")

corr = df[ALL_NUMERIC].corr(method="pearson")

fig, ax = plt.subplots(figsize=(12, 9))
mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
# Mask the upper triangle to avoid redundancy
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

sns.heatmap(
    corr,
    mask=mask_upper,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.4,
    linecolor="white",
    annot_kws={"size": 8},
    ax=ax,
)
ax.set_title("Pearson Correlation — Audio Features", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> plots/correlation_heatmap.png\n")

# ---------------------------------------------------------------------------
# 5. Report highly correlated pairs (|r| >= 0.70)
# ---------------------------------------------------------------------------
print("=" * 55)
print("Highly correlated feature pairs  |r| >= 0.70")
print("=" * 55)

high_corr = []
for i, r in enumerate(ALL_NUMERIC):
    for j, c in enumerate(ALL_NUMERIC):
        if j <= i:
            continue
        v = corr.loc[r, c]
        if abs(v) >= 0.70:
            high_corr.append((r, c, v))

if high_corr:
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    for r, c, v in high_corr:
        direction = "positive" if v > 0 else "negative"
        print(f"  {r:20s} <-> {c:20s}  r = {v:+.3f}  ({direction})")
else:
    print("  No pairs above threshold.")

print()

# ---------------------------------------------------------------------------
# 6. Standardize with StandardScaler
#    - SCALE_ALL features get z-scored  (mean=0, std=1)
#    - tempo is kept in BPM as-is
#    - key is kept as integer 0-11 as-is
#    - mode, time_signature are included in SCALE_ALL (they're ordinal/binary
#      and benefit from the same scale space as the other features)
# ---------------------------------------------------------------------------
print("Scaling features ...")
print(f"  Scaled  : {SCALE_ALL}")
print(f"  Kept raw: {NO_SCALE}\n")

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[SCALE_ALL] = scaler.fit_transform(df[SCALE_ALL])

# Verify
print("Post-scaling stats (should be ~mean=0, std=1 for scaled cols):")
print(df_scaled[SCALE_ALL].describe().loc[["mean", "std"]].round(4).to_string())
print()

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
df_scaled.to_csv("songs_scaled.csv", index=False)
print(f"Saved {len(df_scaled)} rows -> songs_scaled.csv")
print("Columns:", list(df_scaled.columns))
