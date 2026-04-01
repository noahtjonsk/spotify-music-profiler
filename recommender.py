"""
recommender.py
--------------
Builds a KNN-based song similarity model from songs_clustered.csv and saves it.

Input:  a song name (partial match supported)
Output: 10 most similar tracks with artist, cluster name, and Euclidean distance

Usage after saving:
    import pickle
    with open("recommender.pkl", "rb") as f:
        rec = pickle.load(f)
    rec.recommend("Bohemian Rhapsody")

Requirements:
    pip install pandas scikit-learn
"""

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Force UTF-8 output so track names with non-ASCII characters print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("songs_clustered.csv")
print(f"Loaded {len(df)} tracks")

# Drop songs with the same (track_name, artist) but different track_ids
before = len(df)
df = df.drop_duplicates(subset=["track_name", "artist"], keep="first").reset_index(drop=True)
print(f"After (name, artist) dedup: {len(df)} tracks (removed {before - len(df)} duplicates)\n")

# Features used for similarity — same set used in clustering (no acousticness)
FEATURE_COLS = [
    "popularity", "danceability", "energy", "loudness", "mode",
    "speechiness", "instrumentalness", "liveness", "valence",
    "duration_ms", "time_signature", "tempo", "key",
]

# Use the scaled columns where available (popularity, danceability, etc. are
# already z-scored in songs_scaled; tempo and key are raw).
# songs_clustered.csv carries the raw values from songs_raw, so we re-apply
# the same StandardScaler used in eda_and_scale.py.
from sklearn.preprocessing import StandardScaler

NO_SCALE = ["tempo", "key"]
SCALE_COLS = [c for c in FEATURE_COLS if c not in NO_SCALE]

scaler = StandardScaler()
X_scale = scaler.fit_transform(df[SCALE_COLS])
X_raw   = df[NO_SCALE].values
X       = np.hstack([X_scale, X_raw])   # shape: (n_tracks, 13)

# ---------------------------------------------------------------------------
# 2. Fit NearestNeighbors
#    algorithm="ball_tree" is efficient for Euclidean distance at this size.
#    We ask for 11 neighbours so we can drop the query song itself (rank 0).
# ---------------------------------------------------------------------------
N_NEIGHBORS = 31   # extra headroom so recommend() can dedup and still return 10

knn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="euclidean", algorithm="ball_tree")
knn.fit(X)
print(f"NearestNeighbors fitted on {X.shape[0]} tracks, {X.shape[1]} features\n")

# ---------------------------------------------------------------------------
# 3. Import SongRecommender from its own module
#    This ensures pickle can resolve the class when loaded from any script.
# ---------------------------------------------------------------------------
from song_recommender import SongRecommender


# Instantiate and save
recommender = SongRecommender(
    knn=knn,
    scaler=scaler,
    X=X,
    df=df,
    feature_cols=FEATURE_COLS,
    scale_cols=SCALE_COLS,
    no_scale=NO_SCALE,
)

with open("recommender.pkl", "wb") as f:
    pickle.dump(recommender, f)
print("Model saved -> recommender.pkl\n")

# ---------------------------------------------------------------------------
# 4. Test on 5 well-known songs from different genres
# ---------------------------------------------------------------------------
TEST_SONGS = [
    "Bohemian Rhapsody",     # rock / classic rock
    "Billie Jean",           # pop / R&B
    "Lose Yourself",         # hip-hop
    "Clair de Lune",         # classical
    "Thunderstruck",         # metal
]

SEP = "=" * 65

for song in TEST_SONGS:
    print(SEP)
    try:
        results = recommender.recommend(song, n=10)
        print(results.to_string(index=False))
    except ValueError as e:
        print(f"  {e}")
    print()

print(SEP)
print("Done. Recommender saved to recommender.pkl")
