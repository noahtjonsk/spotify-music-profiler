"""
collect_songs.py
----------------
Loads the Spotify Tracks Dataset (dataset.csv) and saves a cleaned copy
as songs_raw.csv, keeping all 114 genres.

Dataset source: https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset
~114k tracks, 114 genres, all 13 Spotify audio features.

Requirements:
    pip install pandas
"""

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load the raw dataset
# ---------------------------------------------------------------------------
CSV_PATH = "dataset.csv"

print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, index_col=0)  # first column is an unnamed row index
print(f"Raw shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ---------------------------------------------------------------------------
# 2. Rename columns to match the project schema
# ---------------------------------------------------------------------------
RENAME = {
    "track_id":         "track_id",
    "track_name":       "track_name",
    "artists":          "artist",
    "album_name":       "album",
    "popularity":       "popularity",
    "track_genre":      "genre",
    "danceability":     "danceability",
    "energy":           "energy",
    "key":              "key",
    "loudness":         "loudness",
    "mode":             "mode",
    "speechiness":      "speechiness",
    "acousticness":     "acousticness",
    "instrumentalness": "instrumentalness",
    "liveness":         "liveness",
    "valence":          "valence",
    "tempo":            "tempo",
    "duration_ms":      "duration_ms",
    "time_signature":   "time_signature",
}

# Keep only columns present in the file (guards against schema drift)
available = {k: v for k, v in RENAME.items() if k in df.columns}
df = df[list(available.keys())].rename(columns=available)

# ---------------------------------------------------------------------------
# 3. Normalise genre names
#    The dataset uses "r-n-b" — rename to "r&b" for readability.
# ---------------------------------------------------------------------------
GENRE_MAP = {"r-n-b": "r&b"}
df["genre"] = df["genre"].replace(GENRE_MAP)
print(f"Genres in dataset: {df['genre'].nunique()}")

# ---------------------------------------------------------------------------
# 4. Deduplicate by track_id — the dataset lists each track once per genre,
#    so the same song can appear multiple times with different genre labels.
#    We keep the first occurrence.
# ---------------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset="track_id", keep="first")
print(f"After track_id dedup: {len(df)} rows (removed {before - len(df)} duplicates)")

# Also drop songs with the same (track_name, artist) but different track_ids
# (same song released on multiple albums / regions)
before2 = len(df)
df = df.drop_duplicates(subset=["track_name", "artist"], keep="first")
print(f"After (name, artist) dedup: {len(df)} rows (removed {before2 - len(df)} duplicates)\n")

print(f"Top 20 genres by track count:")
print(df["genre"].value_counts().head(20).to_string())

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
output_path = "songs_raw.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} tracks -> '{output_path}'")
print(df.head())
