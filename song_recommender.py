"""
song_recommender.py
-------------------
SongRecommender class — imported by both recommender.py (to build + pickle)
and app.py (to unpickle). Keeping it in its own module means pickle can always
resolve the class regardless of which script is __main__.
"""

import pandas as pd
import numpy as np


class SongRecommender:
    """
    KNN-based song similarity tool.

    Attributes
    ----------
    knn          : fitted NearestNeighbors model
    scaler       : fitted StandardScaler
    X            : scaled feature matrix aligned with df
    df           : songs_clustered dataframe (track metadata + cluster labels)
    feature_cols : all feature column names used at fit time
    scale_cols   : subset that was z-scored
    no_scale     : subset kept in raw units (tempo, key)
    """

    def __init__(self, knn, scaler, X, df, feature_cols, scale_cols, no_scale):
        self.knn          = knn
        self.scaler       = scaler
        self.X            = X
        self.df           = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.scale_cols   = scale_cols
        self.no_scale     = no_scale

    def _find_track(self, song_name: str) -> int:
        """
        Return the dataframe index of the best match for song_name.
        Tries exact match first, then case-insensitive substring match.
        Raises ValueError if nothing is found.
        """
        name_lower = song_name.lower()
        col = self.df["track_name"]

        # 1. Exact (case-insensitive)
        exact = col[col.str.lower() == name_lower]
        if not exact.empty:
            return exact.index[0]

        # 2. Substring
        sub = col[col.str.lower().str.contains(name_lower, regex=False, na=False)]
        if not sub.empty:
            if len(sub) > 1:
                print(f"  [{len(sub)} matches for '{song_name}', using: "
                      f"\"{self.df.loc[sub.index[0], 'track_name']}\" "
                      f"by {self.df.loc[sub.index[0], 'artist']}]")
            return sub.index[0]

        raise ValueError(
            f"Song '{song_name}' not found in dataset. "
            "Try a partial title or check spelling."
        )

    def recommend(self, song_name: str, n: int = 10, idx: int = None) -> pd.DataFrame:
        """
        Return the n most similar songs to song_name.

        Parameters
        ----------
        song_name : str      — full or partial track name (used when idx is None)
        n         : int      — number of recommendations (default 10)
        idx       : int|None — pre-resolved df index; skips _find_track when provided
                               (use this from the app to avoid double-resolving)

        Returns
        -------
        pd.DataFrame with columns:
            rank, track_name, artist, genre, cluster_name, distance
        """
        if idx is None:
            idx = self._find_track(song_name)
        query_vec = self.X[idx].reshape(1, -1)
        query_row = self.df.loc[idx]

        print(f"Query: \"{query_row['track_name']}\" by {query_row['artist']}")
        print(f"  Genre: {query_row['genre']}  |  "
              f"Cluster: {query_row['cluster']} — {query_row['cluster_name']}\n")

        # Fetch extra neighbours to account for duplicates being removed
        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=n * 3 + 1)
        distances = distances.flatten()
        indices   = indices.flatten()

        idx_int   = int(idx)
        mask      = indices != idx_int
        distances = distances[mask]
        # Convert to plain Python ints so iloc has no type ambiguity
        positions = list(map(int, indices[mask]))

        col_positions = [self.df.columns.get_loc(c)
                         for c in ["track_id", "track_name", "artist", "genre", "cluster", "cluster_name"]]
        results = self.df.iloc[positions, col_positions].copy()
        results["distance"] = distances.round(4)

        # Remove the query track itself (by track_id, more robust than position)
        query_track_id = self.df.at[int(idx), "track_id"]
        results = results[results["track_id"] != query_track_id]

        # Deduplicate by track_id, then by (track_name, artist) for same song with different IDs
        results = results.drop_duplicates(subset="track_id", keep="first")
        results = results.drop_duplicates(subset=["track_name", "artist"], keep="first")

        results = results.drop(columns="track_id").head(n).reset_index(drop=True)
        results.insert(0, "rank", range(1, len(results) + 1))
        return results
