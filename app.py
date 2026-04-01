"""
app.py
------
Streamlit app for the Spotify song similarity recommender.

Run with:
    streamlit run app.py
"""

import os
import pickle
import numpy as np
from song_recommender import SongRecommender  # noqa: F401 — must be imported so pickle can resolve the class
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load the recommender (cached so it only loads once per session)
# ---------------------------------------------------------------------------
def _pkl_mtime():
    """Return the mtime of recommender.pkl so the cache invalidates on rebuild."""
    return os.path.getmtime("recommender.pkl")

@st.cache_resource
def load_recommender(_mtime):
    with open("recommender.pkl", "rb") as f:
        return pickle.load(f)

rec = load_recommender(_pkl_mtime())
n_tracks = len(rec.df)
n_genres = rec.df["genre"].nunique()

# Audio features shown on the radar chart (human-readable labels)
RADAR_FEATURES = [
    "danceability", "energy", "valence", "speechiness",
    "instrumentalness", "liveness",
]
RADAR_LABELS = [
    "Danceability", "Energy", "Valence", "Speechiness",
    "Instrumentalness", "Liveness",
]

# ---------------------------------------------------------------------------
# Radar chart helper
# ---------------------------------------------------------------------------
def radar_chart(query_vals: list[float], mean_vals: list[float],
                labels: list[str], query_name: str) -> plt.Figure:
    """
    Draw a radar (spider) chart comparing the query song to the result average.
    All values are assumed to be in [0, 1].
    """
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]
    q_vals = query_vals + query_vals[:1]
    m_vals = mean_vals  + mean_vals[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1F2E")

    # Grid styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.5, color="white")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=6.5, color="#888888")
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color("#333333")
    ax.grid(color="#333333", linewidth=0.7)

    # Results average — filled
    ax.fill(angles, m_vals, alpha=0.25, color="#4ECDC4")
    ax.plot(angles, m_vals, color="#4ECDC4", linewidth=2, label="Results avg")

    # Query song — filled
    ax.fill(angles, q_vals, alpha=0.20, color="#1DB954")
    ax.plot(angles, q_vals, color="#1DB954", linewidth=2, linestyle="--", label=query_name[:28])

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=8, facecolor="#1A1F2E", edgecolor="#333333",
        labelcolor="white",
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#1DB954; margin-bottom:0'>
        🎵 Spotify Song Recommender
    </h1>
    <p style='text-align:center; color:#aaaaaa; margin-top:4px'>
        Search for a song, pick the exact match, and find the 10 most similar tracks by audio profile
    </p>
    <hr style='border-color:#333333; margin:16px 0'>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Build a lookup: "Track Name — Artist" -> dataframe index
# Used to populate the dropdown and resolve the selected song unambiguously
# ---------------------------------------------------------------------------
@st.cache_data
def build_song_options(_df: pd.DataFrame) -> dict[str, int]:
    """Return an ordered dict of display label -> df index for every track."""
    seen: dict[str, int] = {}
    for i, label in enumerate(_df.index):
        name   = _df.at[label, "track_name"]
        artist = _df.at[label, "artist"]
        # Skip rows where name or artist is NaN
        if not isinstance(name, str) or not isinstance(artist, str):
            continue
        display = f"{name}  —  {artist}"
        if display not in seen:
            seen[display] = int(label)
    return seen  # {display_label: df_index}

song_options = build_song_options(rec.df)
all_labels   = [""] + sorted(song_options.keys())   # blank first so nothing is pre-selected

# ---------------------------------------------------------------------------
# Search bar — text input filters the selectbox; selectbox picks exact match
# ---------------------------------------------------------------------------
filter_text = st.text_input(
    label="Type to filter songs",
    placeholder="e.g. bad, bohemian, clair …",
    label_visibility="collapsed",
)

# Filter the option list in real time as the user types
if filter_text.strip():
    filtered = [
        lbl for lbl in all_labels[1:]   # skip the blank placeholder
        if filter_text.strip().lower() in lbl.lower()
    ]
    if not filtered:
        st.warning(f"No songs match '{filter_text}'. Try a different keyword.")
        st.stop()
    dropdown_options = filtered
else:
    dropdown_options = all_labels  # show everything if nothing typed yet

col_select, col_btn = st.columns([4, 1])
with col_select:
    selected_label = st.selectbox(
        label="Select a song",
        options=dropdown_options,
        label_visibility="collapsed",
    )
with col_btn:
    search = st.button("🔍 Search", use_container_width=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if selected_label:
    try:
        idx = song_options[selected_label]
        query_row  = rec.df.loc[idx]
        query_vals = [float(rec.df.loc[idx, f]) for f in RADAR_FEATURES]

        # Pass idx directly — avoids _find_track re-resolving to a different track
        results = rec.recommend(query_row["track_name"], n=10, idx=idx)

        st.markdown(
            f"<h3 style='color:white; margin-bottom:2px'>Results for "
            f"<span style='color:#1DB954'>&ldquo;{query_row['track_name']}&rdquo;</span> "
            f"<span style='color:#aaaaaa; font-size:0.75em'>by {query_row['artist']}</span></h3>"
            f"<p style='color:#aaaaaa; margin-top:0'>Genre: <b>{query_row['genre']}</b> &nbsp;|&nbsp; "
            f"Cluster: <b>{int(query_row['cluster'])} — {query_row['cluster_name']}</b></p>",
            unsafe_allow_html=True,
        )

        # ---- Two-column layout: table left, radar right ------------------
        col_table, col_radar = st.columns([3, 2], gap="large")

        with col_table:
            st.markdown("#### 10 Most Similar Songs")

            # Add a colour-coded cluster badge column
            display = results[["rank", "track_name", "artist", "genre", "cluster_name", "distance"]].copy()
            display.columns = ["#", "Track", "Artist", "Genre", "Cluster", "Distance"]
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "#":        st.column_config.NumberColumn(width="small"),
                    "Distance": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Cluster":  st.column_config.TextColumn(width="medium"),
                },
            )

            # Genre breakdown bar
            genre_counts = results["genre"].value_counts()
            st.markdown("**Genre breakdown of results**")
            genre_df = genre_counts.reset_index()
            genre_df.columns = ["Genre", "Count"]
            st.bar_chart(genre_df.set_index("Genre"), color="#1DB954", height=180)

        with col_radar:
            st.markdown("#### Audio Profile Comparison")

            # Compute mean radar values — look up result tracks in rec.df by name+artist
            result_keys = list(zip(results["track_name"], results["artist"]))
            mask = rec.df.set_index(["track_name", "artist"]).index.isin(result_keys)
            result_rows = rec.df[mask].head(len(results))
            mean_vals = [float(result_rows[f].mean()) for f in RADAR_FEATURES]

            # Clamp both to [0, 1] — radar features are already in this range
            query_vals_clamped = [max(0.0, min(1.0, v)) for v in query_vals]
            mean_vals_clamped  = [max(0.0, min(1.0, v)) for v in mean_vals]

            fig = radar_chart(
                query_vals_clamped,
                mean_vals_clamped,
                RADAR_LABELS,
                query_row["track_name"],
            )
            st.pyplot(fig, use_container_width=True)

            # Feature delta table beneath the radar
            st.markdown("**Feature values**")
            delta_df = pd.DataFrame({
                "Feature":      RADAR_LABELS,
                "Query":        [f"{v:.3f}" for v in query_vals_clamped],
                "Results avg":  [f"{v:.3f}" for v in mean_vals_clamped],
            })
            st.dataframe(delta_df, use_container_width=True, hide_index=True)

    except ValueError as e:
        st.error(str(e))
        st.info(
            f"💡 Try a partial song title. The dataset contains {n_tracks:,} tracks "
            f"across {n_genres} genres. Not every song is included."
        )

elif search and not selected_label:
    st.warning("Please select a song from the dropdown.")

# ---------------------------------------------------------------------------
# Footer / dataset info
# ---------------------------------------------------------------------------
st.markdown("<hr style='border-color:#333333; margin-top:40px'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.metric("Tracks indexed", f"{n_tracks:,}", help=f"Filtered from 114,000-track Kaggle dataset ({n_genres} genres, deduplicated)")
col2.metric("Genres", n_genres)
col3.metric("Audio features", "13")
