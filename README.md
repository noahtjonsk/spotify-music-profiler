# Spotify Music Profiler

An end-to-end data science project that clusters ~81,000 songs by audio profile and surfaces similar tracks through a KNN recommender with an interactive Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)

---

## What it does

1. **Data collection** - loads a Kaggle dataset of 114k Spotify tracks across 114 genres, deduplicates by track ID and by (track name, artist) → 81,344 unique tracks
2. **EDA & preprocessing** - distribution plots, Pearson correlation heatmap, drops `acousticness` (r = −0.73 with `energy`), StandardScaler on 11 features (tempo and key kept in raw units)
3. **Clustering** - K-Means sweep k=2..12 with silhouette + elbow selection, Ward hierarchical dendrogram for validation, final model at k=5
4. **KNN recommender** - `NearestNeighbors` (Euclidean, ball-tree) fitted on the scaled feature matrix, wrapped in a `SongRecommender` class and serialised to `recommender.pkl`
5. **Streamlit app** - live-filtered song search dropdown, results table, radar chart comparing the query song's audio profile to the result average

---

## Project structure

```
spotify-profiler/
├── collect_songs.py       # data pipeline: loads dataset.csv, deduplicates, saves songs_raw.csv
├── eda_and_scale.py       # EDA plots + StandardScaler → songs_scaled.csv
├── cluster.py             # K-Means + hierarchical clustering → songs_clustered.csv
├── recommender.py         # fits KNN, runs test queries, saves recommender.pkl
├── song_recommender.py    # SongRecommender class (imported by recommender.py + app.py)
├── app.py                 # Streamlit app
├── plots/                 # saved figures (distributions, heatmap, dendrogram, profiles)
├── requirements.txt
└── README.md
```

Generated files (not in repo — see Setup):
```
dataset.csv               # raw Kaggle download (~114k tracks)
songs_raw.csv             # deduplicated 81,344 tracks
songs_scaled.csv          # z-scored features
songs_clustered.csv       # + cluster labels
recommender.pkl           # serialised SongRecommender
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset
Download **[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)** from Kaggle and place `dataset.csv` in the project root.

### 3. Run the pipeline
Each script reads from the previous step's output:
```bash
python collect_songs.py    # dataset.csv       -> songs_raw.csv
python eda_and_scale.py    # songs_raw.csv     -> songs_scaled.csv  +  plots/
python cluster.py          # songs_scaled.csv  -> songs_clustered.csv
python recommender.py      # songs_clustered.csv -> recommender.pkl
```

### 4. Launch the app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Dropped `acousticness` | Pearson r = −0.73 with `energy` — redundant in distance calculations |
| `tempo` and `key` not z-scored | Tempo has natural BPM units meaningful for distance; key is a circular categorical (0–11) where z-scoring implies a false ordinal relationship |
| Dedup by (track name, artist) | The Kaggle dataset lists the same song under multiple track IDs (different releases/regions). Deduplicating by both track ID and (name, artist) removes ~8,400 redundant rows |
| k=5 over silhouette-optimal k=3 | k=3 splits the corpus too coarsely. k=5 yields musically interpretable clusters (silhouette = 0.44, still strong) |
| Ball-tree for KNN | More efficient than brute-force for Euclidean distance on 81k × 13 matrices |
| `SongRecommender` in its own module | Avoids `pickle` `AttributeError` when deserialising from a different `__main__` context |

---

## Cluster profiles

| # | Name | n | Dominant genre | Key traits |
|---|------|---|----------------|------------|
| 0 | Mid-Tempo Balanced | 11,092 | sleep | Lowest energy, quietest, slowest (~77 BPM) |
| 1 | Fast & Intense | 16,271 | hardstyle | High energy + fast tempo (~144 BPM) |
| 2 | Groovy Mid-Tempo | 19,526 | salsa | High danceability, ~100 BPM |
| 3 | Instrumental | 23,761 | chicago-house | Highest instrumentalness, balanced danceability |
| 4 | Vocal / Rap | 10,694 | drum-and-bass | Highest speechiness + energy, fastest (~174 BPM) |

---

## Sample recommendations

**Query: "Clair de Lune" — Debussy**
| # | Track | Artist | Cluster |
|---|-------|--------|---------|
| 1 | Liebestraum No. 1 | Liszt & Barenboim | Mid-Tempo Balanced |
| 2 | 3 Etudes de Concert No. 3 | Liszt & Trifonov | Mid-Tempo Balanced |
| 3 | Gaspard de la nuit: Ondine | Ravel | Mid-Tempo Balanced |

**Query: "Billie Jean" — The Civil Wars**
| # | Track | Artist | Cluster |
|---|-------|--------|---------|
| 1 | Waiting for the Sun | The Doors | Groovy Mid-Tempo |
| 2 | Hurt | Johnny Cash | Groovy Mid-Tempo |
| 3 | No Existe el Amor | Cesar Costa | Groovy Mid-Tempo |
