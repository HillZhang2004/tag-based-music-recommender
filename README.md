# Tag-Based Music Recommender

A small tag-based music recommender that models my Spotify taste using metadata from **Spotify + Last.fm**.  
The pipeline builds a labeled dataset of tracks, represents each track with **bag-of-tags** features, and trains a **logistic regression** model to predict the probability that a track is a “like.”

- **Report (2 pages + appendix):** `docs/Project_Report.pdf`
- **Slides:** https://docs.google.com/presentation/d/1LXsRlMVp0Dyil7INCxJqCdAjInD7x64NV65LvkJY0xQ/edit?usp=sharing

---

## Quickstart (train + recommend)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train from the included labeled dataset
python3 train_from_api_dataset.py

# Rank a CSV of candidate tracks
python3 recommend_from_csv.py --csv data/top_tracks_lastfm.csv --topn 20
```

**Outputs**
- Saved model artifacts: `artifacts/tag_model.joblib`, `artifacts/tag_vectorizer.joblib`
- Ranked CSV written by the recommender script (default: `recommendations_from_top_tracks_lastfm.csv`)
  - An example output is included at `outputs/recommendations_example.csv`

---

## Try the Streamlit UI

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

This ranks tracks from `data/top_tracks_lastfm.csv` by predicted like probability.

(Optional) Add a screenshot at `docs/ui.png`, then embed it here:

```md
![Streamlit UI](docs/ui.png)
```

---

## Repo structure

### Core
- `train_from_api_dataset.py`  
  Loads `data/api_labeled_tracks.csv`, builds tag features with `CountVectorizer`, runs a 60/20/20 train–dev–test split, tunes `C`, evaluates, then refits and saves:
  - `artifacts/tag_model.joblib`
  - `artifacts/tag_vectorizer.joblib`
- `preprocess.py`  
  Helpers for loading the labeled dataset and converting the `tags` column into bag-of-tags features.
- `recommend_from_csv.py`  
  Loads the saved model + vectorizer and ranks a CSV of candidate tracks.
- `sample_and_score.py`  
  Samples tracks from a CSV (biased toward popular tags), scores them, and writes an output CSV with scores.
- `interactive_demo.py`  
  Terminal demo: input comma-separated tags (e.g. `soul,rnb,bedroom pop`) and return a predicted like probability.

### Apps
- `streamlit_app.py` — UI that ranks tracks from `data/top_tracks_lastfm.csv`.
- `app.py` — type arbitrary tags and return a single predicted like probability.

### Optional
- `test_api.py` — rebuild `data/api_labeled_tracks.csv` from scratch using Spotify + Last.fm APIs (requires keys in `.env`).

---

## Data

- `data/api_labeled_tracks.csv` — labeled dataset (529 rows: 329 positives, 200 negatives).
- `data/top_tracks_lastfm.csv` — candidate tracks used for ranking demos.
- `data/hill_top_tracks_lastfm.csv` — additional candidate track file (optional).
- `data/derived/tag_counts.csv` and `data/derived/tag_dictionary.csv` — derived tag summaries used for analysis/debugging.

API keys are **not required** for training and demos because the CSVs above are included.  
Keys are only needed if you run `test_api.py` to rebuild the dataset.

---

## Model summary

- **Positives (label 1):** my medium-term top tracks and saved tracks on Spotify.
- **Negatives (label 0):** Last.fm tracks whose tags do not overlap with the positive tag set  
  and do not duplicate any positive (track, artist) pair.
- **After cleaning:** 529 tracks (329 positive, 200 negative), 1123 unique tags.
- **Features:** binary bag-of-tags via `CountVectorizer` with a custom tokenizer.
- **Model:** logistic regression (`liblinear`), tuned over `C ∈ {0.01, 0.1, 1.0, 10.0}` using dev-set F1.

Note: In the included split, the tuned model reaches very high test accuracy.  
The dataset construction makes negatives intentionally far from positives in tag space, so this is an “easy” classification setting.

---

## Optional: rebuild the dataset from APIs

If you want to rebuild the labeled dataset from scratch, create a `.env` file and add your keys
(Spotify + Last.fm), then run:

```bash
python3 test_api.py
```
