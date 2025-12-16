# Tag-Based Music Recommender

**Hill Zhang**  
Last updated: **2025-12-13**

A small tag-based music recommender that models my Spotify taste using metadata from Spotify + Last.fm. The pipeline builds a labeled dataset of tracks, represents each track with bag-of-tags features, and trains a logistic regression model to predict the probability that a track is a “like.”

A short writeup is included in **`docs/Project_Report.pdf`** (2 pages + appendix).  
Slides: https://docs.google.com/presentation/d/1LXsRlMVp0Dyil7INCxJqCdAjInD7x64NV65LvkJY0xQ/edit?usp=sharing
---

## Repo layout
- `train_from_api_dataset.py` — main training script. Loads `data/api_labeled_tracks.csv`, builds tag features with `CountVectorizer`, runs a 60/20/20 train–dev–test split, tunes `C`, evaluates, then refits and saves `artifacts/tag_model.joblib` and `artifacts/tag_vectorizer.joblib`.
- `preprocess.py` — helper functions for loading the labeled dataset and converting the `tags` column into bag-of-tags features.
- `recommend_from_csv.py` — CLI script that loads the saved model + vectorizer, scores a CSV of candidate tracks, and prints/saves a ranked list.
- `sample_and_score.py` — samples tracks from a CSV (biased toward popular tags), scores them, and writes an output CSV with scores.
- `interactive_demo.py` — terminal demo: input a comma-separated tag list (e.g. `soul,rnb,bedroom pop`) and return a predicted like probability.
- `streamlit_app.py` — Streamlit UI that ranks tracks from `data/top_tracks_lastfm.csv`.
- `app.py` — small Streamlit app: type arbitrary tags and return a single predicted like probability.
- `data/api_labeled_tracks.csv` — labeled dataset (529 rows: 329 positives, 200 negatives).
- `data/top_tracks_lastfm.csv` — candidate tracks used for ranking demos.
- `test_api.py` — optional script to rebuild `data/api_labeled_tracks.csv` from scratch using Spotify + Last.fm APIs (requires keys in `.env`).

---

## Setup

Python **3.10+** recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

If there is no requirements.txt, typical deps are:
pandas numpy scikit-learn joblib streamlit spotipy python-dotenv requests

API keys are not required for training + demos because data/api_labeled_tracks.csv and data/top_tracks_lastfm.csv are already included.

How to run
1) Train the model from the labeled CSV

Retrains logistic regression on data/api_labeled_tracks.csv and overwrites artifacts/tag_model.joblib and artifacts/tag_vectorizer.joblib.

source .venv/bin/activate
python train_from_api_dataset.py


Expected output includes train/dev/test sizes, dev-set scores across C, and a final test-set report.

2) Score a CSV of candidate tracks

Ranks tracks by predicted like probability. The input CSV should contain track_name, artist_name, and tags.

python recommend_from_csv.py --csv data/top_tracks_lastfm.csv --topn 20


This prints top recommendations and writes: recommendations_from_top_tracks_lastfm.csv

3) Sample-and-score sanity check
python sample_and_score.py --csv data/top_tracks_lastfm.csv --n_samples 50 --out sampled_songs_with_scores.csv

4) Interactive demos

Terminal demo:

python interactive_demo.py
# example input: soul,rnb,trap,bedroom pop


Streamlit ranking UI:

streamlit run streamlit_app.py


Small “type tags → probability” app:

streamlit run app.py

Data and model (short summary)

Positives (label 1): my medium-term top tracks and saved tracks on Spotify.

Negatives (label 0): Last.fm tracks whose tags do not overlap with the positive tag set and do not duplicate any positive (track, artist) pair.

After cleaning: 529 tracks (329 positive, 200 negative), 1123 unique tags.

Features: binary bag-of-tags via CountVectorizer with a custom tokenizer.

Model: logistic regression (liblinear), tuned over C ∈ {0.01, 0.1, 1.0, 10.0} using dev-set F1.


In the included split, the final tuned model reaches ~99% test accuracy; the dataset construction makes negatives intentionally far from positives in tag space, so this is an “easy” classification setting.

