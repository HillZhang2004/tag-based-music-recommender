# Tag-Based Music Recommender
**Hill Zhang**  
Last updated: **2025-12-13**

A small tag-based music recommender that models my Spotify taste using metadata from Spotify + Last.fm. The pipeline builds a labeled dataset of tracks, represents each track with bag-of-tags features, and trains a logistic regression model to predict the probability that a track is a “like.”

- Report (2 pages + appendix): `docs/Project_Report.pdf`
- Slides: https://docs.google.com/presentation/d/1LXsRlMVp0Dyil7INCxJqCdAjInD7x64NV65LvkJY0xQ/edit?usp=sharing

---

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 train_from_api_dataset.py
python3 recommend_from_csv.py --csv data/top_tracks_lastfm.csv --topn 20
```
Repo layout

train_from_api_dataset.py — main training script. Loads data/api_labeled_tracks.csv, builds tag features with CountVectorizer, runs a 60/20/20 train–dev–test split, tunes C, evaluates, then refits and saves:

artifacts/tag_model.joblib

artifacts/tag_vectorizer.joblib

preprocess.py — helper functions for loading the labeled dataset and converting the tags column into bag-of-tags features.

recommend_from_csv.py — CLI script that loads the saved model + vectorizer, scores a CSV of candidate tracks, and prints/saves a ranked list.

sample_and_score.py — samples tracks from a CSV (biased toward popular tags), scores them, and writes an output CSV with scores.

interactive_demo.py — terminal demo: input a comma-separated tag list (e.g. soul,rnb,bedroom pop) and return a predicted like probability.

streamlit_app.py — Streamlit UI that ranks tracks from data/top_tracks_lastfm.csv.

app.py — small Streamlit app: type arbitrary tags and return a single predicted like probability.

test_api.py — optional script to rebuild data/api_labeled_tracks.csv from scratch using Spotify + Last.fm APIs (requires keys in .env).

Data

data/api_labeled_tracks.csv — labeled dataset (529 rows: 329 positives, 200 negatives).

data/top_tracks_lastfm.csv — candidate tracks used for ranking demos.

data/hill_top_tracks_lastfm.csv — additional candidate track file (optional).

data/derived/tag_counts.csv and data/derived/tag_dictionary.csv — derived tag summaries used for analysis/debugging.

API keys are not required for training + demos because the CSVs above are included. Keys are only needed if you run test_api.py to rebuild the dataset.

How to run
1) Train the model

Retrains logistic regression on data/api_labeled_tracks.csv and overwrites:

artifacts/tag_model.joblib

artifacts/tag_vectorizer.joblib

source .venv/bin/activate
python3 train_from_api_dataset.py


Expected output includes train/dev/test sizes, dev-set scores across C, and a final test-set report.

2) Score a CSV of candidate tracks

Ranks tracks by predicted like probability. The input CSV should contain: track_name, artist_name, and tags.

python3 recommend_from_csv.py --csv data/top_tracks_lastfm.csv --topn 20


This prints top recommendations and writes an output CSV under outputs/ (example: outputs/recommendations_example.csv).

3) Sample-and-score sanity check
python3 sample_and_score.py --csv data/top_tracks_lastfm.csv --n_samples 50 --out outputs/sampled_songs_with_scores.csv

4) Interactive demos

Terminal demo:

python3 interactive_demo.py
# example input: soul,rnb,trap,bedroom pop


Streamlit ranking UI:

streamlit run streamlit_app.py


Small “type tags → probability” app:

streamlit run app.py

Model summary

Positives (label 1): my medium-term top tracks and saved tracks on Spotify.

Negatives (label 0): Last.fm tracks whose tags do not overlap with the positive tag set and do not duplicate any positive (track, artist) pair.

After cleaning: 529 tracks (329 positive, 200 negative), 1123 unique tags.

Features: binary bag-of-tags via CountVectorizer with a custom tokenizer.

Model: logistic regression (liblinear), tuned over C ∈ {0.01, 0.1, 1.0, 10.0} using dev-set F1.

Note: In the included split, the tuned model reaches ~99% test accuracy. The dataset construction makes negatives intentionally far from positives in tag space, so this is an “easy” classification setting.
