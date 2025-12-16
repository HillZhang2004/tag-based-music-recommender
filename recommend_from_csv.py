"""
CS158 Final Project: Tag-Based Music Recommender
File: recommend_from_csv.py
Author: Hill, Nathaniel, Shreyan
Date: 2025-12-02

AI Use:
We used an AI assistant for a few specific parts of this script:
  - After we wrote a rough bullet outline for what this file should do
    (parse args, load the model and vectorizer, load a candidate CSV,
     run predict_proba, sort by like_prob, and save a ranked CSV),
    we asked the AI to help proofread and smooth out the comments and
    print messages so that they read more clearly.
  - We asked about the standard way to set up a small CLI with argparse
    using a required "--csv" argument and an optional "--topn" argument.
    We already knew we wanted those flags from class and past scripts;
    the AI mainly helped confirm that our use of ArgumentParser and
    add_argument was reasonable.
  - We hit a specific bug when we first tried to pull out probabilities:
    we forgot to index the second column of predict_proba and got confused
    about the shape. We asked the AI to explain how predict_proba works
    for binary classification. After reading the explanation, we fixed
    the code to use model.predict_proba(X)[:, 1] in the section where
    we compute like_prob for each track.

The overall design of this script (command-line flags, loading the
saved joblib model/vectorizer, using predict_proba to get like scores,
sorting by probability, and saving a CSV) is based on the CS158
logistic regression workflow and our own notes. We also kept our
scratch work and Google Docs drafts that show how we wrote the training
pipeline and thought about the feature-building and evaluation logic,
although some of that text was proofread and checked by AI.
The AI did not design the algorithm; it helped us debug that predict_proba
detail and polish the comments/strings so the script is easier to read.

Summary:
This script scores a CSV of candidate tracks using the trained tag model.
It:
  1. Parses command-line arguments (--csv for the input file, --topn for how
     many recommendations to print).
  2. Loads artifacts/tag_model.joblib and artifacts/tag_vectorizer.joblib.
  3. Loads candidate tracks from the CSV and checks that the required columns
     exist (track_name, artist_name, tags).
  4. Transforms the "tags" column with the saved vectorizer and calls
     predict_proba to get like-probabilities.
  5. Sorts tracks by like_prob, prints the top N, and writes the full ranked
     list to recommendations_from_<input_name>.csv.

Usage:
  - Make sure artifacts/tag_model.joblib and artifacts/tag_vectorizer.joblib are in the current folder.
  - Then run something like:
        python recommend_from_csv.py --csv data/top_tracks_lastfm.csv --topn 20
"""

import argparse
import os

import joblib
import pandas as pd


def parse_tag_string(tag_str: str) -> str:
    """
    Normalize a Last.fm tag string like
    'pop, female vocalists, indie pop'
    into 'pop female_vocalists indie_pop'.

    This needs to stay consistent with the function that existed when the
    vectorizer was originally trained.
    """
    if not isinstance(tag_str, str):
        return ""
    tags = []
    for raw in str(tag_str).split(","):
        t = raw.strip().lower()
        if not t:
            continue
        t = t.replace(" ", "_")
        tags.append(t)
    return " ".join(tags)


def load_candidates(csv_path: str) -> pd.DataFrame:
    """
    Load candidate tracks from a CSV and make sure it has the required columns.
    The CSV must include at least: track_name, artist_name, tags.
    """
    df = pd.read_csv(csv_path)

    required = {"track_name", "artist_name", "tags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["tags"] = df["tags"].fillna("").astype(str)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Score candidate tracks from a CSV using the trained tag model."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file with candidate tracks (e.g., data/top_tracks_lastfm.csv)",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=20,
        help="How many top recommendations to print",
    )
    args = parser.parse_args()

    print("=== Step 1: Load model and vectorizer ===")
    model = joblib.load("artifacts/tag_model.joblib")
    vectorizer = joblib.load("artifacts/tag_vectorizer.joblib")

    print(f"=== Step 2: Load candidate tracks from {args.csv} ===")
    df = load_candidates(args.csv)
    print("Columns in CSV:", list(df.columns))

    print("=== Step 3: Build tag features and predict like-probabilities ===")
    # The vectorizer was originally fit on tag strings, so we can feed the same format here.
    X = vectorizer.transform(df["tags"])
    like_probs = model.predict_proba(X)[:, 1]
    df["like_prob"] = like_probs

    df_sorted = df.sort_values("like_prob", ascending=False).reset_index(drop=True)

    topn = min(args.topn, len(df_sorted))
    print("\n=== Top recommendations ===")
    for i in range(topn):
        row = df_sorted.iloc[i]
        print(
            f"{i+1:2d}. p_like={row['like_prob']:.3f}  "
            f"{row['track_name']} — {row['artist_name']}"
        )

    base = os.path.basename(args.csv)
    out_path = f"recommendations_from_{base}"
    df_sorted.to_csv(out_path, index=False)
    print(f"\nSaved full ranked list to {out_path}")


if __name__ == "__main__":
    main()
