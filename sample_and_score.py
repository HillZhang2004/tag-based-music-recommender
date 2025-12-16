"""
CS158 Final Project: Tag-Based Music Recommender
File: sample_and_score.py
Author: Hill, Nathaniel, Shreyan
Date: 2025-12-02

AI Use:
This script is a side experiment for our project: instead of scoring every
song in a big CSV, we randomly sample songs with a bias toward popular tags
and then ask the model to score just that subset.

We designed the main idea ourselves:
  - Use a Counter over all tags in the CSV to estimate how "popular" each tag is.
  - Turn those tag counts into probabilities p(tag).
  - Give each song a popularity_score equal to the average p(tag) over its tags.
  - Sample songs without replacement using popularity_score as weights.
  - Score the sampled songs with our trained logistic regression model.

We used an AI assistant for a few concrete tasks:
  - We already knew we wanted to sample with probabilities proportional to
    popularity_score, but we asked the AI to confirm that using
    np.random.choice with a normalized weight vector was the standard way
    to do this in NumPy.
  - The AI helped us clean up some of the printing and error messages
    (for example, checking that the 'tags' column exists and explaining
    what happens if no tags are found).
  - After we retrained the CountVectorizer with a custom tokenizer
    parse_tag_string in train_from_api_dataset.py, we ran into a joblib
    error when loading artifacts/tag_vectorizer.joblib here. The AI explained that
    the vectorizer pickle was referencing __main__.parse_tag_string from
    the training script, so this file also needs a matching
    parse_tag_string definition so unpickling works.

We wrote the sampling logic, the Counter-based popularity computation,
and the scoring flow ourselves based on the course material. The AI’s
role was mainly to double-check our use of NumPy sampling, help us
resolve that parse_tag_string compatibility issue, and tidy up comments
so another student could follow what is happening.
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
import joblib


def parse_tag_string(tag_str):
    """
    Turn a tag string into a list of tag tokens.

    This matches the helper we used in train_from_api_dataset.py when we
    trained the vectorizer with tokenizer=parse_tag_string, which is why
    the joblib pickle expects to find parse_tag_string on __main__.

    Handles strings like:
      "pop, dance, female vocalists"
      "['pop', 'dance', 'female vocalists']"
    """
    if not isinstance(tag_str, str):
        return []

    s = tag_str.strip()

    # Strip list-style brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    # Remove common quote characters
    for ch in ["'", '"']:
        s = s.replace(ch, "")

    # Split on commas, keep non-empty pieces
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def _split_tags(tag_str):
    """
    Split a comma-separated tag string into a list of tags.

    Example: "soul,rnb, trap" -> ["soul", "rnb", "trap"]
    """
    if pd.isna(tag_str):
        return []
    text = str(tag_str)
    parts = [t.strip() for t in text.split(",")]
    parts = [t for t in parts if t]  # drop empty strings
    return parts


def compute_song_popularity(df):
    """
    Build tag popularity from all songs in df, then assign each song
    a "popularity_score" equal to the average popularity of its tags.

    Returns a DataFrame with:
      - tag_list          : list of tags for the song
      - popularity_score  : float > 0

    Songs with no valid tags get score 0 and are dropped.
    """
    if "tags" not in df.columns:
        raise ValueError("Input DataFrame must have a 'tags' column.")

    # Turn the tags column into lists of tags
    df = df.copy()
    df["tag_list"] = df["tags"].fillna("").apply(_split_tags)

    # Count how often each tag appears
    counter = Counter()
    for tags in df["tag_list"]:
        counter.update(tags)

    if not counter:
        raise ValueError("No tags found when computing popularity.")

    tag_counts = pd.DataFrame(
        {
            "tag": list(counter.keys()),
            "count": list(counter.values()),
        }
    )
    tag_counts = tag_counts.sort_values("count", ascending=False).reset_index(drop=True)
    tag_counts["p_tag"] = tag_counts["count"] / tag_counts["count"].sum()

    # Map tag -> popularity
    p_tag_map = dict(zip(tag_counts["tag"], tag_counts["p_tag"]))

    # For each song, popularity = average p_tag over its tags
    def song_popularity(tags):
        if not tags:
            return 0.0
        vals = [p_tag_map.get(t, 0.0) for t in tags]
        if not any(vals):
            return 0.0
        return float(np.mean(vals))

    df["popularity_score"] = df["tag_list"].apply(song_popularity)

    # Keep only songs with positive popularity
    df = df[df["popularity_score"] > 0].copy()
    df.reset_index(drop=True, inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample songs from popular tags and score them."
    )
    parser.add_argument(
        "--csv",
        default="data/top_tracks_lastfm.csv",
        help="CSV with candidate tracks (must have a 'tags' column).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of songs to randomly sample.",
    )
    parser.add_argument(
        "--out",
        default="sampled_songs_with_scores.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    # 1) Load candidate songs
    print(f"Loading candidate songs from {args.csv} ...")
    df = pd.read_csv(args.csv)

    if "tags" not in df.columns:
        raise ValueError("Input CSV must have a 'tags' column.")

    print(f"Total songs in input: {df.shape[0]}")

    # 2) Compute popularity-based sampling weights
    print("Computing tag popularity and song popularity scores ...")
    df = compute_song_popularity(df)
    if df.empty:
        raise ValueError("No songs with non-zero popularity_score found.")

    print(f"Songs with positive popularity_score: {df.shape[0]}")
    weights = df["popularity_score"].values.astype(float)
    weights = weights / weights.sum()

    # 3) Randomly sample songs, biased toward popular tags
    n_samples = min(args.n_samples, df.shape[0])
    print(f"Sampling {n_samples} songs (without replacement) ...")

    sampled_idx = np.random.choice(
        df.index,
        size=n_samples,
        replace=False,
        p=weights,
    )
    sampled_df = df.loc[sampled_idx].copy().reset_index(drop=True)

    # 4) Load trained model + vectorizer
    print("Loading trained model and vectorizer ...")
    clf = joblib.load("artifacts/tag_model.joblib")
    vectorizer = joblib.load("artifacts/tag_vectorizer.joblib")

    # 5) Vectorize tags and predict probabilities P(like = 1 | tags)
    print("Scoring sampled songs with the model ...")
    X_sampled = vectorizer.transform(sampled_df["tags"].fillna(""))
    probs = clf.predict_proba(X_sampled)[:, 1]
    sampled_df["score"] = probs

    # Sort by score (highest first) for convenience
    sampled_df = sampled_df.sort_values("score", ascending=False).reset_index(drop=True)

    # 6) Save results
    sampled_df.to_csv(args.out, index=False)
    print(f"Saved {n_samples} sampled songs with scores to {args.out}")

    # Show a small preview
    cols_to_show = [
        c for c in ["track_name", "artist_name", "tags", "score"]
        if c in sampled_df.columns
    ]
    print("\nTop 5 sampled songs (by score):")
    print(sampled_df[cols_to_show].head())


if __name__ == "__main__":
    main()
