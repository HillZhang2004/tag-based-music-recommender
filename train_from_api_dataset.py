"""
CS158 Final Project: Tag-Based Music Recommender
File: train_from_api_dataset.py
Author: Hill, Nathaniel, Shreyan
Date: 2025-12-02

AI Use:
We used an AI assistant for a few specific parts of this script:
  - After we drafted the basic training pipeline on paper (clean the API-labeled
    dataset, vectorize tags, do a train/dev/test split, tune C on the dev set,
    and refit on all data), we asked the AI to help proofread and tighten the
    comments and print messages so they read more clearly.
  - We asked about a standard pattern for using CountVectorizer with a custom
    tokenizer. In an earlier version, we were confused about how the tokenizer
    interacts with token_pattern and whether returning lists vs. strings was
    okay. The AI explained that passing tokenizer=parse_tag_string is allowed,
    and that token_pattern is ignored in that case, which matched what we
    wanted. We then wrote parse_tag_string ourselves to handle both plain
    comma-separated tags and list-style strings, and kept this current behavior.
  - We also used the AI to double-check the train/dev/test split logic. We knew
    we wanted a 60/20/20 split with stratification, but we initially wrote a few
    different combinations of test_size and stratify parameters. The AI helped
    confirm that:
        X_train, X_temp, y_train, y_temp = train_test_split(..., test_size=0.4)
        X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    gives us 60/20/20 overall when we stratify on y.
  - Finally, we asked for a quick sanity check on saving models with joblib.
    We already knew joblib.dump(model, "tag_model.joblib") from past labs; the
    AI mostly helped us make sure we were saving both the final LogisticRegression
    model and the fitted CountVectorizer so that downstream scripts could load
    them consistently.

The idea of training a logistic regression classifier on bag-of-tags features,
tuning C using F1 on a dev set, and then refitting on the full dataset is based
on the CS158 course material and our own notes. The AI did not design the
algorithm; it helped us debug small details (like how the split proportions
work with train_test_split) and polish the comments and print statements so the
code is easier to read.

Summary:
This script trains a logistic regression model on tags from api_labeled_tracks.csv
using a CountVectorizer with a custom tokenizer. It:
  1. Loads and cleans the API-labeled dataset (ensuring non-empty tags and labels).
  2. Builds a CountVectorizer over tags using parse_tag_string and creates a
     sparse feature matrix X and label vector y.
  3. Splits the data into train/dev/test (60/20/20) with stratification.
  4. Trains LogisticRegression models for several values of C and picks the best
     one based on F1 on the dev set.
  5. Evaluates the best model on the held-out test set.
  6. Refits the final model on the full dataset and saves:
       - tag_model.joblib
       - tag_vectorizer.joblib

Usage:
  - Make sure api_labeled_tracks.csv is in the current folder.
  - Then run:
        python train_from_api_dataset.py
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import joblib


def parse_tag_string(tag_str):
    """
    Turn the 'tags' column into a list of tag tokens.

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


def main():
    print("=== Step 1: load API-labeled dataset ===")
    df = pd.read_csv("api_labeled_tracks.csv")
    print(f"Raw shape: {df.shape}")

    # Keep only rows that have tags and labels
    df = df.copy()
    df["tags"] = df["tags"].fillna("").astype(str)
    df = df[df["tags"].str.strip() != ""]
    df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)

    print(f"After cleaning: {df.shape}")
    print("Label counts:")
    print(df["label"].value_counts())

    # Step 2: vectorize tags
    print("\n=== Step 2: build tag vectorizer and feature matrix ===")

    vectorizer = CountVectorizer(
        tokenizer=parse_tag_string,
        lowercase=False,
        binary=True,
    )

    X = vectorizer.fit_transform(df["tags"].tolist())
    y = df["label"].to_numpy()

    print(f"Feature matrix shape: {X.shape}")

    # Step 3: train / dev / test split
    print("\n=== Step 3: train/dev/test split ===")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train size: {X_train.shape[0]}")
    print(f"Dev size:   {X_dev.shape[0]}")
    print(f"Test size:  {X_test.shape[0]}")

    # Step 4: train logistic regression and tune C on the dev set
    print("\n=== Step 4: train logistic regression (no weights) ===")
    C_values = [0.01, 0.1, 1.0, 10.0]
    best_C = None
    best_f1 = -1.0
    best_model = None

    for C in C_values:
        clf = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear",
        )
        clf.fit(X_train, y_train)
        dev_pred = clf.predict(X_dev)
        f1 = f1_score(y_dev, dev_pred)
        print(f"C = {C:>4}: dev F1 = {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_C = C
            best_model = clf

    print(f"\nBest C on dev = {best_C} (F1 = {best_f1:.3f})")

    # Step 5: evaluate on test set
    print("\n=== Step 5: evaluation on held-out test set ===")
    test_pred = best_model.predict(X_test)
    print(classification_report(y_test, test_pred, digits=3))

    # Step 6: refit on full (train + dev + test) and save artifacts
    print("\n=== Step 6: refit on full data and save model/vectorizer ===")
    final_model = LogisticRegression(
        C=best_C,
        max_iter=1000,
        solver="liblinear",
    )
    final_model.fit(X, y)

    joblib.dump(final_model, "tag_model.joblib")
    joblib.dump(vectorizer, "tag_vectorizer.joblib")
    print("Saved tag_model.joblib and tag_vectorizer.joblib")


if __name__ == "__main__":
    main()
