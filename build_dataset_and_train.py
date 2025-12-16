"""
CS158 Final Project: Tag-Based Music Recommender
File: build_dataset_and_train.py
Author: Hill Zhang
Date: 2025-12-02

AI Use:
I used an AI assistant for a few specific things on this script:
  - Proofreading and rephrasing comments and print messages after I first wrote
    rough bullet-style notes about each step (data loading, splitting, training,
    and saving models). The ideas and ordering came from class and my own notes;
    the AI mainly helped clean up the wording and make it more readable.
  - Double-checking some scikit-learn boilerplate. For example, I asked about the
    standard way to do a 60/20/20 train/dev/test split with stratification, and
    then adapted that pattern into my own code using train_test_split.
  - Debugging a couple of specific issues: one example was understanding why
    newer versions of scikit-learn use vectorizer.get_feature_names_out() instead
    of the older get_feature_names(). I asked the AI about the AttributeError,
    read the explanation, and then updated the code to use get_feature_names_out()
    in the section where I print out the top positive/negative tags.

The core idea of the model (logistic regression on bag-of-tags features, tuning C
on a dev set, and evaluating on a held-out test set) is directly from CS158 and
the course materials. I wrote the training pipeline, feature-building calls, and
evaluation logic myself, then used the AI to sanity-check the structure and to
polish the comments so the code is easier to read.

Summary:
This script trains a logistic regression model on Last.fm tag features for my
Spotify recommendation project. It:
  1. Checks that api_labeled_tracks.csv exists.
  2. Loads the labeled dataset (positives + negatives built in test_api.py).
  3. Builds a tag dictionary and tag counts.
  4. Builds a tag feature matrix from the "tags" column.
  5. Splits into train/dev/test and tunes C using F1 on the dev set.
  6. Trains a final model, prints top positive/negative tags, and saves:
       - tag_model.joblib
       - tag_vectorizer.joblib
       - labeled_dataset.csv
       - tag_features.csv

Usage:
  - Make sure api_labeled_tracks.csv is in the current folder
    (run test_api.py first to build it).
  - Then run:
        python build_dataset_and_train.py
"""

# Train a logistic regression model on tag features.
#
# Steps:
# 1. Check that api_labeled_tracks.csv exists
# 2. Load the labeled dataset (positives + negatives from test_api)
# 3. Build a tag dictionary + tag counts
# 4. Build the tag feature matrix
# 5. Train / dev / test split + logistic regression with C tuning
# 6. Save the model, vectorizer, and datasets

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

from preprocess import (
    load_labeled_dataset,
    build_tag_features,
    build_tag_dictionary_from_dfs,
)


def main():
    print("build_dataset_and_train.py: starting training script...")

    # ---------- STEP 1: sanity check CSV ----------
    print("\n=== Step 1: Check training CSV exists ===")
    user_csv = "api_labeled_tracks.csv"
    cwd = os.getcwd()
    print("Current working directory:", cwd)
    print("Files in folder:", os.listdir(cwd))
    print(f"Checking {user_csv} ... exists:", os.path.exists(user_csv))

    if not os.path.exists(user_csv):
        print("ERROR: api_labeled_tracks.csv not found. Run test_api.py first.")
        return

    # ---------- STEP 2: load + inspect labeled dataset ----------
    print("\n=== Step 2: Load labeled dataset ===")
    labeled_df = load_labeled_dataset(user_csv=user_csv)

    cols_to_show = [c for c in ["track_name", "artist_name", "label"] if c in labeled_df.columns]
    print("\nFirst few rows (track_name, artist_name, label):")
    print(labeled_df[cols_to_show].head())

    print("\nLabel counts in labeled dataset:")
    print(labeled_df["label"].value_counts())

    # ---------- STEP 3: build tag dictionary from songs ----------
    print("\n=== Step 3: Build tag dictionary from songs ===")
    tag_dict = build_tag_dictionary_from_dfs(
        [labeled_df],
        max_tags=2500,
        dict_path="tag_dictionary.csv",
        counts_path="tag_counts.csv",
    )
    print(f"Tag dictionary size: {len(tag_dict)} tags (saved to tag_dictionary.csv)")
    print("Tag counts saved to tag_counts.csv")

    # ---------- STEP 4: build tag feature matrix ----------
    print("\n=== Step 4: Build tag features from 'tags' column ===")
    features_df, vectorizer = build_tag_features(labeled_df, tag_dict=tag_dict)

    X = features_df.drop(columns=["label"]).values
    y = features_df["label"].values

    print("Feature matrix shape:", X.shape)
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Unique labels and counts:", dict(zip(unique_labels, counts)))

    if len(unique_labels) < 2:
        print(
            "\n[ERROR] Only one label present in the data "
            f"(labels found: {unique_labels}).\n"
            "LogisticRegression needs both positive (1) and negative (0) examples."
        )
        return

    # ---------- STEP 5: train/dev/test split + logistic regression ----------
    print("\n=== Step 5: Train/dev/test split & logistic regression with tuning ===")

    # First split off the test set (20% of the data, stratified by label)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Then split the remaining 80% into train (60%) and dev (20%)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,  # 0.25 of 80% = 20%, so 60/20/20 split overall
        random_state=42,
        stratify=y_temp,
    )

    print(
        f"Train size: {X_train.shape[0]}  "
        f"Dev size: {X_dev.shape[0]}  "
        f"Test size: {X_test.shape[0]}"
    )

    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_C = None
    best_f1 = -1.0

    print("\nTuning C on dev set using F1 score:")
    for C in C_values:
        clf_tmp = LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
        clf_tmp.fit(X_train, y_train)
        dev_preds = clf_tmp.predict(X_dev)
        f1 = f1_score(y_dev, dev_preds)
        print(f"  C={C}: dev F1 = {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_C = C

    print(f"\nBest C on dev: {best_C} (F1={best_f1:.3f})")

    # Retrain on train + dev combined with the best C
    clf = LogisticRegression(
        C=best_C,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )
    clf.fit(X_temp, y_temp)

    y_pred = clf.predict(X_test)
    print("\nClassification report on test set with tuned C:")
    print(classification_report(y_test, y_pred, digits=3))

    # ---------- STEP 6: inspect weights & save artifacts ----------
    print("\n=== Step 6: Inspect learned tag weights ===")
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]

    top_like_idx = coefs.argsort()[::-1][:15]
    print("\nTop 15 positive tags (associated with like=1):")
    for idx in top_like_idx:
        print(f"{feature_names[idx]:20s}  coef = {coefs[idx]: .4f}")

    top_dislike_idx = coefs.argsort()[:15]
    print("\nTop 15 negative tags (associated with dislike=0):")
    for idx in top_dislike_idx:
        print(f"{feature_names[idx]:20s}  coef = {coefs[idx]: .4f}")

    print("\n=== Step 7: Save artifacts ===")
    joblib.dump(clf, "tag_model.joblib")
    joblib.dump(vectorizer, "tag_vectorizer.joblib")
    print("Saved: tag_model.joblib, tag_vectorizer.joblib")

    labeled_df.to_csv("labeled_dataset.csv", index=False)
    features_df.to_csv("tag_features.csv", index=False)
    print("Saved: labeled_dataset.csv, tag_features.csv")

    print("\nbuild_dataset_and_train.py: done. Model + vectorizer are ready for use.")


if __name__ == "__main__":
    main()
