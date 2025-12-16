"""
CS158 Final Project: Tag-Based Music Recommender
File: preprocess.py
Author: Hill, Nathaniel, Shreyan
Date: 2025-12-02

AI Use:
This file holds most of our shared helpers for tags and features. We drafted the
basic structure ourselves: a small function to split comma-separated tags, a
tag counter to build a dictionary, and a wrapper around CountVectorizer to turn
the "tags" column into a feature matrix. These ideas come from the CS158
lectures on bag-of-words, plus our notes from earlier homework.

We did use an AI assistant in a few concrete ways:
  - After writing a rough outline for what this module should contain
    (load_labeled_dataset, tag counting, tag dictionary builder, and feature
    builders), we asked the AI to sanity-check that our use of Counter plus
    CountVectorizer made sense and that we were not accidentally
    re-fitting the vectorizer when we only wanted to transform.
  - We originally wrote slightly messier versions of _split_tags and the tag
    counting loop. The AI suggested small cleanups (e.g., using fillna(""),
    stripping whitespace, and dropping empty tokens) without changing the
    underlying logic.
  - When we started building the smaller demo apps, we realized we needed a
    helper that takes an unlabeled DataFrame and an already-fitted vectorizer
    and just returns X. We knew the idea (call vectorizer.transform on the
    "tags" column), but we asked the AI to help us wrap that into a simple
    function build_tag_features_for_unlabeled so we could reuse it in app.py
    and interactive_demo.py.

We also asked the AI to proofread the comments and docstrings in this file
after we wrote them, mainly to make sure the descriptions matched what the
code is actually doing. The design — what functions exist and how they fit
into the training / scoring pipeline — is ours; the AI helped us tidy up
naming, comments, and a couple of edge cases around missing tags.

Summary:
This module provides:
  - load_labeled_dataset: read a labeled CSV with 'tags' and 'label'.
  - build_tag_counts_from_dfs: count tag frequencies across one or more DataFrames.
  - build_tag_dictionary_from_dfs / load_tag_dictionary: build/load the tag
    dictionary used as a vocabulary.
  - build_tag_features: turn labeled "tags" into a feature matrix + label column.
  - build_tag_features_for_unlabeled: turn unlabeled "tags" into a feature matrix.
  - parse_tag_string: a compatibility helper for older vectorizers that may
    reference preprocess.parse_tag_string during unpickling.

Usage:
  - Training scripts (build_dataset_and_train.py, train_from_api_dataset.py)
    call load_labeled_dataset, build_tag_dictionary_from_dfs, and build_tag_features.
  - The smaller demos (app.py, interactive_demo.py) call
    build_tag_features_for_unlabeled to transform new "tags" using an
    existing vectorizer.
"""

import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def parse_tag_string(tag_str: str) -> str:
    """
    Normalize a Last.fm-style tag string into a single space-separated string.

    Example:
      'pop, female vocalists, indie pop'
    becomes:
      'pop female_vocalists indie_pop'

    This exists mainly for backward compatibility: some older vectorizers
    may reference preprocess.parse_tag_string when they were originally
    pickled. The current training pipeline uses _split_tags/_tag_tokenizer
    directly, but we keep this function here so unpickling remains safe.
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


def _split_tags(tag_str):
    """
    Split a comma-separated tag string into a clean list of tags.

    Example:
      "soul,rnb, trap" -> ["soul", "rnb", "trap"]
    """
    if pd.isna(tag_str):
        return []
    text = str(tag_str)
    parts = [t.strip() for t in text.split(",")]
    parts = [t for t in parts if t]  # drop empty strings
    return parts


def load_labeled_dataset(user_csv="api_labeled_tracks.csv", global_csv=None):
    """
    Load the labeled dataset from user_csv.

    We assume user_csv already has:
      - a 'tags' column with comma-separated tags
      - a 'label' column with 0/1 values

    global_csv is accepted for backwards compatibility, but not used here.
    """
    df = pd.read_csv(user_csv)

    if "tags" not in df.columns:
        raise ValueError(f"Expected a 'tags' column in {user_csv}")
    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column in {user_csv}")

    df = df.dropna(subset=["label"]).reset_index(drop=True)
    return df


def build_tag_counts_from_dfs(df_list):
    """
    Given a list of DataFrames that each have a 'tags' column,
    count how often each tag appears across all of them.

    Returns a DataFrame with columns ['tag', 'count'], sorted by count desc.
    """
    counter = Counter()

    for df in df_list:
        if df is None:
            continue
        if "tags" not in df.columns:
            continue

        tags_series = df["tags"].fillna("")
        for tag_str in tags_series:
            tags = _split_tags(tag_str)
            counter.update(tags)

    tag_counts = pd.DataFrame(
        {
            "tag": list(counter.keys()),
            "count": list(counter.values()),
        }
    )

    tag_counts = tag_counts.sort_values("count", ascending=False)
    tag_counts = tag_counts.reset_index(drop=True)
    return tag_counts


def build_tag_dictionary_from_dfs(
    df_list,
    max_tags=2500,
    dict_path="tag_dictionary.csv",
    counts_path="tag_counts.csv",
):
    """
    Build a tag dictionary (up to max_tags tags) based on tag frequencies
    in the provided DataFrames.

    Saves:
      - counts_path: CSV with columns ['tag', 'count']
      - dict_path:   CSV with a single column 'tag'

    Returns:
      - A Python list of tag strings in the dictionary.
    """
    tag_counts = build_tag_counts_from_dfs(df_list)
    tag_counts.to_csv(counts_path, index=False)

    top_tags = tag_counts["tag"].head(max_tags).tolist()
    pd.Series(top_tags, name="tag").to_csv(dict_path, index=False)

    return top_tags


def load_tag_dictionary(dict_path="tag_dictionary.csv"):
    """
    Load tag dictionary from CSV and return it as a list of tag strings.
    """
    df = pd.read_csv(dict_path)
    if "tag" not in df.columns:
        raise ValueError(f"Expected a 'tag' column in {dict_path}")
    return df["tag"].tolist()


def _tag_tokenizer(text):
    """
    Tokenizer used by CountVectorizer.
    Expects a comma-separated tag string.
    """
    return _split_tags(text)


def build_tag_features(labeled_df, tag_dict=None):
    """
    Turn the 'tags' column in labeled_df into a bag-of-words feature matrix.

    If tag_dict is provided, we use it as a fixed vocabulary (for example,
    the top ~2500 tags by frequency). If tag_dict is None, we let
    CountVectorizer build its own vocabulary from the labeled data.

    Returns:
      - features_df: DataFrame with one column per tag feature + 'label'
      - vectorizer:  fitted CountVectorizer
    """
    if "tags" not in labeled_df.columns:
        raise ValueError("Expected a 'tags' column in labeled_df.")
    if "label" not in labeled_df.columns:
        raise ValueError("Expected a 'label' column in labeled_df.")

    tags_series = labeled_df["tags"].fillna("")

    if tag_dict is not None:
        vectorizer = CountVectorizer(
            tokenizer=_tag_tokenizer,
            lowercase=False,
            vocabulary=tag_dict,
            binary=True,
        )
        X = vectorizer.transform(tags_series)
    else:
        vectorizer = CountVectorizer(
            tokenizer=_tag_tokenizer,
            lowercase=False,
            binary=True,
        )
        X = vectorizer.fit_transform(tags_series)

    feature_names = vectorizer.get_feature_names_out()
    features_df = pd.DataFrame(X.toarray(), columns=feature_names)
    features_df["label"] = labeled_df["label"].values

    return features_df, vectorizer


def build_tag_features_for_unlabeled(df, vectorizer):
    """
    Turn the 'tags' column in an unlabeled DataFrame into a feature matrix
    using an already-fitted CountVectorizer.

    This is used by the smaller demo scripts (app.py, interactive_demo.py)
    when we want to score new tag strings with the existing model.

    Returns:
      - X: feature matrix suitable for clf.predict / clf.predict_proba
    """
    if "tags" not in df.columns:
        raise ValueError("Expected a 'tags' column in df.")

    tags_series = df["tags"].fillna("")
    X = vectorizer.transform(tags_series)
    return X
