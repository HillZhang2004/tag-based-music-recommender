"""
CS158 Final Project: Tag-Based Music Recommender
File: interactive_demo.py
Author: Hill, Nathaniel, Shreyan
Date: 2025-12-02

AI Use:
This file is a small command-line playground for our tag-based model.
We wrote the basic idea ourselves: load the saved model and vectorizer, ask
the user for a comma-separated tag list, build features with
build_tag_features_for_unlabeled, and then print out a like probability.

We did use an AI assistant for a few concrete details:
  - We wanted a simple REPL-style loop that supports quitting with an empty
    line, 'quit', 'q', or 'exit', and also exits cleanly on Ctrl+C. We had a
    rough loop written, but asked the AI how to handle EOFError and
    KeyboardInterrupt in a neat way. The try/except pattern around input()
    and the friendly "Bye!" messages came out of that discussion.
  - In an early draft, we were a little unsure about indexing the output of
    predict_proba when there is exactly one example. The AI explained that
    clf.predict_proba(X)[0, 1] is the standard way to grab P(label=1) for the
    first (and only) row, which is what we use here.
  - After retraining our vectorizer from train_from_api_dataset.py, we hit a
    joblib error that said it could not find 'parse_tag_string' on __main__.
    We asked the AI why that happens and learned that the vectorizer pickle
    was referencing the custom tokenizer defined in the training script. The
    fix (which we implemented here) was to copy the same parse_tag_string
    definition into this file so unpickling can find it.
  - We also asked the AI to look over the printed instructions at the top of
    the script and help clean up the wording so that another student could
    quickly understand how to test different tag combinations.

The overall design (using build_tag_features_for_unlabeled, keeping everything
in a while True loop, and formatting the probability to three decimals) is
ours. The AI mainly helped us make the interactive loop more robust, fix that
joblib/parse_tag_string compatibility issue, and make the user prompts less
awkward.

Summary:
This interactive script:
  1. Loads artifacts/tag_model.joblib and artifacts/tag_vectorizer.joblib.
  2. Repeatedly prompts the user for a comma-separated tag list.
  3. Converts that tag string into features with build_tag_features_for_unlabeled.
  4. Prints the predicted like probability for each input until the user exits.

Usage:
  - Make sure artifacts/tag_model.joblib, artifacts/tag_vectorizer.joblib, and preprocess.py are
    in the same folder.
  - From the project directory, run:
        python interactive_demo.py
"""

import joblib
import pandas as pd

from preprocess import build_tag_features_for_unlabeled

MODEL_PATH = "artifacts/tag_model.joblib"
VECTORIZER_PATH = "artifacts/tag_vectorizer.joblib"


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


def main():
    print("Loading model and vectorizer...")
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("\nWelcome to Hill's tag-based music recommender!")
    print("Type tags like: 'soul,rnb,trap,bedroom pop' or 'pop,indie'.")
    print("Press Enter on an empty line or type 'quit' to exit.\n")

    while True:
        try:
            raw = input("Enter a comma-separated tag list: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        # Exit on empty line
        if raw.strip() == "":
            print("Empty line detected, exiting. Bye!")
            break

        # Exit on 'quit' / 'q' / 'exit'
        if raw.strip().lower() in ("quit", "q", "exit"):
            print("Bye!")
            break

        tags_str = raw.strip()

        # Build a tiny DataFrame with one row of tags
        df = pd.DataFrame({"tags": [tags_str]})

        # Use the existing vectorizer to make features
        X = build_tag_features_for_unlabeled(df, vectorizer)

        # Predict probability of like (label = 1)
        like_prob = clf.predict_proba(X)[0, 1]
        print(f"Predicted like probability: {like_prob:.3f}\n")


if __name__ == "__main__":
    main()
