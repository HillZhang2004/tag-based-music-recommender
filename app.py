"""
CS158 Final Project: Tag-Based Music Recommender
File: app.py
Author: Hill(UI lead and engineer), Nathaniel, Shreyan
Date: 2025-12-02

AI Use:
This small app is a simpler playground on top of the main Streamlit demo.
After we had the main streamlit_app.py working, Hill wanted a quick version
where you can just type in a custom tag list and see one predicted probability.

- Hill already had the idea: reuse the trained tag model, take a text box for
  comma-separated tags, clean them, run them through our existing
  build_tag_features_for_unlabeled helper, and then call predict_proba.
- Hill asked an AI assistant mainly for help wiring up the minimal Streamlit
  pattern (st.text_input + st.button + st.write) and for a sanity check that
  the DataFrame/feature-building code was compatible with our existing
  tag_vectorizer and model.
- In an early draft, we were unsure about how to index the output of
  predict_proba for a single example. The AI reminded us that
  clf.predict_proba(X)[0, 1] is the standard way to grab P(y=1) for the first
  row, which matches what we use here.

The idea of the UI (single text input, button, and a single predicted
probability) and the feature pipeline (build_tag_features_for_unlabeled) are
ours; the AI mainly helped Hill stitch together the Streamlit calls and confirm
the correct shape/indexing for predict_proba.

Summary:
This app:
  1. Loads tag_model.joblib and tag_vectorizer.joblib.
  2. Lets the user type in a comma-separated tag string.
  3. Cleans the tags, builds features with build_tag_features_for_unlabeled,
     and prints the predicted like probability for that one input.

Usage:
  - Make sure tag_model.joblib, tag_vectorizer.joblib, and preprocess.py are in
    the same folder.
  - From the project directory, run:
        streamlit run app.py
"""

# app.py
import streamlit as st
import joblib
import pandas as pd
from preprocess import build_tag_features_for_unlabeled

clf = joblib.load("tag_model.joblib")
vectorizer = joblib.load("tag_vectorizer.joblib")

st.title("Hill's Tag-Based Music Recommender")

tags_input = st.text_input(
    "Enter tags (comma separated):",
    "soul,rnb,trap,bedroom pop",
)

if st.button("Predict like probability"):
    tags_str = ",".join(
        [t.strip().lower() for t in tags_input.split(",") if t.strip()]
    )
    df = pd.DataFrame({"tags": [tags_str]})
    X = build_tag_features_for_unlabeled(df, vectorizer)
    like_prob = float(clf.predict_proba(X)[0, 1])
    st.write(f"**Predicted like probability: {like_prob:.3f}**")
