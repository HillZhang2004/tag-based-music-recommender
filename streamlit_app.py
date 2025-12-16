"""
CS158 Final Project: Tag-Based Music Recommender
File: streamlit_app.py
Author: Hill, Nathaniel, Shreyan
Date: 2025-12-02

AI Use:
For this Streamlit app, Hill did not have prior experience with Streamlit or
building small web UIs in Python. After we talked as a group and agreed that
it was okay to use an AI assistant for UI boilerplate (but not for designing
the ML model itself), Hill took the lead on this part and used GPT mostly as
a tutorial and style checker:

- Hill first wrote down what the app should do: load the saved logistic
  regression model and vectorizer, load data/top_tracks_lastfm.csv, precompute
  like-probabilities, and then show a simple page with a slider for "how many
  recommendations to show" and a table of results.
- We then asked the AI how to structure a minimal Streamlit app and what the
  basic patterns are (st.title, st.write, st.slider, st.button, st.dataframe).
  Based on that explanation, we put together this layout and wiring.
- We also asked why caching is useful in Streamlit. The AI suggested using
  @st.cache_resource around the function that loads the joblib artifacts and
  precomputes scores, so we would not reload the model or recompute predictions
  on every button click. After that, we read the docs and kept this pattern.
- Finally, we used the AI to proofread the text shown in the app (the short
  explanation under the title) and to check that our comments/docstrings were
  clear and not misleading.

The idea of ranking candidate tracks by predict_proba and showing a top-N list
comes from our CS158 logistic regression work and our own notes about how we
wanted the demo to look. The AI helped Hill learn the basic Streamlit API and
clean up the UI wording, but the overall flow (load artifacts, precompute
scores, slider for N, button to reveal the table) is our design, and we all
agreed on this usage.

Summary:
This Streamlit app:
  1. Loads artifacts/tag_model.joblib, artifacts/tag_vectorizer.joblib, and data/top_tracks_lastfm.csv.
  2. Uses the saved vectorizer and model to compute like_prob for each track.
  3. Lets the user pick how many recommendations to display with a slider.
  4. On button click, shows the top-N tracks sorted by like_prob in a table.

Usage:
  - Make sure artifacts/tag_model.joblib, artifacts/tag_vectorizer.joblib, and data/top_tracks_lastfm.csv
    are in the same folder as this file.
  - From the project directory, run:
        streamlit run streamlit_app.py
"""

import joblib
import pandas as pd
import streamlit as st


def parse_tag_string(tag_str: str) -> str:
    """
    Normalize a Last.fm tag string like
    'pop, female vocalists, indie pop'
    into 'pop female_vocalists indie_pop'.

    Must exist so joblib can unpickle artifacts/tag_vectorizer.joblib,
    which references this function.
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


@st.cache_resource
def load_artifacts():
    """Load model, vectorizer, and candidate tracks; precompute scores."""
    model = joblib.load("artifacts/tag_model.joblib")
    vectorizer = joblib.load("artifacts/tag_vectorizer.joblib")

    df = pd.read_csv("data/top_tracks_lastfm.csv")
    df["tags"] = df["tags"].fillna("").astype(str)

    X = vectorizer.transform(df["tags"])
    df["like_prob"] = model.predict_proba(X)[:, 1]

    return model, vectorizer, df


def main():
    model, vectorizer, candidates = load_artifacts()

    st.title("Hill's Tag-Based Music Recommender")

    st.write(
        """
This app uses a logistic regression model trained on my own listening history.
It takes Last.fm-style tags (e.g. `pop,female vocalists,indie pop`) and
predicts how likely I am to like a track with those tags.

For this demo, we rank tracks from `data/top_tracks_lastfm.csv` by their
predicted probability of being a "like".
"""
    )

    topn = st.slider(
        "How many recommendations to show?",
        min_value=5,
        max_value=50,
        value=20,
    )

    if st.button("Recommend tracks"):
        ranked = (
            candidates.sort_values("like_prob", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            ranked.loc[: topn - 1, ["like_prob", "track_name", "artist_name", "tags"]],
            use_container_width=True,
        )
    else:
        st.info("Click the button above to see recommendations.")


if __name__ == "__main__":
    main()
