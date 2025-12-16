"""
CS158 Final Project: Tag-Based Music Recommender
File: test_api.py
Author: Hill, Nathaniel, Shreyan
Date: 2025-12-02

AI Use:
For this script, we mainly used an AI assistant as a helper while we were wiring
up the Spotify and Last.fm APIs:

- We first sketched the whole pipeline ourselves: pull our own top and saved
  tracks from Spotify, count “positive” tags, expand with Last.fm tag.getTopTracks,
  and then build a negative set that does not share tags with the positive pool.
  After that, we asked the AI to look over our rough code and help clean up some
  of the comments and print messages so they were clearer.
- When we initially called the Last.fm API, the script sometimes hung or failed
  with vague errors. We asked the AI how to add a simple wrapper around
  requests.get with a timeout and basic error handling. The idea of using
  timeout=10 and catching exceptions came from that debugging step, but we
  wrote the lastfm_get function around it ourselves and adapted the error
  messages to match our own style.
- We also double-checked some edge-case handling. For example, in building
  negative songs we wanted to skip tracks that share any positive tag or are
  already in the positive (name, artist) set. We had this logic in pseudocode
  and in a first draft; the AI mostly confirmed that the filtering order made
  sense and helped us phrase comments so another student could follow the reasoning.

Overall, the design of this script (positive track collection, tag counting,
expanding tags with Last.fm, and constructing negatives that avoid the positive
tag set) came from our own brainstorming and discussions about positive/
negative examples. Actually, we struggled the negative lavels the most,
discussed and brainsotmred so many times for various ideas because 
that is the most tricky part.
The AI did not design the strategy; it helped us debug API
issues, add a safe wrapper around requests, and polish the documentation.

Summary:
This script:
  1. Uses the Spotify API to pull our top tracks and saved tracks and treats
     them as “positive” songs, with a higher weight for top tracks.
  2. Calls the Last.fm API to fetch tags for each positive track and counts how
     often each tag appears.
  3. Expands the tag dictionary by querying Last.fm for top tracks under those
     tags and collecting more tags into an expanded_tag_counts Counter.
  4. Builds a pool of “negative” songs: tracks from tags we do not use
     positively, that have tags, and that do not share any tags with our
     positive set or overlap in (name, artist).
  5. Combines positives and negatives into a single DataFrame with columns
     [track_id, track_name, artist_name, tags, label, weight] and saves
     api_labeled_tracks.csv, which is used by our training scripts.

Usage (high level):
  - Make sure your .env file has valid Spotify and Last.fm API keys.
  - Activate the virtual environment.
  - Run:
        python test_api.py
  - The script will talk to both APIs and write api_labeled_tracks.csv in
    the current folder.
"""

import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
from collections import Counter
import pandas as pd

load_dotenv()

# ---------- Spotify setup ----------
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        scope=os.getenv("SPOTIPY_SCOPE", "user-top-read,user-library-read"),
        cache_path=".cache",
        open_browser=True,
    )
)

# user_positive: key = Spotify track id, value = [name, artist, weight]
# weight = 2 for "top" tracks, 1 for saved-only tracks
user_positive = {}

# --- top tracks: label weight = 2 ---
offset = 0
for _ in range(2):
    current_top = sp.current_user_top_tracks(
        limit=50,
        time_range="medium_term",
        offset=offset,
    )
    items = current_top.get("items", [])
    if not items:
        break
    for track in items:
        tid = track["id"]
        name = track["name"]
        artist = track["artists"][0]["name"]
        if tid not in user_positive:
            user_positive[tid] = [name, artist, 2]
    offset += 50

# --- saved tracks: label weight = 1 if not already 2 ---
offset = 0
while True:
    print("Fetching saved tracks, offset =", offset)
    current_saved = sp.current_user_saved_tracks(limit=50, offset=offset)
    items = current_saved.get("items", [])
    if not items:
        break
    for item in items:
        track = item["track"]
        tid = track["id"]
        name = track["name"]
        artist = track["artists"][0]["name"]
        if tid not in user_positive:
            user_positive[tid] = [name, artist, 1]
    offset += 50

print(f"Total positive tracks (top + saved): {len(user_positive)}")

# ---------- Last.fm helpers ----------
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_ROOT = "https://ws.audioscrobbler.com/2.0/"


def lastfm_get(params: dict) -> dict:
    """
    Wrapper around requests.get for Last.fm with timeout and basic error handling.
    Returns {} on any error so callers can safely .get(...) on the result.
    """
    full_params = {"api_key": LASTFM_API_KEY, "format": "json", **params}
    try:
        # Use a timeout so the script does not hang forever on a bad request.
        resp = requests.get(LASTFM_ROOT, params=full_params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            # Last.fm-level error (e.g., bad API key, unknown method)
            raise RuntimeError(f"Last.fm error {data['error']}: {data['message']}")
        return data
    except Exception as e:
        print(f"[lastfm_get] Error for params={params}: {e}")
        return {}


def get_tags_for_track(name: str, artist: str):
    """
    Fetch tags for a track + artist from Last.fm.
    Returns a list of tag strings.
    """
    ttags = []
    atags = []

    # track-level tags
    info = lastfm_get(
        {
            "method": "track.getInfo",
            "track": name,
            "artist": artist,
            "autocorrect": 1,
        }
    )
    if info:
        ttags = [
            t["name"]
            for t in info.get("track", {})
            .get("toptags", {})
            .get("tag", [])
        ]

    # artist-level tags
    ainfo = lastfm_get(
        {
            "method": "artist.getTopTags",
            "artist": artist,
            "autocorrect": 1,
        }
    )
    if ainfo:
        atags = [
            t["name"]
            for t in ainfo.get("toptags", {})
            .get("tag", [])
        ]

    # optional: dedupe while preserving order
    all_tags = ttags + atags
    seen = set()
    deduped = []
    for tag in all_tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


# ---------- Fetch tags for ALL user's positive tracks ----------
user_tag_counts = Counter()
track_tags = {}

print("Fetching tags for all positive tracks (top + saved)...")
for tid, (name, artist, weight) in user_positive.items():
    print(f"Fetching tags for track id={tid}, name={name}, artist={artist}")
    all_tags = get_tags_for_track(name, artist)
    track_tags[tid] = all_tags

    for tag in all_tags:
        user_tag_counts[tag] += weight

print("Number of positive tracks with tags:", len(track_tags))
print("Top user tags:", user_tag_counts.most_common(20))


def calculate_tag_prob(tag_counts: Counter):
    total = float(sum(tag_counts.values()))
    tag_prob = {}
    if total == 0:
        return tag_prob
    for tag, count in tag_counts.items():
        tag_prob[tag] = count / total
    return tag_prob


print("Tag probabilities (first 10):")
tag_prob = calculate_tag_prob(user_tag_counts)
for tag in list(tag_prob.keys())[:10]:
    print(tag, "->", tag_prob[tag])

# ---------- Expand tag dictionary starting from top user tags ----------
seed_tags = list(user_tag_counts.keys())
expanded_tag_counts = Counter(user_tag_counts)
processed_tracks = set(track_tags.keys())

max_tracks_per_tag = 10

for tag in seed_tags:
    print(f"Fetching top tracks for seed tag: {tag}")
    resp = lastfm_get(
        {
            "method": "tag.getTopTracks",
            "tag": tag,
            "limit": max_tracks_per_tag,
        }
    )
    tracks = resp.get("tracks", {}).get("track", []) if resp else []

    for track in tracks:
        tid = track.get("mbid") or (
            track.get("name", "") + track.get("artist", {}).get("name", "")
        )
        if tid in processed_tracks:
            continue

        processed_tracks.add(tid)
        name = track.get("name", "")
        artist = track.get("artist", {}).get("name", "")

        tags_for_track = get_tags_for_track(name, artist)
        for t in tags_for_track:
            expanded_tag_counts[t] += 1

        # keep a rough cap on size
        if len(expanded_tag_counts) >= 1500:
            break

    if len(expanded_tag_counts) >= 1500:
        break

print(f"Expanded dictionary size: {len(expanded_tag_counts)}")
print("Top expanded tags:", expanded_tag_counts.most_common(20))

tag_dict = set(expanded_tag_counts.keys())
user_tags = set(user_tag_counts.keys())

not_in_user = tag_dict - user_tags
print(f"Number of tags NOT in positive tag set: {len(not_in_user)}")

# ---------- Build negative songs ----------
negative_songs = []
max_songs = 200

positive_tag_set = set(user_tags)
positive_pairs = {
    (name, artist)
    for (_, (name, artist, _)) in user_positive.items()
}

for tag in not_in_user:
    if len(negative_songs) >= max_songs:
        break

    print(f"Fetching candidate negatives for tag: {tag}")
    resp = lastfm_get(
        {
            "method": "tag.getTopTracks",
            "tag": tag,
            "limit": 50,
        }
    )
    tracks = resp.get("tracks", {}).get("track", []) if resp else []

    for track in tracks:
        if len(negative_songs) >= max_songs:
            break

        tid = track.get("mbid") or (
            track.get("name", "") + track.get("artist", {}).get("name", "")
        )

        name = track.get("name", "")
        artist = track.get("artist", {}).get("name", "")

        # skip if same (name, artist) as a positive song
        if (name, artist) in positive_pairs:
            continue

        neg_tags = get_tags_for_track(name, artist)
        if not neg_tags:
            continue

        # eliminate negative candidates that share ANY positive tag
        if any(t in positive_tag_set for t in neg_tags):
            continue

        negative_songs.append(
            {
                "tid": tid,
                "name": name,
                "artist": artist,
                "tags": neg_tags,
                "label": 0,
            }
        )

print(f"Sampled {len(negative_songs)} negative tracks from unliked tags")
print("Sample negative tracks:", negative_songs[:3])

# ---------- Build training DataFrame and save ----------
rows = []

# positives (label = 1, weight = 1 or 2 depending on top vs saved)
for tid, (name, artist, weight) in user_positive.items():
    tags = track_tags.get(tid, [])
    rows.append(
        {
            "track_id": tid,
            "track_name": name,
            "artist_name": artist,
            "tags": ",".join(tags),
            "label": 1,
            "weight": weight,
        }
    )

# negatives (label = 0, weight = 1)
for song in negative_songs:
    rows.append(
        {
            "track_id": song["tid"],
            "track_name": song["name"],
            "artist_name": song["artist"],
            "tags": ",".join(song["tags"]),
            "label": 0,
            "weight": 1,
        }
    )

df = pd.DataFrame(rows)
df.to_csv("api_labeled_tracks.csv", index=False)

print("Saved training dataset to api_labeled_tracks.csv")
print("Shape:", df.shape)
print("Label counts:")
print(df["label"].value_counts())
