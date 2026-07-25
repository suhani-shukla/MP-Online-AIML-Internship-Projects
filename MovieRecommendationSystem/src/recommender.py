"""
recommender.py

Core recommendation logic for the Movie Recommendation System.

Responsibilities:
1. Build a TF-IDF matrix from cleaned genres.
2. Compute the cosine similarity matrix between all movies.
3. Save/load the similarity matrix (and supporting data) using pickle,
   so the Streamlit app doesn't need to recompute everything on every run.
4. Provide a simple get_recommendations() function used by the UI.
"""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_movies, load_ratings
from preprocess import prepare_data

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SIMILARITY_PATH = os.path.join(MODELS_DIR, "cosine_similarity.pkl")
MOVIES_DATA_PATH = os.path.join(MODELS_DIR, "movies_data.pkl")


def build_similarity_matrix(movies_prepared: pd.DataFrame):
    """
    Build the TF-IDF matrix from genres, then compute cosine similarity.

    Args:
        movies_prepared: DataFrame that already has a 'genres_clean' column
                          (output of preprocess.prepare_data)

    Returns:
        A 2D numpy array (cosine similarity matrix), where cell [i][j]
        is the similarity between movie i and movie j.
    """
    # token_pattern ensures genre names like "Sci-Fi" are treated as one token
    tfidf = TfidfVectorizer(token_pattern=r"[a-zA-Z\-]+")
    tfidf_matrix = tfidf.fit_transform(movies_prepared["genres_clean"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim


def save_model(cosine_sim, movies_prepared: pd.DataFrame) -> None:
    """
    Save the cosine similarity matrix and the prepared movies DataFrame
    to disk using pickle, so they can be reloaded instantly later.

    Args:
        cosine_sim: the cosine similarity matrix (numpy array)
        movies_prepared: the DataFrame used to build it (needed later
                          to map titles to matrix indices, and to
                          display genres/ratings in the UI)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(SIMILARITY_PATH, "wb") as f:
        pickle.dump(cosine_sim, f)

    with open(MOVIES_DATA_PATH, "wb") as f:
        pickle.dump(movies_prepared, f)

    print(f"Saved similarity matrix to: {SIMILARITY_PATH}")
    print(f"Saved movies data to: {MOVIES_DATA_PATH}")


def load_model():
    """
    Load the previously saved cosine similarity matrix and movies DataFrame.

    Returns:
        A tuple: (cosine_sim, movies_prepared)

    Raises:
        FileNotFoundError: if the model hasn't been built/saved yet.
    """
    if not os.path.exists(SIMILARITY_PATH) or not os.path.exists(MOVIES_DATA_PATH):
        raise FileNotFoundError(
            "Model files not found. Please run this script directly first "
            "(python src/recommender.py) to generate them, or run the "
            "notebook's model-building steps."
        )

    with open(SIMILARITY_PATH, "rb") as f:
        cosine_sim = pickle.load(f)

    with open(MOVIES_DATA_PATH, "rb") as f:
        movies_prepared = pickle.load(f)

    return cosine_sim, movies_prepared


def get_recommendations(title: str, cosine_sim, movies_prepared: pd.DataFrame, top_n: int = 10):
    """
    Return the top-N movies most similar in genre to the given title.

    Args:
        title: the exact movie title to find recommendations for
        cosine_sim: precomputed cosine similarity matrix
        movies_prepared: DataFrame aligned with cosine_sim's row/column order
        top_n: how many recommendations to return (default 10)

    Returns:
        A DataFrame with columns [title, genres_clean, avg_rating, rating_count]
        for the top_n most similar movies, OR None if the title isn't found.
    """
    # Build a title -> row index lookup
    indices = pd.Series(movies_prepared.index, index=movies_prepared["title"]).drop_duplicates()

    if title not in indices:
        return None

    idx = indices[title]

    # Pair up every movie's index with its similarity score to the selected movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity score, highest first
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip index 0 — that's the movie itself (perfect similarity with itself)
    sim_scores = sim_scores[1: top_n + 1]

    recommended_indices = [i[0] for i in sim_scores]

    return movies_prepared.loc[
        recommended_indices, ["title", "genres_clean", "avg_rating", "rating_count"]
    ].reset_index(drop=True)


def build_and_save_model():
    """
    Full pipeline: load raw data -> preprocess -> build similarity matrix
    -> save to disk. Run this once (or whenever the dataset changes).
    """
    movies_df = load_movies()
    ratings_df = load_ratings()

    movies_prepared = prepare_data(movies_df, ratings_df)
    cosine_sim = build_similarity_matrix(movies_prepared)

    save_model(cosine_sim, movies_prepared)


# Running this file directly builds and saves the model.
if __name__ == "__main__":
    build_and_save_model()

    # Quick manual test after building
    cosine_sim, movies_prepared = load_model()
    results = get_recommendations("Toy Story (1995)", cosine_sim, movies_prepared, top_n=5)

    print("\nSample recommendations for 'Toy Story (1995)':")
    print(results)