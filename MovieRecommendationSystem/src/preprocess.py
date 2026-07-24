"""
preprocess.py

Responsible for cleaning and preparing data before it is used
for building the recommendation model.

Two main jobs:
1. Clean the 'genres' column so it can be fed into TF-IDF.
2. Aggregate ratings (average rating + rating count) and merge
   them into the movies DataFrame for display purposes only.
"""

import pandas as pd


def clean_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the genres column so it's ready for TF-IDF vectorization.

    MovieLens genres look like: "Adventure|Animation|Children"
    TF-IDF treats text as space-separated words, so we replace
    the '|' separator with spaces: "Adventure Animation Children"

    Also handles the special case "(no genres listed)" by replacing
    it with an empty string, so it doesn't get treated as a real genre.

    Args:
        movies_df: raw movies DataFrame with a 'genres' column

    Returns:
        A copy of movies_df with a new column 'genres_clean'
    """
    df = movies_df.copy()

    # Replace the "no genres listed" placeholder with an empty string
    df["genres"] = df["genres"].replace("(no genres listed)", "")

    # Replace '|' with a space so each genre acts like a separate "word"
    df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)

    return df


def aggregate_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average rating and number of ratings per movie.

    Args:
        ratings_df: raw ratings DataFrame with columns
                    userId, movieId, rating, timestamp

    Returns:
        A DataFrame with columns: movieId, avg_rating, rating_count
    """
    rating_stats = (
        ratings_df.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )

    # Round average rating to 1 decimal place for cleaner display
    rating_stats["avg_rating"] = rating_stats["avg_rating"].round(1)

    return rating_stats


def merge_movies_with_ratings(
    movies_df: pd.DataFrame, rating_stats_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge aggregated rating stats into the movies DataFrame.

    Movies with no ratings will have NaN for avg_rating and
    rating_count — these are filled with sensible defaults so
    the UI doesn't break later.

    Args:
        movies_df: DataFrame containing at least 'movieId'
        rating_stats_df: output of aggregate_ratings()

    Returns:
        Merged DataFrame with avg_rating and rating_count columns added.
    """
    merged_df = movies_df.merge(rating_stats_df, on="movieId", how="left")

    # Fill missing ratings info (movies with zero ratings)
    merged_df["avg_rating"] = merged_df["avg_rating"].fillna(0.0)
    merged_df["rating_count"] = merged_df["rating_count"].fillna(0).astype(int)

    return merged_df


def prepare_data(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: clean genres + merge rating stats.

    This is the single function other modules (notebook, recommender)
    should call to get a ready-to-use movies DataFrame.

    Args:
        movies_df: raw movies DataFrame
        ratings_df: raw ratings DataFrame

    Returns:
        A fully prepared DataFrame ready for TF-IDF and display.
    """
    movies_clean = clean_genres(movies_df)
    rating_stats = aggregate_ratings(ratings_df)
    final_df = merge_movies_with_ratings(movies_clean, rating_stats)

    return final_df


# Simple manual test — only runs when this file is executed directly
if __name__ == "__main__":
    from data_loader import load_movies, load_ratings

    movies = load_movies()
    ratings = load_ratings()

    prepared = prepare_data(movies, ratings)

    print("Prepared data shape:", prepared.shape)
    print(prepared[["title", "genres_clean", "avg_rating", "rating_count"]].head())