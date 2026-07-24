"""
data_loader.py

Responsible for loading raw data from CSV files.
This module does NOT clean or transform data — it only reads it.
Keeping "loading" separate from "processing" makes the code easier
to test and understand.
"""

import pandas as pd
import os

# Base path to the data folder (relative to project root)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_movies(filename: str = "movies.csv") -> pd.DataFrame:
    """
    Load the movies dataset.

    Expected columns: movieId, title, genres

    Args:
        filename: name of the CSV file inside the data/ folder

    Returns:
        A pandas DataFrame containing movie data.
    """
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find '{filename}' in the data/ folder. "
            f"Please download the MovieLens ml-latest-small dataset "
            f"and place it in: {DATA_DIR}"
        )

    movies_df = pd.read_csv(file_path)
    return movies_df


def load_ratings(filename: str = "ratings.csv") -> pd.DataFrame:
    """
    Load the ratings dataset.

    Expected columns: userId, movieId, rating, timestamp

    Args:
        filename: name of the CSV file inside the data/ folder

    Returns:
        A pandas DataFrame containing rating data.
    """
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find '{filename}' in the data/ folder. "
            f"Please download the MovieLens ml-latest-small dataset "
            f"and place it in: {DATA_DIR}"
        )

    ratings_df = pd.read_csv(file_path)
    return ratings_df


# Simple manual test — only runs when this file is executed directly,
# not when it's imported elsewhere.
if __name__ == "__main__":
    movies = load_movies()
    ratings = load_ratings()

    print("Movies dataset shape:", movies.shape)
    print(movies.head())

    print("\nRatings dataset shape:", ratings.shape)
    print(ratings.head())