"""
app.py

Streamlit user interface for the Movie Recommendation System.

Responsibilities:
- Load the precomputed similarity model.
- Let the user pick a movie from a dropdown.
- Show top-N recommended movies when the user clicks "Recommend".

This file contains NO machine learning logic itself — all of that
lives in recommender.py. This file only handles presentation and
user interaction.
"""

import streamlit as st
from recommender import load_model, get_recommendations


# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
)


# ---------- Load Model (cached so it only loads once per session) ----------
@st.cache_resource
def get_cached_model():
    """
    Load the cosine similarity matrix and movies data once,
    and reuse it across user interactions instead of reloading
    from disk every time.
    """
    cosine_sim, movies_prepared = load_model()
    return cosine_sim, movies_prepared


# ---------- Header ----------
st.title("🎬 Movie Recommendation System")
st.write(
    "A simple **content-based** movie recommender built using genre "
    "similarity. Pick a movie you like, and get similar movies "
    "recommended based on shared genres."
)
st.divider()


# ---------- Load model with error handling ----------
try:
    cosine_sim, movies_prepared = get_cached_model()
except FileNotFoundError as e:
    st.error(
        "⚠️ Model files not found. Please run `python src/recommender.py` "
        "first to generate the recommendation model."
    )
    st.stop()


# ---------- Movie Selector ----------
movie_titles = sorted(movies_prepared["title"].dropna().unique())

selected_movie = st.selectbox(
    "Select a movie you like:",
    options=movie_titles,
    index=None,
    placeholder="Start typing or select a movie...",
)

top_n = st.slider("Number of recommendations:", min_value=5, max_value=10, value=5)

recommend_clicked = st.button("Recommend", type="primary")


# ---------- Handle Recommendation Logic ----------
st.divider()

if recommend_clicked:
    if not selected_movie:
        # Graceful handling of missing selection
        st.warning("⚠️ Please select a movie before clicking Recommend.")
    else:
        results = get_recommendations(
            title=selected_movie,
            cosine_sim=cosine_sim,
            movies_prepared=movies_prepared,
            top_n=top_n,
        )

        if results is None or results.empty:
            # Graceful handling of an unknown/invalid title
            st.error(f"Sorry, no recommendations could be found for '{selected_movie}'.")
        else:
            st.subheader(f"Movies similar to '{selected_movie}':")

            for _, row in results.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**")

                    # Show genre tags if available
                    genres = row["genres_clean"].strip()
                    if genres:
                        genre_tags = " ".join(f"`{g}`" for g in genres.split(" "))
                        st.markdown(genre_tags)
                    else:
                        st.caption("No genre information available.")

                    # Show rating info, with graceful fallback
                    if row["rating_count"] > 0:
                        st.caption(
                            f"⭐ {row['avg_rating']} average "
                            f"({int(row['rating_count'])} ratings)"
                        )
                    else:
                        st.caption("No ratings yet.")


# ---------- Footer ----------
st.divider()
st.caption(
    "Built with Python, Scikit-learn, and Streamlit · "
    "Content-based filtering using movie genres (MovieLens dataset)"
)