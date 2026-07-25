# 🎬 Movie Recommendation System

A simple, beginner-friendly **content-based movie recommendation system**
built with Python, Scikit-learn, and Streamlit. Given a movie you like,
it recommends similar movies based on shared **genres**.

This project was built as an academic assignment to demonstrate a clean,
modular machine learning pipeline — from raw data to an interactive web app.

---

## 📌 Project Overview

- **Approach:** Content-based filtering (genres only)
- **No** deep learning, collaborative filtering, embeddings, or LLMs
- **No** database, authentication, or cloud deployment — runs entirely locally
- **Dataset:** [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)

### How it works

1. Each movie's genres (e.g. `Adventure|Animation|Children`) are cleaned
   into a space-separated string (`Adventure Animation Children`).
2. These genre strings are converted into numeric vectors using **TF-IDF**
   (Term Frequency–Inverse Document Frequency).
3. **Cosine similarity** is computed between every pair of movies based on
   these vectors — movies sharing more (and rarer) genres score higher.
4. The similarity matrix is precomputed once and saved with **pickle**, so
   the app can load it instantly instead of recomputing it every run.
5. When a user picks a movie in the Streamlit app, the app looks up the
   most similar movies and displays them, along with genre tags and
   average rating (where available).

---

## 🗂️ Project Structure

```text
MovieRecommendationSystem/
│
├── data/                           # Raw dataset (not committed to git)
│   ├── movies.csv
│   └── ratings.csv
│
├── notebooks/
│   └── Movie_Recommendation.ipynb  # EDA, preprocessing, TF-IDF, similarity testing
│
├── src/
│   ├── data_loader.py              # Loads movies.csv and ratings.csv
│   ├── preprocess.py               # Cleans genres, aggregates ratings
│   ├── recommender.py              # Builds TF-IDF + cosine similarity, saves/loads model
│   └── app.py                      # Streamlit user interface
│
├── models/                         # Generated similarity matrix (not committed to git)
│   ├── cosine_similarity.pkl
│   └── movies_data.pkl
│
├── screenshots/
│   ├── home.png
│   └── recommendations.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>/MovieRecommendationSystem
```

### 2. Install dependencies

This project shares a Python virtual environment with the rest of the
repo's ML projects. Activate your existing venv, then install:

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The dataset is **not included** in this repository. Download the
`ml-latest-small` dataset from the official MovieLens page:

🔗 https://grouplens.org/datasets/movielens/

Extract it, and place the following two files into the `data/` folder:

MovieRecommendationSystem/data/movies.csv
MovieRecommendationSystem/data/ratings.csv

### 4. Build the recommendation model

This generates the TF-IDF similarity matrix and saves it as `.pkl` files
inside `models/`:

```bash
cd src
python recommender.py
```

You should see confirmation messages once the model files are saved,
along with a sample recommendation printed to the console.

### 5. Run the Streamlit app

From the `MovieRecommendationSystem/` folder:

```bash
streamlit run src/app.py
```

The app will open automatically in your browser (typically at
`http://localhost:8501`).

---

## 📓 Exploring the Notebook

To see the step-by-step development process (EDA, preprocessing, TF-IDF,
similarity computation, and testing), open:

```bash
jupyter notebook notebooks/Movie_Recommendation.ipynb
```

---

## 🖥️ Usage

1. Select a movie from the searchable dropdown.
2. Choose how many recommendations you'd like (5–10).
3. Click **Recommend**.
4. View similar movies displayed as cards, each showing:
   - Movie title
   - Genre tags (if available)
   - Average rating and number of ratings (if available)

If no movie is selected, or a movie has no valid match, the app displays
a friendly message instead of crashing.

---

## 📸 Screenshots

| Home Page | Recommendations |
|---|---|
| ![Home](screenshots/home.png) | ![Recommendations](screenshots/recommendations.png) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data loading and manipulation |
| Scikit-learn | TF-IDF vectorization + cosine similarity |
| Streamlit | Interactive web UI |
| Pickle | Saving/loading the precomputed model |
| Jupyter | Exploratory data analysis notebook |

---

## 📄 File Descriptions

| File | Description |
|---|---|
| `src/data_loader.py` | Loads `movies.csv` and `ratings.csv` from the `data/` folder |
| `src/preprocess.py` | Cleans the genres column and aggregates rating statistics per movie |
| `src/recommender.py` | Builds the TF-IDF matrix, computes cosine similarity, saves/loads the model, and provides the `get_recommendations()` function |
| `src/app.py` | Streamlit UI — movie selector, Recommend button, and results display |
| `notebooks/Movie_Recommendation.ipynb` | Step-by-step exploration: EDA, preprocessing, TF-IDF, similarity, and testing |

---

## 🚫 Scope / Limitations

This is an intentionally simple academic project. It does **not** include:

- Deep learning or neural network–based recommendations
- Collaborative filtering
- User embeddings or LLM-based recommendations
- User authentication or accounts
- A database backend
- External APIs
- Cloud deployment or Docker

Ratings are used only to **display** average rating information alongside
recommendations — they do not influence the similarity computation itself,
which is based purely on genre overlap.

---

## 🙌 Acknowledgements

Dataset provided by [GroupLens Research](https://grouplens.org/datasets/movielens/)
(MovieLens `ml-latest-small`).