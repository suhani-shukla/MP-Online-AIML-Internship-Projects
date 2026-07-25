"""
train.py
Trains baseline (Logistic Regression) and Random Forest models
on the Adult Income dataset, then saves the best pipeline to disk.
"""

import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from data_loader import load_clean_data
from preprocess import build_preprocessor, prepare_features_target, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_pipeline(model):
    """Wraps a classifier with the shared preprocessing step."""
    preprocessor = build_preprocessor()
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def train_and_evaluate(name, model, X_train, y_train, cv=5):
    """Trains a pipeline and reports cross-validated F1 score."""
    pipeline = build_pipeline(model)

    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
    )
    print(f"{name}: CV F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    pipeline.fit(X_train, y_train)
    return pipeline, scores.mean()


def main():
    # 1. Load and prepare data
    df = load_clean_data()
    X, y, label_encoder = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results = {}

    # 2. Baseline: Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg_pipeline, log_reg_score = train_and_evaluate(
        "Logistic Regression", log_reg, X_train, y_train
    )
    results["logistic_regression"] = (log_reg_pipeline, log_reg_score)

    # 3. Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_pipeline, rf_score = train_and_evaluate(
        "Random Forest", rf, X_train, y_train
    )
    results["random_forest"] = (rf_pipeline, rf_score)

    # 4. Pick best model by CV F1 score
    best_name, (best_pipeline, best_score) = max(
        results.items(), key=lambda item: item[1][1]
    )
    print(f"\nBest model: {best_name} (CV F1 = {best_score:.4f})")

    # 5. Save best pipeline + label encoder together
    model_path = MODEL_DIR / "random_forest.pkl"
    joblib.dump({
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "model_name": best_name
    }, model_path)
    print(f"Saved best model to {model_path}")

    # Also stash the test split for evaluate.py to reuse
    joblib.dump((X_test, y_test), MODEL_DIR / "test_split.pkl")


if __name__ == "__main__":
    main()