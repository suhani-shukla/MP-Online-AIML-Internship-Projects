"""
predict.py
Loads the trained pipeline and exposes a simple function to predict
income class for new, raw (unprocessed) input data.
"""

import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "random_forest.pkl"

# Columns the pipeline expects, in the same shape as training data
# (after data_loader's column normalization, before 'education'/'fnlwgt' are dropped
#  by preprocess.prepare_features_target — but note: the saved pipeline's
#  ColumnTransformer only looks at NUMERIC_COLS/CATEGORICAL_COLS, so extra
#  columns like 'education' are harmless if present, and required ones must exist)
REQUIRED_COLUMNS = [
    "age", "workclass", "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country"
]

_artifacts = None  # lazy-loaded cache


def _load_artifacts():
    """Load model + label encoder once, reuse across calls."""
    global _artifacts
    if _artifacts is None:
        _artifacts = joblib.load(MODEL_PATH)
    return _artifacts


def _validate_input(data: dict):
    missing = [col for col in REQUIRED_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def predict_income(input_data: dict) -> dict:
    """
    Predict income class for a single person.

    Parameters
    ----------
    input_data : dict
        Raw feature values, e.g.:
        {
            "age": 39,
            "workclass": "Private",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }

    Returns
    -------
    dict with predicted label, probability of '>50K', and probability of '<=50K'
    """
    _validate_input(input_data)
    artifacts = _load_artifacts()
    pipeline = artifacts["pipeline"]
    label_encoder = artifacts["label_encoder"]

    X = pd.DataFrame([input_data])

    pred_class = pipeline.predict(X)[0]
    pred_proba = pipeline.predict_proba(X)[0]

    label = label_encoder.inverse_transform([pred_class])[0]

    return {
        "predicted_income": label,
        "probability_gt_50k": float(pred_proba[list(label_encoder.classes_).index(">50K")]),
        "probability_le_50k": float(pred_proba[list(label_encoder.classes_).index("<=50K")]),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict income class for multiple rows at once.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the same raw columns as REQUIRED_COLUMNS.

    Returns
    -------
    pd.DataFrame with added 'predicted_income' and 'probability_gt_50k' columns.
    """
    artifacts = _load_artifacts()
    pipeline = artifacts["pipeline"]
    label_encoder = artifacts["label_encoder"]

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    preds = pipeline.predict(df)
    probas = pipeline.predict_proba(df)

    result = df.copy()
    result["predicted_income"] = label_encoder.inverse_transform(preds)
    result["probability_gt_50k"] = probas[:, list(label_encoder.classes_).index(">50K")]

    return result


if __name__ == "__main__":
    sample = {
        "age": 39,
        "workclass": "Private",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    result = predict_income(sample)
    print(result)