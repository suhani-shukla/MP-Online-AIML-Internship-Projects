"""
preprocess.py
Builds preprocessing pipeline (imputation, encoding, scaling)
and train/test split for the Adult Income dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Columns dropped: 'education' is redundant with 'education-num',
# 'fnlwgt' is a census sampling weight, not predictive of the individual.
DROP_COLS = ["education", "fnlwgt"]

NUMERIC_COLS = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

CATEGORICAL_COLS = [
    "workclass", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

TARGET_COL = "income"


def build_preprocessor() -> ColumnTransformer:
    """Creates the ColumnTransformer for numeric + categorical features."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS)
    ])

    return preprocessor


def prepare_features_target(df: pd.DataFrame):
    """Drops unused columns, separates features/target, encodes target."""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # '<=50K' -> 0, '>50K' -> 1 (alphabetical)

    return X, y_encoded, le


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    from data_loader import load_clean_data

    df = load_clean_data()
    X, y, label_encoder = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Classes:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    print("Preprocessing pipeline built successfully.")