"""
Stage 4: Preprocessing pipeline.
- Drop zero-variance genes (fit on train, applied to both)
- log2(x+1) transform
- Stratified 80/20 split
- StandardScaler (fit on train only)
- Label encoding
Saves processed arrays + fitted transformers for reuse in later stages.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "raw_data/data.csv"
LABELS_PATH = "raw_data/labels.csv"
OUT_DIR = "outputs/processed"
MODEL_DIR = "outputs/models"

import os
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH, index_col=0)
    labels = pd.read_csv(LABELS_PATH, index_col=0)
    merged = data.join(labels, how="inner")

    X = merged.drop(columns=["Class"])
    y_raw = merged["Class"]

    # 1. Stratified split BEFORE any fitting (avoid leakage)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Drop zero-variance genes — fit (identify) on TRAIN only
    train_var = X_train.var(axis=0)
    zero_var_cols = train_var[train_var == 0].index
    print(f"Dropping {len(zero_var_cols)} zero-variance genes (identified from train set)")

    X_train = X_train.drop(columns=zero_var_cols)
    X_test = X_test.drop(columns=zero_var_cols)

    # 3. log2(x+1) transform (deterministic, no fitting needed, safe on both)
    X_train_log = np.log2(X_train + 1)
    X_test_log = np.log2(X_test + 1)

    # 4. StandardScaler — fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)

    # 5. Label encoding — fit on train only (classes are same across both anyway)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    print(f"Label classes: {list(le.classes_)}")

    # Save everything needed for downstream stages
    np.save(f"{OUT_DIR}/X_train_scaled.npy", X_train_scaled)
    np.save(f"{OUT_DIR}/X_test_scaled.npy", X_test_scaled)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/y_test.npy", y_test)

    # Save gene names that survived (needed later for interpretability/frontend)
    pd.Series(X_train.columns).to_csv(f"{OUT_DIR}/surviving_genes.csv", index=False, header=["gene"])

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.joblib")
    joblib.dump(zero_var_cols, f"{MODEL_DIR}/zero_var_cols.joblib")

    print("\nSaved processed arrays to outputs/processed/")
    print("Saved scaler, label encoder, zero-variance column list to outputs/models/")
    print("Stage 4 complete ✅")

if __name__ == "__main__":
    main()
