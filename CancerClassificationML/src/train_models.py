"""
Stage 6: Model Training
Trains and tunes 4 classifiers on the Stage 5 selected features (SelectKBest k=200):
  1. SVM (RBF kernel)
  2. Random Forest
  3. XGBoost
  4. MLPClassifier (neural net)

Each is tuned via RandomizedSearchCV with stratified k-fold CV, scored on
macro-F1 (not accuracy) to keep the COAD minority class from being ignored.
The best model is selected on CV score, never on train-set performance.
"""

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier

PROCESSED_DIR = Path("outputs/processed")
MODELS_DIR = Path("outputs/models")
REPORTS_DIR = Path("outputs/reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ITER = 25  # RandomizedSearchCV budget per model

# ---------------------------------------------------------------------------
# 1. Load Stage 5 outputs
# ---------------------------------------------------------------------------
X_train = np.load(PROCESSED_DIR / "X_train_selected.npy")
X_test = np.load(PROCESSED_DIR / "X_test_selected.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

print(f"Loaded X_train {X_train.shape}, X_test {X_test.shape}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []
fitted_models = {}

# ---------------------------------------------------------------------------
# 2. SVM (RBF)
# ---------------------------------------------------------------------------
print("\n[1/4] Tuning SVM (RBF)...")
svm_param_dist = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "class_weight": [None, "balanced"],
}
svm_search = RandomizedSearchCV(
    SVC(kernel="rbf", random_state=RANDOM_STATE),
    param_distributions=svm_param_dist,
    n_iter=N_ITER, cv=cv, scoring="f1_macro",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
t0 = time.time()
svm_search.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"  Best params: {svm_search.best_params_}")
print(f"  Best CV macro-F1: {svm_search.best_score_:.4f}  ({elapsed:.1f}s)")
results.append(("SVM_RBF", svm_search.best_score_, svm_search.best_params_, elapsed))
fitted_models["SVM_RBF"] = svm_search.best_estimator_

# ---------------------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------------------
print("\n[2/4] Tuning Random Forest...")
rf_param_dist = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=rf_param_dist,
    n_iter=N_ITER, cv=cv, scoring="f1_macro",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
t0 = time.time()
rf_search.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"  Best params: {rf_search.best_params_}")
print(f"  Best CV macro-F1: {rf_search.best_score_:.4f}  ({elapsed:.1f}s)")
results.append(("RandomForest", rf_search.best_score_, rf_search.best_params_, elapsed))
fitted_models["RandomForest"] = rf_search.best_estimator_

# ---------------------------------------------------------------------------
# 4. XGBoost
# ---------------------------------------------------------------------------
print("\n[3/4] Tuning XGBoost...")
xgb_param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        tree_method="hist",
    ),
    param_distributions=xgb_param_dist,
    n_iter=N_ITER, cv=cv, scoring="f1_macro",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
t0 = time.time()
xgb_search.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"  Best params: {xgb_search.best_params_}")
print(f"  Best CV macro-F1: {xgb_search.best_score_:.4f}  ({elapsed:.1f}s)")
results.append(("XGBoost", xgb_search.best_score_, xgb_search.best_params_, elapsed))
fitted_models["XGBoost"] = xgb_search.best_estimator_

# ---------------------------------------------------------------------------
# 5. MLPClassifier (neural net)
# ---------------------------------------------------------------------------
print("\n[4/4] Tuning MLPClassifier...")
mlp_param_dist = {
    "hidden_layer_sizes": [(100,), (128, 64), (100, 50), (64, 32, 16)],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "learning_rate_init": [0.001, 0.005, 0.01],
    "activation": ["relu", "tanh"],
}
mlp_search = RandomizedSearchCV(
    MLPClassifier(
        random_state=RANDOM_STATE,
        max_iter=1000,
        early_stopping=True,
    ),
    param_distributions=mlp_param_dist,
    n_iter=N_ITER, cv=cv, scoring="f1_macro",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
t0 = time.time()
mlp_search.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"  Best params: {mlp_search.best_params_}")
print(f"  Best CV macro-F1: {mlp_search.best_score_:.4f}  ({elapsed:.1f}s)")
results.append(("MLPClassifier", mlp_search.best_score_, mlp_search.best_params_, elapsed))
fitted_models["MLPClassifier"] = mlp_search.best_estimator_

# ---------------------------------------------------------------------------
# 6. Compare all models, pick best on CV macro-F1
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results, columns=["model", "cv_macro_f1", "best_params", "tuning_time_sec"])
results_df = results_df.sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)

print("\n=== Model Comparison (sorted by CV macro-F1) ===")
print(results_df[["model", "cv_macro_f1", "tuning_time_sec"]].to_string(index=False))

results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["model"]
best_model = fitted_models[best_model_name]
print(f"\nBest model (by CV macro-F1): {best_model_name}")

# ---------------------------------------------------------------------------
# 7. Save all tuned models + the best one separately
# ---------------------------------------------------------------------------
for name, model in fitted_models.items():
    joblib.dump(model, MODELS_DIR / f"model_{name}.joblib")

joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
with open(MODELS_DIR / "best_model_name.txt", "w") as f:
    f.write(best_model_name)

print(f"\nSaved: outputs/models/model_<name>.joblib for all 4 models")
print(f"Saved: outputs/models/best_model.joblib (= {best_model_name})")
print(f"Saved: outputs/reports/model_comparison.csv")
