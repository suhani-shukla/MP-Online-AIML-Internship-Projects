"""
Stage 7: Evaluation
Evaluates all 4 tuned models (Stage 6) on the held-out TEST set (never seen
during training or CV tuning). Generates per-model classification reports,
confusion matrix plots, and a focused check on COAD (minority class) recall,
since high overall/macro scores can still hide poor performance on the
smallest class.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

PROCESSED_DIR = Path("outputs/processed")
MODELS_DIR = Path("outputs/models")
FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load test data + label encoder (for readable class names)
# ---------------------------------------------------------------------------
X_test = np.load(PROCESSED_DIR / "X_test_selected.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")
label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
class_names = label_encoder.classes_  # alphabetical: BRCA, COAD, KIRC, LUAD, PRAD

print(f"Loaded X_test {X_test.shape}, y_test {y_test.shape}")
print(f"Class names (encoded 0-4): {list(class_names)}")

# ---------------------------------------------------------------------------
# 2. Load all 4 tuned models
# ---------------------------------------------------------------------------
model_files = {
    "SVM_RBF": "model_SVM_RBF.joblib",
    "RandomForest": "model_RandomForest.joblib",
    "XGBoost": "model_XGBoost.joblib",
    "MLPClassifier": "model_MLPClassifier.joblib",
}
models = {name: joblib.load(MODELS_DIR / fname) for name, fname in model_files.items()}

# ---------------------------------------------------------------------------
# 3. Evaluate each model on the TEST set
# ---------------------------------------------------------------------------
summary_rows = []
coad_idx = list(class_names).index("COAD")

for name, model in models.items():
    print(f"\n{'='*60}\n{name} - Test Set Evaluation\n{'='*60}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Per-class recall specifically for COAD (minority class sanity check)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    coad_recall = recall_per_class[coad_idx]

    print(f"Test Accuracy:        {acc:.4f}")
    print(f"Test Macro-F1:        {f1_macro:.4f}")
    print(f"Test Weighted-F1:     {f1_weighted:.4f}")
    print(f"Test Macro-Precision: {precision_macro:.4f}")
    print(f"Test Macro-Recall:    {recall_macro:.4f}")
    print(f"COAD Recall:          {coad_recall:.4f}  <-- minority class check")

    report_text = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    print(f"\nClassification Report:\n{report_text}")

    # Save per-model text report
    with open(REPORTS_DIR / f"classification_report_{name}.txt", "w") as f:
        f.write(f"{name} - Test Set Classification Report\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test Macro-F1: {f1_macro:.4f}\n")
        f.write(f"Test Weighted-F1: {f1_weighted:.4f}\n")
        f.write(f"COAD Recall: {coad_recall:.4f}\n\n")
        f.write(report_text)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {name} (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()

    summary_rows.append({
        "model": name,
        "test_accuracy": acc,
        "test_macro_f1": f1_macro,
        "test_weighted_f1": f1_weighted,
        "test_macro_precision": precision_macro,
        "test_macro_recall": recall_macro,
        "coad_recall": coad_recall,
    })

# ---------------------------------------------------------------------------
# 4. Summary table across all models, sorted by test macro-F1
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame(summary_rows).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
print(f"\n{'='*60}\n=== Test Set Summary (sorted by macro-F1) ===\n{'='*60}")
print(summary_df.to_string(index=False))

summary_df.to_csv(REPORTS_DIR / "test_set_evaluation_summary.csv", index=False)

best_test_model = summary_df.iloc[0]["model"]
print(f"\nBest model on TEST set: {best_test_model}")

# Sanity check: does the CV-selected best model (MLPClassifier) still lead on test?
cv_best = "MLPClassifier"
if best_test_model != cv_best:
    print(f"\nNOTE: CV picked {cv_best} as best, but on the held-out test set "
          f"{best_test_model} scores higher. Worth investigating before "
          f"finalizing the model for Stage 8/9 (possible CV overfit on {cv_best}).")
else:
    print(f"\nCV-selected best model ({cv_best}) is confirmed as best on the "
          f"held-out test set too - no signs of CV overfitting.")

print(f"\nSaved: outputs/reports/classification_report_<model>.txt (x4)")
print(f"Saved: outputs/figures/confusion_matrix_<model>.png (x4)")
print(f"Saved: outputs/reports/test_set_evaluation_summary.csv")
