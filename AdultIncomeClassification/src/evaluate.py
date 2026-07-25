"""
evaluate.py
Loads the trained pipeline and held-out test set, then reports
classification metrics and saves diagnostic plots to screenshots/.
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)


def load_artifacts():
    saved = joblib.load(MODEL_DIR / "random_forest.pkl")
    X_test, y_test = joblib.load(MODEL_DIR / "test_split.pkl")
    return saved["pipeline"], saved["label_encoder"], saved["model_name"], X_test, y_test


def print_metrics(y_test, y_pred, y_proba, label_encoder):
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")


def plot_confusion_matrix(y_test, y_pred, label_encoder):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {SCREENSHOT_DIR / 'confusion_matrix.png'}")


def plot_roc_curve(pipeline, X_test, y_test):
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve to {SCREENSHOT_DIR / 'roc_curve.png'}")


def plot_feature_importance(pipeline, top_n=15):
    """Only works for tree-based models with feature_importances_."""
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        print("Model has no feature_importances_ attribute — skipping plot.")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importances[idx], align="center")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to {SCREENSHOT_DIR / 'feature_importance.png'}")


def main():
    pipeline, label_encoder, model_name, X_test, y_test = load_artifacts()
    print(f"Evaluating model: {model_name}\n")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print_metrics(y_test, y_pred, y_proba, label_encoder)
    plot_confusion_matrix(y_test, y_pred, label_encoder)
    plot_roc_curve(pipeline, X_test, y_test)
    plot_feature_importance(pipeline)


if __name__ == "__main__":
    main()