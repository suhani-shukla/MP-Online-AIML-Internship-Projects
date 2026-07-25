"""
evaluate.py
Loads the trained model, evaluates it on the test set, and generates
diagnostic plots: training history curves and a confusion matrix.
"""

import os
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from data_loader import load_raw_data
from preprocess import normalize_images, encode_labels


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.pkl")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")


def load_artifacts():
    """Loads the trained model and class names from disk."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "rb") as f:
        class_names = pickle.load(f)
    return model, class_names


def plot_training_history(save_path=None):
    """Plots and saves accuracy/loss curves from the saved training history."""
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["accuracy"], label="Train")
    axes[0].plot(history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history["loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training history plot saved to {save_path}")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Computes and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def evaluate():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    print("Loading model and data...")
    model, class_names = load_artifacts()
    (_, _), (x_test, y_test_raw) = load_raw_data()

    x_test = normalize_images(x_test)
    y_test = encode_labels(y_test_raw)

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    print("Generating predictions for confusion matrix...")
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test_raw.reshape(-1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png")
    )

    plot_training_history(
        save_path=os.path.join(SCREENSHOTS_DIR, "training_history.png")
    )

    return test_loss, test_acc


if __name__ == "__main__":
    evaluate()