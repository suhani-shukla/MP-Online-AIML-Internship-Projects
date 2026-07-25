"""
predict.py
Loads the trained model and provides utilities to predict the class
of a single image, either from a file path or a numpy array.
Also generates a grid of sample predictions for the README/screenshots.
"""

import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from data_loader import load_raw_data
from preprocess import normalize_images


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.pkl")


def load_artifacts():
    """Loads the trained model and class names from disk."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "rb") as f:
        class_names = pickle.load(f)
    return model, class_names


def preprocess_image(image_path, target_size=(32, 32)):
    """
    Loads an image from disk and preprocesses it to match model input:
    resized to 32x32, RGB, normalized to [0, 1], batched.

    Args:
        image_path: path to an image file
        target_size: (H, W) expected by the model

    Returns:
        numpy array of shape (1, H, W, 3), dtype float32
    """
    img = Image.open(image_path).convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(image_path, model=None, class_names=None):
    """
    Predicts the class of a single image file.

    Returns:
        dict with keys: predicted_class, confidence, all_probabilities
    """
    if model is None or class_names is None:
        model, class_names = load_artifacts()

    x = preprocess_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "all_probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }
    }


def predict_array(image_array, model=None, class_names=None):
    """
    Predicts the class of an already-loaded image array
    (e.g. from Streamlit file uploader), shape (H, W, 3), values 0-255.

    Returns:
        dict with keys: predicted_class, confidence, all_probabilities
    """
    if model is None or class_names is None:
        model, class_names = load_artifacts()

    img = Image.fromarray(image_array.astype("uint8")).convert("RGB").resize((32, 32))
    x = np.expand_dims(np.array(img).astype("float32") / 255.0, axis=0)

    probs = model.predict(x, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "all_probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }
    }


def generate_sample_predictions_grid(num_samples=10, save_path=None):
    """
    Picks random test images, predicts them, and saves a grid image
    showing each with its true vs predicted label.
    """
    model, class_names = load_artifacts()
    (_, _), (x_test, y_test) = load_raw_data()

    idxs = np.random.choice(len(x_test), num_samples, replace=False)
    images = x_test[idxs]
    true_labels = y_test[idxs].reshape(-1)

    x_norm = normalize_images(images)
    probs = model.predict(x_norm, verbose=0)
    pred_labels = np.argmax(probs, axis=1)

    cols = 5
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i])
        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        color = "green" if true_name == pred_name else "red"
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=9)
        ax.axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Sample predictions grid saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CIFAR-10 image class")
    parser.add_argument("--image", type=str, help="Path to an image file")
    parser.add_argument(
        "--grid", action="store_true",
        help="Generate a sample predictions grid from the test set"
    )
    args = parser.parse_args()

    if args.grid:
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        generate_sample_predictions_grid(
            num_samples=10,
            save_path=os.path.join(SCREENSHOTS_DIR, "sample_predictions.png")
        )
    elif args.image:
        result = predict_image(args.image)
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print("Provide --image <path> for a single prediction, or --grid to generate a sample grid.")