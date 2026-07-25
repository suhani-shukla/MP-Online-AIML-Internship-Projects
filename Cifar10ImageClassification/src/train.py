"""
train.py
Trains the CNN on CIFAR-10, saves the trained model, class names,
and training history for later evaluation.
"""

import os
import pickle
import json

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_loader import load_raw_data, get_class_names
from preprocess import preprocess_pipeline
from model import build_model


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.pkl")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")


def get_callbacks():
    """Returns training callbacks: early stopping, checkpointing, LR schedule."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]


def train(epochs=50, batch_size=64, val_size=0.1):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_raw_data()

    print("Preprocessing data...")
    data = preprocess_pipeline(x_train, y_train, x_test, y_test, val_size=val_size)

    print("Building model...")
    model = build_model(input_shape=data["x_train"].shape[1:], num_classes=10)
    model.summary()

    print("Training...")
    history = model.fit(
        data["x_train"], data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Save final model (in case checkpoint didn't trigger on last epoch)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save class names
    with open(CLASS_NAMES_PATH, "wb") as f:
        pickle.dump(get_class_names(), f)
    print(f"Class names saved to {CLASS_NAMES_PATH}")

    # Save training history for plotting in evaluate.py
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f)
    print(f"Training history saved to {HISTORY_PATH}")

    return model, history


if __name__ == "__main__":
    train()