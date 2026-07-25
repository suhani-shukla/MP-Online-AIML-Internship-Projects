"""
preprocess.py
Handles normalization, label encoding, train/validation splitting,
and data augmentation for CIFAR-10 images.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def normalize_images(x):
    """
    Scales pixel values from [0, 255] to [0, 1].

    Args:
        x: numpy array of images, dtype uint8 or float

    Returns:
        numpy array of images, dtype float32, scaled to [0, 1]
    """
    return x.astype("float32") / 255.0


def encode_labels(y, num_classes=10):
    """
    One-hot encodes integer class labels.

    Args:
        y: numpy array of shape (N, 1) or (N,) with integer labels
        num_classes: total number of classes

    Returns:
        numpy array of shape (N, num_classes)
    """
    y = y.reshape(-1)
    return tf.keras.utils.to_categorical(y, num_classes=num_classes)


def split_train_val(x_train, y_train, val_size=0.1, random_state=42):
    """
    Splits training data into train and validation subsets.

    Args:
        x_train, y_train: full training arrays
        val_size: fraction reserved for validation
        random_state: seed for reproducibility

    Returns:
        x_train, x_val, y_train, y_val
    """
    return train_test_split(
        x_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train.argmax(axis=1) if y_train.ndim > 1 else y_train
    )


def get_data_augmentation():
    """
    Returns a Keras Sequential model of augmentation layers,
    intended to be applied only during training (e.g. as the
    first layers of the model, or via .map() on a tf.data pipeline).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")


def preprocess_pipeline(x_train, y_train, x_test, y_test, val_size=0.1):
    """
    Full preprocessing pipeline: normalize images, one-hot encode
    labels, and split off a validation set from training data.

    Returns:
        dict with keys: x_train, x_val, x_test, y_train, y_val, y_test
    """
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    y_train = encode_labels(y_train)
    y_test = encode_labels(y_test)

    x_train, x_val, y_train, y_val = split_train_val(
        x_train, y_train, val_size=val_size
    )

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }


if __name__ == "__main__":
    from data_loader import load_raw_data

    (x_train, y_train), (x_test, y_test) = load_raw_data()
    data = preprocess_pipeline(x_train, y_train, x_test, y_test)

    for k, v in data.items():
        print(k, v.shape)