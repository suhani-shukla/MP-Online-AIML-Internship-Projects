"""
data_loader.py
Handles downloading/loading the CIFAR-10 dataset and provides
the class name mapping used throughout the project.
"""

import numpy as np
import tensorflow as tf


# CIFAR-10 class names in label-index order (0-9)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_raw_data():
    """
    Loads the CIFAR-10 dataset via Keras' built-in loader.
    Downloads and caches to ~/.keras/datasets on first run.

    Returns:
        (x_train, y_train), (x_test, y_test): raw numpy arrays.
        x_* shape: (N, 32, 32, 3), dtype uint8
        y_* shape: (N, 1), dtype uint8 (class indices 0-9)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def get_class_names():
    """Returns the list of CIFAR-10 class names, indexed by label."""
    return CLASS_NAMES


def dataset_summary(x_train, y_train, x_test, y_test):
    """Prints a quick sanity-check summary of the loaded dataset."""
    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)
    print("Number of classes:", len(np.unique(y_train)))
    print("Class names:", CLASS_NAMES)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_raw_data()
    dataset_summary(x_train, y_train, x_test, y_test)