"""
model.py
Defines the CNN architecture used for CIFAR-10 classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(32, 32, 3), num_classes=10, use_augmentation=True):
    """
    Builds and compiles a CNN for CIFAR-10 classification.

    Architecture: 3 Conv blocks (Conv2D -> BatchNorm -> Conv2D -> BatchNorm
    -> MaxPool -> Dropout), followed by a Dense classifier head.

    Args:
        input_shape: shape of input images (H, W, C)
        num_classes: number of output classes
        use_augmentation: if True, prepends augmentation layers
                           (active only during training, no-op at inference)

    Returns:
        Compiled tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    if use_augmentation:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.05)(x)
        x = layers.RandomTranslation(0.1, 0.1)(x)
        x = layers.RandomZoom(0.1)(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Classifier head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cifar10_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()