"""Train a deeper CNN with multiple convolutional blocks."""

import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers

from training_utils import (
    BATCH_SIZE,
    create_artifact_dir,
    get_class_weight,
    load_train_val_data,
    make_train_batches,
    save_artifacts,
)
from config import load_config


def build_model(
    input_shape=(256, 256, 3),
    num_classes=2,
    block_filters=(32, 64, 128, 256),
    block_dropout=0.2,
    dense_units=256,
    dense_dropout=0.5,
):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in block_filters:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(block_dropout)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="complex_cnn")


def main():
    config = load_config("complex")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="complex")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.config != "complex":
        config = load_config(args.config)
    epochs = args.epochs or config["epochs"]
    batch_size = args.batch_size or config["batch_size"]

    timestamp, artifact_dir = create_artifact_dir("complex_cnn")
    train_images, train_labels, val_images, val_labels = load_train_val_data()
    model = build_model(
        block_filters=config["block_filters"],
        block_dropout=config["block_dropout"],
        dense_units=config["dense_units"],
        dense_dropout=config["dense_dropout"],
    )
    model.compile(
        optimizer=keras.optimizers.Adam(config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        make_train_batches(
            train_images, train_labels, batch_size, config["augmentation"]
        ),
        epochs=epochs,
        validation_data=(val_images, val_labels),
        class_weight=get_class_weight(train_labels),
        verbose=2,
    )
    save_artifacts(
        model, history, artifact_dir, timestamp, "complex_cnn",
        {**config, "batch_size": batch_size, "epochs": epochs},
    )


if __name__ == "__main__":
    main()
