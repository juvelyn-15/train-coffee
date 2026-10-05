"""Train the coffee bean CNN with standalone Keras 3."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# Configure Keras before importing it: Keras 3 will use PyTorch as backend.
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from keras import layers
from sklearn.utils.class_weight import compute_class_weight

from augmentation import Augmentation
from config import load_config
from preprocessing import DataPreprocessor

BATCH_SIZE = 32
EPOCHS = 20
ARTIFACTS_DIR = Path("artifacts")


def build_model(
    input_shape=(256, 256, 3),
    num_classes=2,
    dropout=0.25,
    dense_dropout=0.5,
):

    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(dropout),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(dropout),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(dropout),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dense_dropout),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def load_train_val_data():
    file_table = DataPreprocessor().build_file_table()
    train_images, train_labels = DataPreprocessor().load_split(file_table, "train")
    val_images, val_labels = DataPreprocessor().load_split(file_table, "val")
    return (
        train_images,
        train_labels,
        val_images,
        val_labels,
    )


def main():
    config = load_config("simple")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="simple")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.config != "simple":
        config = load_config(args.config)
    epochs = args.epochs or config["epochs"]
    batch_size = args.batch_size or config["batch_size"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_DIR / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=False)

    train_images, train_labels, val_images, val_labels = load_train_val_data()
    labels = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=labels, y=train_labels
    )
    class_weight = {int(label): float(weight) for label, weight in zip(labels, weights)}

    model = build_model(
        dropout=config["dropout"], dense_dropout=config["dense_dropout"]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    train_batches = Augmentation(**config["augmentation"]).flow(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    history = model.fit(
        train_batches,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        class_weight=class_weight,
        verbose=2,
    )

    model_path = artifact_dir / "keras_coffee_bean.keras"
    model.save(model_path)
    (artifact_dir / "history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    (artifact_dir / "training_config.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "backend": "keras-torch",
                **config,
                "batch_size": batch_size,
                "epochs": epochs,
                "class_weight": class_weight,
                "model": model_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Artifacts saved to {artifact_dir}")


if __name__ == "__main__":
    main()
