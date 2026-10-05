"""Shared data and artifact helpers for the three Keras training scripts."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from augmentation import Augmentation
from preprocessing import DataPreprocessor


ARTIFACTS_DIR = Path("artifacts")
BATCH_SIZE = 32


def load_train_val_data():
    preprocessor = DataPreprocessor()
    file_table = preprocessor.build_file_table()
    train_images, train_labels = preprocessor.load_split(file_table, "train")
    val_images, val_labels = preprocessor.load_split(file_table, "val")
    return train_images, train_labels, val_images, val_labels


def get_class_weight(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def make_train_batches(images, labels, batch_size=BATCH_SIZE, augmentation=None):
    augmenter = Augmentation(**(augmentation or {}))
    return augmenter.flow(
        images, labels, batch_size=batch_size, shuffle=True, seed=42
    )


def create_artifact_dir(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_DIR / f"{timestamp}_{model_name}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return timestamp, artifact_dir


def save_artifacts(model, history, artifact_dir, timestamp, model_name, config):
    model_path = artifact_dir / f"{model_name}.keras"
    model.save(model_path)
    (artifact_dir / "history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    (artifact_dir / "training_config.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "backend": "keras-torch",
                "model_name": model_name,
                "model": model_path.name,
                **config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Artifacts saved to {artifact_dir}")
