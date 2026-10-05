"""Validate a Keras or PyTorch model on the test split."""

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import DataPreprocessor


# Load Keras with the same PyTorch backend used during training.
os.environ.setdefault("KERAS_BACKEND", "torch")

BATCH_SIZE = 32
CLASS_NAMES = ("normal", "defect")
ARTIFACTS_DIR = Path("artifacts")


def load_test_data():
    file_table = DataPreprocessor().build_file_table()
    images, labels = DataPreprocessor().load_split(file_table, "test")
    return images.astype(np.float32) / 255.0, labels


def print_metrics(model_path, labels, predictions, loss=None, accuracy=None):
    print(f"Model: {model_path}")
    if loss is not None:
        print(f"Test loss: {loss:.4f}")
    if accuracy is not None:
        print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(
        labels, predictions, target_names=CLASS_NAMES,
        digits=4, zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))


def validate_keras(model_path, images, labels):
    try:
        import keras
    except ImportError as exc:
        raise SystemExit("Keras standalone is required to validate this model.") from exc

    model = keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(images, labels, batch_size=BATCH_SIZE, verbose=0)
    predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    print_metrics(model_path, labels, predictions, loss, accuracy)


def latest_model_path():
    models = sorted(
        list(ARTIFACTS_DIR.glob("*/*.keras"))
        + list(ARTIFACTS_DIR.glob("*/*.h5"))
    )
    if not models:
        raise FileNotFoundError(f"No trained model found under {ARTIFACTS_DIR}/")
    return models[-1]


def validate(model_path):
    images, labels = load_test_data()
    validate_keras(model_path, images, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained model")
    parser.add_argument(
        "model", nargs="?", type=Path,
        help="Path to a .keras/.h5 model; defaults to the latest artifact",
    )
    args = parser.parse_args()
    model_path = args.model or latest_model_path()
    if not model_path.exists():
        parser.error(f"Model file not found: {model_path}")
    validate(model_path)
