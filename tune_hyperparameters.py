"""Small validation-only hyperparameter search for the three model types."""

import argparse
import json
import os
from datetime import datetime
from itertools import product

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras

from train_complex_cnn import build_model as build_complex_model
from train_simple import build_model as build_simple_model
from train_transfer_learning import build_model as build_transfer_model
from config import load_config
from training_utils import (
    ARTIFACTS_DIR,
    get_class_weight,
    load_train_val_data,
    make_train_batches,
)


def build_for(name, learning_rate, dropout):
    if name == "simple":
        model = build_simple_model(dropout=dropout)
    elif name == "complex":
        model = build_complex_model(block_dropout=dropout)
    else:
        model = build_transfer_model(weights="imagenet", dropout=dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("simple", "complex", "transfer"), default="simple")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    search_space = load_config("search_space")
    train_images, train_labels, val_images, val_labels = load_train_val_data()
    trials = [
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout": dropout,
        }
        for learning_rate, batch_size, dropout in product(
            search_space["learning_rate"],
            search_space["batch_size"],
            search_space["dropout"],
        )
    ][:args.trials]
    results = []
    for trial in trials:
        model = build_for(
            args.model, trial["learning_rate"], trial["dropout"]
        )
        history = model.fit(
            make_train_batches(
                train_images, train_labels, batch_size=trial["batch_size"]
            ),
            epochs=args.epochs,
            validation_data=(val_images, val_labels),
            class_weight=get_class_weight(train_labels),
            verbose=2,
        )
        results.append(
            {
                "learning_rate": trial["learning_rate"],
                "batch_size": trial["batch_size"],
                "val_accuracy": float(history.history["val_accuracy"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
            }
        )

    best = max(results, key=lambda result: result["val_accuracy"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ARTIFACTS_DIR / f"{timestamp}_tuning_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "tuning_results.json").write_text(
        json.dumps(
            {"model": args.model, "trials": results, "best": best}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Best configuration: {best}")
    print(f"Tuning results saved to {output_dir}")


if __name__ == "__main__":
    main()
