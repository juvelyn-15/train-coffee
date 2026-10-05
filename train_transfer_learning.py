"""Train a MobileNetV2 transfer-learning model."""

import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers

from config import load_config
from training_utils import (
    BATCH_SIZE,
    create_artifact_dir,
    get_class_weight,
    load_train_val_data,
    make_train_batches,
    save_artifacts,
)


def build_model(
    input_shape=(256, 256, 3),
    num_classes=2,
    weights="imagenet",
    fine_tune_layers=0,
    dropout=0.3,
):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = fine_tune_layers > 0
    if fine_tune_layers > 0:
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def main():
    config = load_config("transfer_learning")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="transfer_learning")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--weights", choices=("imagenet", "none"),
        help="Use ImageNet pretrained weights or train the backbone from scratch",
    )
    parser.add_argument(
        "--fine-tune-layers", type=int,
        help="Unfreeze this many final MobileNetV2 layers after loading weights",
    )
    args = parser.parse_args()
    if args.config != "transfer_learning":
        config = load_config(args.config)

    epochs = args.epochs or config["epochs"]
    batch_size = args.batch_size or config["batch_size"]
    weights_name = args.weights or config["weights"]
    fine_tune_layers = (
        args.fine_tune_layers
        if args.fine_tune_layers is not None
        else config["fine_tune_layers"]
    )
    weights = None if weights_name == "none" else weights_name
    timestamp, artifact_dir = create_artifact_dir("transfer_learning")
    train_images, train_labels, val_images, val_labels = load_train_val_data()
    model = build_model(
        weights=weights,
        fine_tune_layers=fine_tune_layers,
        dropout=config["dropout"],
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
        model, history, artifact_dir, timestamp, "transfer_learning",
        {
            **config,
            "batch_size": batch_size,
            "epochs": epochs,
            "weights": weights_name,
            "backbone": "MobileNetV2",
            "fine_tune_layers": fine_tune_layers,
        },
    )


if __name__ == "__main__":
    main()
