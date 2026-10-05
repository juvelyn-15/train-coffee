"""Keras 3 legacy ImageDataGenerator for raw RGB train images."""

import os

# Keras must be configured before importing its legacy preprocessing API.
os.environ.setdefault("KERAS_BACKEND", "torch")

from keras.src.legacy.preprocessing.image import ImageDataGenerator


class Augmentation(ImageDataGenerator):
    """Augment training images in the raw [0, 255] pixel range."""

    def __init__(
        self,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.9, 1.1),
    ):
        super().__init__(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            brightness_range=brightness_range,
            fill_mode="nearest",
            data_format="channels_last",
            dtype="float32",
        )
