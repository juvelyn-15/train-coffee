from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Augmentation(ImageDataGenerator):
    """Augment training images; return float32 batches in [0, 255].

    Pass raw RGB arrays shaped (N, H, W, 3), not normalized images.
    Apply model-specific input preprocessing after augmentation.
    """

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
