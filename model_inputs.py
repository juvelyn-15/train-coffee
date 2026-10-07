"""Model-specific input processing saved with the trained model."""

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package='coffee')
class ResNet50Preprocessing(Layer):
    """Convert raw RGB pixels to the ImageNet ResNet50 input convention."""

    def call(self, images):
        return preprocess_input(images)
