import os
import sys
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import f1_score
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Rescaling
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.models import Sequential, load_model

from preprocessing import DataPreprocessor, ResNet50Preprocessing
from validate import calculate_metrics


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(9)
    for split in ('train', 'val', 'test'):
        for category in ('longberry', 'peaberry', 'premium', 'defect'):
            folder = tmp_path / split / category
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(folder / 'sample.jpg')
    return tmp_path


def test_loader_preserves_raw_pixels(dataset):
    loader = DataPreprocessor(dataset)
    train, validation, test = loader.preprocess()
    for images, labels in (train, validation, test):
        assert images.dtype == np.uint8
        assert images.shape == (4, 256, 256, 3)
        assert labels.shape == (4, 1)
        assert labels.sum() == 1
        assert images.max() > 1


def test_sigmoid_metrics_match_whole_dataset_f1_and_fixed_threshold():
    labels = np.array([[0], [0], [1], [1]], dtype=np.float32)
    probabilities = np.array([[0.1], [0.6], [0.9], [0.5]], dtype=np.float32)
    expected = f1_score(labels.ravel(), probabilities.ravel() > 0.5)
    metric = F1Score(average='micro', threshold=0.5)
    metric.update_state(labels[:2], probabilities[:2])
    metric.update_state(labels[2:], probabilities[2:])
    assert float(metric.result()) == pytest.approx(expected)
    assert calculate_metrics(labels, probabilities)['f1_defect'] == pytest.approx(expected)
    assert calculate_metrics(labels, probabilities)['confusion_matrix'] == [[1, 1], [1, 1]]


def test_saved_model_takes_raw_rgb_without_external_normalization(tmp_path):
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    model = Sequential([Input((256, 256, 3)), Rescaling(1/255),
                        GlobalAveragePooling2D(), Dense(1, activation='sigmoid')])
    model.layers[-1].set_weights([np.ones((3, 1), np.float32), np.zeros(1, np.float32)])
    path = tmp_path / 'model.keras'
    model.save(path)
    restored = load_model(path, compile=False)
    restored.save(path)
    restored = load_model(path, compile=False)
    np.testing.assert_allclose(restored(images).numpy(), model(images).numpy())
    assert not np.allclose(restored(images).numpy(), restored(images / 255).numpy())


def test_resnet50_preprocessing_survives_safe_model_reload(tmp_path):
    pixels = np.array([[[[255., 128., 0.]]]], dtype=np.float32)
    model = Sequential([Input((1, 1, 3)), ResNet50Preprocessing()])
    np.testing.assert_allclose(model(pixels).numpy(), preprocess_input(pixels.copy()))
    path = tmp_path / 'resnet_preprocessing.keras'
    model.save(path)
    restored = load_model(path, compile=False)
    np.testing.assert_allclose(restored(pixels).numpy(), preprocess_input(pixels.copy()))


def test_memory_map_matches_raw_images_without_an_iterator_copy(dataset):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    loader = DataPreprocessor(dataset)
    table = loader.build_file_table()
    raw, labels = loader.load_split(table, 'train')
    mapped, mapped_labels = loader.load_split(table, 'train', memory_map=True)
    np.testing.assert_array_equal(mapped, raw)
    np.testing.assert_array_equal(mapped_labels, labels)
    assert isinstance(mapped, np.memmap)
    assert not mapped.flags.writeable
    iterator = ImageDataGenerator().flow(mapped, labels, batch_size=2, shuffle=False)
    assert np.shares_memory(iterator.x, mapped)
    batch, batch_labels = next(iterator)
    np.testing.assert_array_equal(batch, raw[:2])
    np.testing.assert_array_equal(batch_labels, labels[:2])


def test_saved_image_converter_predicts_without_gpu_xla(tmp_path):
    from keras_hub.layers import ResNetImageConverter
    pixels = np.full((2, 256, 256, 3), 128, dtype=np.uint8)
    model = Sequential([Input((256, 256, 3)),
                        ResNetImageConverter(image_size=(256, 256), scale=1/255),
                        GlobalAveragePooling2D(), Dense(1, activation='sigmoid')])
    expected = model(pixels, training=False).numpy()
    path = tmp_path / 'converter.keras'
    model.save(path)
    restored = load_model(path, compile=False)
    restored.jit_compile = False
    np.testing.assert_allclose(restored.predict(pixels, batch_size=1, verbose=0), expected, rtol=1e-5)
