import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import f1_score
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Rescaling
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.models import Sequential, load_model

from data_loader import load_split
from model_inputs import ResNet50Preprocessing
from validate import calculate_metrics


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(9)
    for split in ('train', 'val', 'test'):
        for category in ('normal', 'defect'):
            folder = tmp_path / split / category
            folder.mkdir(parents=True)
            names = ('premium_1.jpg', 'longberry_1.jpg', 'peaberry_1.jpg') if category == 'normal' else ('sample.jpg',)
            for name in names:
                Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(folder / name)
    return tmp_path


def test_loader_preserves_raw_pixels(dataset):
    for split in ('train', 'val', 'test'):
        images, labels, metadata = load_split(split, dataset)
        assert images.dtype == np.float32
        assert images.shape == (4, 256, 256, 3)
        assert labels.shape == (4, 1)
        np.testing.assert_array_equal(labels.ravel(), [0, 0, 0, 1])
        assert metadata.class_name.tolist() == ['normal', 'normal', 'normal', 'defect']
        assert metadata.relative_path.tolist() == [
            f'{split}/normal/longberry_1.jpg',
            f'{split}/normal/peaberry_1.jpg',
            f'{split}/normal/premium_1.jpg',
            f'{split}/defect/sample.jpg',
        ]
        for index, relative_path in enumerate(metadata.relative_path):
            with Image.open(dataset / relative_path) as image:
                np.testing.assert_array_equal(images[index], np.asarray(image))
        np.testing.assert_array_equal(labels.ravel(), metadata.label)


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
    mapped, labels, metadata = load_split('train', dataset)
    cached, cached_labels, cached_metadata = load_split('train', dataset)
    np.testing.assert_array_equal(cached, mapped)
    np.testing.assert_array_equal(cached_labels, labels)
    assert cached_metadata.equals(metadata)
    assert isinstance(mapped, np.memmap)
    assert not mapped.flags.writeable
    iterator = ImageDataGenerator().flow(mapped, labels, batch_size=2, shuffle=False)
    assert np.shares_memory(iterator.x, mapped)
    batch, batch_labels = next(iterator)
    np.testing.assert_array_equal(batch, mapped[:2])
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


@pytest.mark.parametrize('probability', [np.nan, np.inf, -np.inf])
def test_metrics_reject_non_finite_model_outputs(probability):
    with pytest.raises(ValueError, match='Non-finite'):
        calculate_metrics([0, 1], [0.1, probability])


def test_selection_and_summary_keep_test_results_out_of_selection(tmp_path, monkeypatch):
    import json
    import runpy
    import shutil

    import pandas as pd

    import experiments

    root = Path(__file__).resolve().parents[1]
    runs = [{'model': model, 'experiment': 'E0', 'seed': seed}
            for model in ('simple_cnn', 'complex_cnn') for seed in (42, 43)]
    monkeypatch.setattr(experiments, 'RUNS', runs)
    monkeypatch.setattr(experiments, 'MODELS', dict.fromkeys(('simple_cnn', 'complex_cnn')))
    monkeypatch.setattr(experiments, 'EXPERIMENTS', {'E0': (False, None)})
    for name in ('select_model.py', 'summarize_results.py'):
        shutil.copy(root / name, tmp_path / name)
    for run in runs:
        folder = tmp_path / 'artifacts' / run['model'] / 'E0' / f"seed_{run['seed']}"
        folder.mkdir(parents=True)
        # Validation favors simple; test favors complex.
        probabilities = [0.1, 0.9] if run['model'] == 'simple_cnn' else [0.9, 0.1]
        pd.DataFrame({'label': [0, 1], 'p_defect': probabilities}).to_csv(
            folder / 'validation_predictions.csv', index=False)
        (folder / 'run.json').write_text(json.dumps({
            'threshold': 0.5, 'parameters': 10, 'training_seconds': 1, 'epochs_completed': 1}))
        pd.DataFrame({'phase': ['training'], 'loss': [0.5], 'val_loss': [0.5],
                      'f1_defect': [0.5], 'val_f1_defect': [0.5]}).to_csv(
            folder / 'history.csv', index=False)
    # Selection works before any test output exists.
    runpy.run_path(str(tmp_path / 'select_model.py'))
    selection_path = tmp_path / 'results' / 'selection.json'
    selection = selection_path.read_bytes()
    assert json.loads(selection)['model'] == 'simple_cnn'
    assert json.loads(selection)['mean_validation_f1_defect'] == 1.0
    # A partial test evaluation cannot silently produce a summary.
    with pytest.raises(FileNotFoundError):
        runpy.run_path(str(tmp_path / 'summarize_results.py'))
    for run in runs:
        folder = tmp_path / 'artifacts' / run['model'] / 'E0' / f"seed_{run['seed']}"
        probabilities = [0.9, 0.1] if run['model'] == 'simple_cnn' else [0.1, 0.9]
        (folder / 'metrics.json').write_text(json.dumps(calculate_metrics([0, 1], probabilities)))
    runpy.run_path(str(tmp_path / 'summarize_results.py'))
    assert selection_path.read_bytes() == selection
    summary = pd.read_csv(tmp_path / 'results' / 'experiment_summary.csv').set_index('model')
    assert summary.loc['simple_cnn', 'f1_defect_mean'] == 0.0
    assert summary.loc['complex_cnn', 'f1_defect_mean'] == 1.0
    assert summary.f1_defect_std.eq(0).all()
    assert len(list((tmp_path / 'results' / 'figures').glob('*.png'))) == 8
