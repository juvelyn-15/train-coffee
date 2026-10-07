# Evaluate saved sigmoid models using raw RGB images.
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import json
import sys
from pathlib import Path

import keras_hub
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import DataPreprocessor

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
SMOKE_TEST = os.environ.get('SMOKE_TEST', '0') == '1'
if SMOKE_TEST:
    ARTIFACTS_DIR = ARTIFACTS_DIR / 'smoke'
    RESULTS_DIR = RESULTS_DIR / 'smoke'


def calculate_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels).astype(int).ravel()
    probabilities = np.asarray(probabilities).ravel()
    if len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid prediction count or non-finite probabilities')
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Sigmoid probabilities must lie in [0, 1]')
    predictions = (probabilities > threshold).astype(int)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'precision_defect': float(precision_score(labels, predictions, zero_division=0)),
        'recall_defect': float(recall_score(labels, predictions, zero_division=0)),
        'f1_defect': float(f1_score(labels, predictions, zero_division=0)),
        'average_precision_defect': float(average_precision_score(labels, probabilities)),
        'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


if __name__ == '__main__':
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    # One explicit run directory, or all runs in the locked experiment protocol.
    if len(sys.argv) == 2:
        run_dirs = [Path(sys.argv[1]).resolve()]
    elif len(sys.argv) == 1:
        protocol = json.loads((RESULTS_DIR / 'protocol.json').read_text())
        run_dirs = [ARTIFACTS_DIR / run['model'] / run['experiment'] / f"seed_{run['seed']}"
                    for run in protocol['runs']]
    else:
        raise SystemExit('Usage: python validate.py [run_directory]')
    if not (RESULTS_DIR / 'selection.json').exists():
        raise ValueError('Freeze validation selection with summarize_results.py before test evaluation')
    protocol = json.loads((RESULTS_DIR / 'protocol.json').read_text())
    preprocessor = DataPreprocessor()
    table = preprocessor.build_file_table()
    if preprocessor.fingerprint(table) != protocol['dataset_fingerprint']:
        raise ValueError('Dataset changed after protocol lock')
    if SMOKE_TEST:
        table = table.groupby(['split', 'class_name'], sort=False).head(8)
    test_table = table[table.split == 'test'][['relative_path', 'class_name', 'label']].reset_index(drop=True)
    # Saved models include normalization. Never divide these pixels by 255 here.
    x_test, y_test = preprocessor.load_split(table, 'test', memory_map=True)
    batches = ImageDataGenerator().flow(x_test, y_test, batch_size=16, shuffle=False)
    for run_dir in run_dirs:
        run = json.loads((run_dir / 'run.json').read_text())
        if run['smoke_test'] != SMOKE_TEST or run['dataset_fingerprint'] != protocol['dataset_fingerprint']:
            raise ValueError(f'Run does not belong to the locked dataset: {run_dir}')
        if (run_dir / 'metrics.json').exists():
            print('Already evaluated:', run_dir)
            continue
        tf.keras.backend.clear_session()
        model = load_model(
            run_dir / 'best.keras', compile=False,
            custom_objects={'ResNetBackbone': keras_hub.models.ResNetBackbone,
                            'ResNetImageConverter': keras_hub.layers.ResNetImageConverter},
        )
        # Uncompiled reloads otherwise auto-enable GPU XLA for predict().
        model.jit_compile = False
        if model.input_shape != (None, 256, 256, 3) or model.output_shape != (None, 1):
            raise ValueError(f'Unexpected model input/output: {run_dir}')
        probabilities = model.predict(batches, verbose=0).ravel()
        predictions = test_table.copy()
        predictions['p_defect'] = probabilities
        predictions['prediction'] = (probabilities > run['threshold']).astype(int)
        predictions.to_csv(run_dir / 'test_predictions.csv', index=False)
        metrics = calculate_metrics(y_test, probabilities, run['threshold'])
        (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(run_dir, metrics)
