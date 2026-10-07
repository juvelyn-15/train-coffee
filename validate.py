# Evaluate saved sigmoid models using raw RGB images.

import json
import sys
from pathlib import Path

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

from data_loader import load_split
from experiments import RUNS
from model_inputs import ResNet50Preprocessing

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'


def calculate_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels).astype(int).ravel()
    probabilities = np.asarray(probabilities).ravel()
    if not np.isfinite(probabilities).all():
        raise ValueError('Non-finite probabilities')
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
    # One explicit run directory, or all runs in the fixed experiment grid.
    if len(sys.argv) == 2:
        run_dirs = [Path(sys.argv[1]).resolve()]
    elif len(sys.argv) == 1:
        run_dirs = [ARTIFACTS_DIR / run['model'] / run['experiment'] / f"seed_{run['seed']}"
                    for run in RUNS]
    else:
        raise SystemExit('Usage: python validate.py [run_directory]')
    # Saved models include normalization. Never divide these pixels by 255 here.
    x_test, y_test, test_table = load_split('test')
    batches = ImageDataGenerator().flow(x_test, y_test, batch_size=16, shuffle=False)
    for run_dir in run_dirs:
        run = json.loads((run_dir / 'run.json').read_text())
        if (run_dir / 'metrics.json').exists():
            print('Already evaluated:', run_dir)
            continue
        tf.keras.backend.clear_session()
        custom_objects = {'coffee>ResNet50Preprocessing': ResNet50Preprocessing}
        if run['model'] == 'resnet18':
            import keras_hub
            custom_objects = {'ResNetBackbone': keras_hub.models.ResNetBackbone,
                              'ResNetImageConverter': keras_hub.layers.ResNetImageConverter}
        model = load_model(run_dir / 'best.keras', compile=False, custom_objects=custom_objects)
        # Uncompiled reloads otherwise auto-enable GPU XLA for predict().
        model.jit_compile = False
        probabilities = model.predict(batches, verbose=0).ravel()
        predictions = test_table.copy()
        predictions['p_defect'] = probabilities
        predictions['prediction'] = (probabilities > run['threshold']).astype(int)
        predictions.to_csv(run_dir / 'test_predictions.csv', index=False)
        metrics = calculate_metrics(y_test, probabilities, run['threshold'])
        (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(run_dir, metrics)
