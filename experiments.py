"""Fixed experiment grid shared by the launcher and result scripts."""

import os
from itertools import product

MODELS = {
    'simple_cnn': 'train_simple.py',
    'complex_cnn': 'train_complex_cnn.py',
    'resnet18': 'train_resnet18.py',
    'resnet50': 'train_resnet50.py',
    'efficientnet_b4': 'train_efficientnet_b4.py',
}
EXPERIMENTS = ('E0', 'E1', 'E2', 'E3')
SMOKE_TEST = os.environ.get('SMOKE_TEST', '0') == '1'
SEEDS = (42,) if SMOKE_TEST else (42, 43, 44)
RUNS = [dict(model=model, experiment=experiment, seed=seed)
        for model, experiment, seed in product(MODELS, EXPERIMENTS, SEEDS)]
