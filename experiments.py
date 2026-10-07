"""Fixed experiment grid shared by the launcher and result scripts."""

from itertools import product

MODELS = {
    'simple_cnn': 'train_simple.py',
    'complex_cnn': 'train_complex_cnn.py',
    'resnet18': 'train_resnet18.py',
    'resnet50': 'train_resnet50.py',
    'efficientnet_b4': 'train_efficientnet_b4.py',
}
# Fixed training counts in label order: normal=0, defect=1.
TRAIN_COUNTS = (3599, 1200)
CLASS_WEIGHTS = {label: sum(TRAIN_COUNTS) / (2 * count)
                 for label, count in enumerate(TRAIN_COUNTS)}
# Each scenario specifies (augmentation, class weights).
EXPERIMENTS = {
    'E0': (False, None),
    'E1': (True, None),
    'E2': (False, CLASS_WEIGHTS),
    'E3': (True, CLASS_WEIGHTS),
}
SEEDS = (42, 43, 44)
RUNS = [{'model': model, 'experiment': experiment, 'seed': seed}
        for model, experiment, seed in product(MODELS, EXPERIMENTS, SEEDS)]
