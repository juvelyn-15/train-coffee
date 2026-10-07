# Run each of the 60 approved experiments in a fresh Python process.
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from preprocessing import DataPreprocessor

ROOT = Path(__file__).resolve().parent
MODELS = {
    'simple_cnn': 'train_simple.py',
    'complex_cnn': 'train_complex_cnn.py',
    'resnet18': 'train_resnet18.py',
    'resnet50': 'train_resnet50.py',
    'efficientnet_b4': 'train_efficientnet_b4.py',
}
EXPERIMENTS = ('E0', 'E1', 'E2', 'E3')
SEEDS = (42, 43, 44)
SMOKE_TEST = os.environ.get('SMOKE_TEST', '0') == '1'
# Smoke checks all model/experiment paths once, with a small stratified subset.
if SMOKE_TEST:
    SEEDS = (42,)
ARTIFACTS_DIR = ROOT / 'artifacts'
RESULTS_DIR = ROOT / 'results'
if SMOKE_TEST:
    ARTIFACTS_DIR = ARTIFACTS_DIR / 'smoke'
    RESULTS_DIR = RESULTS_DIR / 'smoke'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

preprocessor = DataPreprocessor()
table = preprocessor.build_file_table()
dataset_fingerprint = preprocessor.fingerprint(table)
protocol = {
    'dataset_fingerprint': dataset_fingerprint,
    'smoke_test': SMOKE_TEST,
    'selection_metric': 'validation_f1_defect', 'threshold': 0.5,
    'runs': [{'model': model, 'experiment': experiment, 'seed': seed}
             for model in MODELS for experiment in EXPERIMENTS for seed in SEEDS],
    'source_hashes': {name: sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in [*MODELS.values(), 'preprocessing.py', 'validate.py']},
}
protocol_path = RESULTS_DIR / 'protocol.json'
if protocol_path.exists():
    if json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing protocol differs; preserve results and choose a fresh results directory')
else:
    protocol_path.write_text(json.dumps(protocol, indent=2))
table.to_csv(RESULTS_DIR / 'dataset_manifest.csv', index=False)

env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env['DATASET_FINGERPRINT'] = dataset_fingerprint
for run in protocol['runs']:
    run_dir = ARTIFACTS_DIR / run['model'] / run['experiment'] / f"seed_{run['seed']}"
    if (run_dir / 'run.json').exists():
        saved = json.loads((run_dir / 'run.json').read_text())
        if saved['dataset_fingerprint'] != dataset_fingerprint:
            raise ValueError(f'Dataset mismatch in completed run: {run_dir}')
        print('Completed, skipping:', run_dir, flush=True)
        continue
    run_dir.mkdir(parents=True, exist_ok=True)
    env['EXPERIMENT'] = run['experiment']
    env['SEED'] = str(run['seed'])
    print('Training:', run, flush=True)
    with (run_dir / 'training.log').open('w') as log:
        subprocess.run([sys.executable, MODELS[run['model']]], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed:', run, flush=True)

# Selection depends on validation only. Then evaluate every predeclared test row.
subprocess.run([sys.executable, 'summarize_results.py'], cwd=ROOT, env=env, check=True)
subprocess.run([sys.executable, 'validate.py'], cwd=ROOT, env=env, check=True)
subprocess.run([sys.executable, 'summarize_results.py'], cwd=ROOT, env=env, check=True)
