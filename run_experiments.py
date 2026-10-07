# Run each of the 60 approved experiments in a fresh Python process.
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import subprocess
import sys
from pathlib import Path

from experiments import MODELS, RUNS, SMOKE_TEST

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / 'artifacts'
RESULTS_DIR = ROOT / 'results'
if SMOKE_TEST:
    ARTIFACTS_DIR = ARTIFACTS_DIR / 'smoke'
    RESULTS_DIR = RESULTS_DIR / 'smoke'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
for run in RUNS:
    run_dir = ARTIFACTS_DIR / run['model'] / run['experiment'] / f"seed_{run['seed']}"
    if (run_dir / 'run.json').exists():
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

# Select using validation, then evaluate the fixed test runs.
subprocess.run([sys.executable, 'summarize_results.py'], cwd=ROOT, env=env, check=True)
subprocess.run([sys.executable, 'validate.py'], cwd=ROOT, env=env, check=True)
subprocess.run([sys.executable, 'summarize_results.py'], cwd=ROOT, env=env, check=True)
