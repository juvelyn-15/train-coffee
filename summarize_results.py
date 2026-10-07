# Summarize runs and select the best model using validation only.
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import MODELS, EXPERIMENTS, RUNS, SMOKE_TEST
from validate import calculate_metrics

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / 'artifacts'
RESULTS_DIR = ROOT / 'results'
if SMOKE_TEST:
    ARTIFACTS_DIR = ARTIFACTS_DIR / 'smoke'
    RESULTS_DIR = RESULTS_DIR / 'smoke'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
rows = []
for item in RUNS:
    run_dir = ARTIFACTS_DIR / item['model'] / item['experiment'] / f"seed_{item['seed']}"
    run = json.loads((run_dir / 'run.json').read_text())
    predictions = pd.read_csv(run_dir / 'validation_predictions.csv')
    val_metrics = calculate_metrics(predictions.label, predictions.p_defect, run['threshold'])
    row = {**item, 'validation_f1_defect': val_metrics['f1_defect'],
           'parameters': run['parameters'], 'training_seconds': run['training_seconds'],
           'epochs_completed': run['epochs_completed']}
    metrics_path = run_dir / 'metrics.json'
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        row.update({k: v for k, v in metrics.items() if k != 'confusion_matrix'})
    rows.append(row)
runs = pd.DataFrame(rows)
runs.to_csv(RESULTS_DIR / 'runs.csv', index=False)
validation = runs.groupby(['model', 'experiment']).validation_f1_defect.agg(['mean', 'std']).reset_index()
validation.to_csv(RESULTS_DIR / 'validation_summary.csv', index=False)
best = validation.sort_values(['mean', 'model', 'experiment'], ascending=[False, True, True]).iloc[0]
selection = {'model': str(best.model), 'experiment': str(best.experiment),
             'mean_validation_f1_defect': float(best['mean']), 'threshold': 0.5,
             'smoke_test': SMOKE_TEST,
             'tie_break': 'model name, then experiment ID',
             'decision_data': 'validation only'}
selection_path = RESULTS_DIR / 'selection.json'
selection_path.write_text(json.dumps(selection, indent=2))
print('Validation selection:', selection)

if 'f1_defect' in runs:
    if runs.f1_defect.isna().any():
        raise ValueError('Some test evaluations are missing')
    columns = ['accuracy', 'balanced_accuracy', 'precision_defect', 'recall_defect',
               'f1_defect', 'average_precision_defect']
    grouped = runs.groupby(['model', 'experiment'])[columns].agg(['mean', 'std'])
    model_order = list(MODELS)
    experiment_order = list(EXPERIMENTS)
    grouped = grouped.reindex(pd.MultiIndex.from_product(
        [model_order, experiment_order], names=['model', 'experiment']))
    grouped.columns = ['_'.join(column) for column in grouped.columns]
    grouped.reset_index().to_csv(RESULTS_DIR / 'experiment_summary.csv', index=False)
    display = grouped.copy()
    for column in columns:
        display[column] = [f'{mean:.4f} +/- {std:.4f}'
                           for mean, std in zip(grouped[column + '_mean'], grouped[column + '_std'])]
    title = '# Smoke check results (not full experiments)' if SMOKE_TEST else '# Experiment results'
    lines = [title, '',
             'Generated from measured test predictions after validation selection.', '',
             '| Model | Experiment | Accuracy | Balanced accuracy | Precision-defect | Recall-defect | F1-defect |',
             '|---|---|---:|---:|---:|---:|---:|']
    for (model, experiment), row in display.iterrows():
        lines.append('| ' + ' | '.join([model, experiment, *[row[c] for c in columns[:5]]]) + ' |')
    (RESULTS_DIR / 'experiment_summary.md').write_text('\n'.join(lines) + '\n')
    figures_dir = RESULTS_DIR / 'figures'
    figures_dir.mkdir(exist_ok=True)
    for item in RUNS:
        run_dir = ARTIFACTS_DIR / item['model'] / item['experiment'] / f"seed_{item['seed']}"
        stem = f"{item['model']}_{item['experiment']}_seed_{item['seed']}"
        history = pd.read_csv(run_dir / 'history.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, metric in zip(axes, ['loss', 'f1_defect']):
            axis.plot(np.arange(1, len(history) + 1), history[metric], label='train', marker='o', markersize=2)
            axis.plot(np.arange(1, len(history) + 1), history['val_' + metric], label='validation', marker='o', markersize=2)
            if (history.phase == 'fine_tune').any():
                axis.axvline(int((history.phase == 'head').sum()) + 0.5, color='gray', linestyle='--')
            axis.set(xlabel='Epoch', ylabel=metric, title=metric)
            axis.legend()
        fig.suptitle(stem)
        fig.tight_layout()
        fig.savefig(figures_dir / f'{stem}_learning.png', dpi=140)
        plt.close(fig)
        metrics = json.loads((run_dir / 'metrics.json').read_text())
        matrix = np.array(metrics['confusion_matrix'])
        fig, axis = plt.subplots(figsize=(5, 4))
        axis.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(matrix[i, j]), ha='center', va='center',
                          color='white' if matrix[i, j] > matrix.max()/2 else 'black')
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['normal', 'defect'],
                 yticklabels=['normal', 'defect'], xlabel='Predicted', ylabel='Actual', title=stem)
        fig.tight_layout()
        fig.savefig(figures_dir / f'{stem}_confusion.png', dpi=140)
        plt.close(fig)
    print('Test results:', RESULTS_DIR / 'experiment_summary.csv')
