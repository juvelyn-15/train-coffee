# Select the best model/scenario using validation only, before test evaluation.

import json
from pathlib import Path

import pandas as pd

from experiments import RUNS
from validate import calculate_metrics

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / 'artifacts'
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
rows = []
for item in RUNS:
    run_dir = ARTIFACTS_DIR / item['model'] / item['experiment'] / f"seed_{item['seed']}"
    predictions = pd.read_csv(run_dir / 'validation_predictions.csv')
    metrics = calculate_metrics(predictions.label, predictions.p_defect)
    rows.append({**item, 'validation_f1_defect': metrics['f1_defect']})
runs = pd.DataFrame(rows)
validation = runs.groupby(['model', 'experiment']).validation_f1_defect.agg(['mean', 'std']).reset_index()
validation.to_csv(RESULTS_DIR / 'validation_summary.csv', index=False)
best = validation.sort_values(['mean', 'model', 'experiment'], ascending=[False, True, True]).iloc[0]
selection = {'model': str(best.model), 'experiment': str(best.experiment),
             'mean_validation_f1_defect': float(best['mean']), 'threshold': 0.5,
             'tie_break': 'model name, then experiment ID',
             'decision_data': 'validation only'}
selection_path = RESULTS_DIR / 'selection.json'
selection_path.write_text(json.dumps(selection, indent=2))
print('Validation selection:', selection)
