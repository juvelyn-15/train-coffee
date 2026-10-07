# defective-coffee-bean-classification

Requirements


Selects an image dataset, and the datasets must be different across groups. Using the selected dataset, each group performs either a classification or segmentation problem as follows:

1) Preprocess the data: clean, normalize, and augment the data.

2) Design your own simple CNN model using convolutional, pooling, and fully connected layers to solve the given problem.

3) Design your own complex CNN model using CNN blocks, convolutional, pooling, and fully connected layers to solve the given problem.

4) Design a CNN model using Transfer Learning/Fine-Tuning to perform the given task.

5) Evaluate the models using appropriate performance metrics.

## Agreed experiment design

The project classifies USK-Coffee images as normal (0) or defect (1).
Normal combines longberry, peaberry and premium.
Each model receives raw RGB 256x256 pixels and outputs a single sigmoid probability of defect.
All models use binary cross-entropy and a fixed 0.5 decision threshold.

| Model | Architecture | Initialization |
|---|---|---|
| Simple CNN | Conv16-Pool, Conv32-Pool, Conv64-Pool, GAP, Dense128, Dropout0.5, sigmoid | Random |
| Complex CNN | Four VGG blocks, two Conv-BN-ReLU sequences each, filters 32/64/128/256, Pool and Dropout0.25; GAP-Dense128-Dropout0.5-sigmoid | Random |
| ResNet18 | Backbone-GAP-Dense128-Dropout0.5-sigmoid | ImageNet, KerasHub preset |
| ResNet50 | Backbone-GAP-Dense128-Dropout0.5-sigmoid | ImageNet, Keras Applications |
| EfficientNetB4 | Backbone-GAP-Dense128-Dropout0.5-sigmoid | ImageNet, Keras Applications |

The simple and complex CNNs are project-designed experiments, not claimed reproductions of published binary USK-Coffee architectures.
The complex script follows the `vgg_block()` pattern taught in Chapter 5.2, slides 6 and 8.
Batch normalization precedes ReLU, matching slide 22.
GAP follows the pooling ideas introduced by NiN and ResNet.
The transfer scripts follow the short backbone-plus-head style of Chapter 5.3 and the provided Python templates.

| Experiment | Augmentation | Class weights |
|---|---|---|
| E0 | Off | Off |
| E1 | On | Off |
| E2 | Off | On |
| E3 | On | On |

The five models, four scenarios and three seeds (42,43,44) form 60 complete training runs.
See [preprocessing](preprocessing.md) for data cleaning and [augmentation](augmentation.md) for the approved transformations.
Oversampling, synthetic images and threshold optimization are separate future experiments.

## Training protocol

All scripts use batch size 16 and Adam.
The custom CNNs train for up to 50 epochs at learning rate 0.001.
Transfer models train their randomly initialized head for 10 epochs with the backbone frozen, then fine-tune the final backbone stage for up to 40 epochs at learning rate 0.00001.
Batch normalization remains frozen during fine-tuning.
Early stopping monitors validation F1-defect with patience 8 for custom CNNs and transfer fine-tuning.
The saved checkpoint has the highest validation F1-defect across the complete run, including both transfer phases.
The main classification metrics use unweighted predictions at the fixed threshold.

## Setup and execution

Use Python 3.13 and an isolated project environment.
The Linux requirements include TensorFlow CUDA libraries; the NVIDIA driver must be installed on the host.
`requirements.lock.txt` records the exact installed environment, while `requirements.txt` lists the direct dependencies.
`setup_gpu.sh` exposes the environment CUDA library directories to the dynamic loader, following the [TensorFlow GPU setup guidance](https://www.tensorflow.org/install/pip).

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
source ./setup_gpu.sh
.venv/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Run the integration sweep before the complete experiments.
It exercises every model and E0-E3 on a small subset, including both transfer phases.
Its results are isolated under `artifacts/smoke` and `results/smoke`.

```bash
SMOKE_TEST=1 .venv/bin/python run_experiments.py
.venv/bin/python run_experiments.py
```

The launcher records the source hashes and dataset fingerprint, starts a fresh process for every run and resumes by skipping completed runs.
An incompatible existing protocol stops execution rather than mixing experiments.
Training logs are saved beside each run's model.
Individual training scripts can also be run directly.

```bash
EXPERIMENT=E3 SEED=42 .venv/bin/python train_complex_cnn.py
```

Training writes `best.keras`, `history.csv`, `run.json` and validation predictions.
After every planned training run completes, `summarize_results.py` freezes the model/scenario with the best mean validation F1-defect across seeds.
Then `validate.py` evaluates the predeclared test rows and the summarizer produces the main table.
Test predictions do not feed back into model selection.

## Checks

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -q
.venv/bin/ruff check *.py tests/test_pipeline.py
```

The integration tests check non-destructive filtering, raw-pixel preservation, malformed images, duplicate leakage, sigmoid F1, safe model reload and memory-mapped batches.

## Outputs

`results/experiment_summary.csv` and `results/experiment_summary.md` contain the 20 model/scenario rows with test mean and sample standard deviation across three seeds.
`results/runs.csv` includes per-run metrics, parameter counts, epoch counts and training time.
`results/selection.json` records the validation-only selection.
`results/figures/` contains learning curves and confusion matrices for each seed.
Histories, predictions, metrics and summary tables are generated outputs and must not be edited manually.
No measured result is claimed until the complete run and test evaluation have finished.

## Dataset-specific research

[Islamy et al.](https://link.springer.com/chapter/10.1007/978-3-031-29078-7_28) report binary ResNet18 accuracy of 91.50% on balanced data and 90.87% on imbalanced data.
Their public abstract does not establish the balancing method or complete training protocol.
[Pereira Neto et al.](https://sol.sbc.org.br/index.php/wvc/article/download/27537/27349) report EfficientNetB4 test accuracy of 88.44% with augmentation on four-class USK-Coffee.
[Moreira's university thesis](https://repositorio.ufu.br/bitstream/123456789/43745/1/ArtificialIntelligenceasaService.pdf), Table 9, reports ResNet50 accuracy of 95.11% on the four-class task with five-fold validation.
These different tasks and protocols motivate the candidate backbones; their published percentages are not expected binary results for this project.
