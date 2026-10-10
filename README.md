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
The cleaned `data/` source directory has only `normal/` and `defect/` folders within each train/val/test split.
Normal filenames preserve the original class as a prefix.
Each model receives raw RGB 256x256 pixels and outputs a single sigmoid probability of defect.
All models use binary cross-entropy and a fixed 0.5 decision threshold.

| Model | Architecture | Initialization |
|---|---|---|
| Simple CNN | Conv16-Pool, Conv32-Pool, Conv64-Pool, GAP, Dense128, Dropout0.5, sigmoid | Random |
| Complex CNN | Four VGG blocks, two Conv-BN-ReLU sequences each, filters 32/64/128/256, MaxPool after each block with block dropout disabled; GAP-Dense128-Dropout0.5-sigmoid, with L2 regularization on convolutional layers and Dense128 | Random |
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

## EfficientNet regularization comparison

The original EfficientNetB4 protocol remains the `baseline` variant: Dropout 0.5, fine-tuning of `block7` for up to 40 epochs at learning rate 0.00001, and frozen batch normalization.
The `regularized` variant keeps the same data, head and fine-tuning stage, but uses Dropout 0.6, up to 20 fine-tuning epochs and learning rate 0.000005.
The controlled comparison uses E3 (augmentation and class weights enabled) with seeds 42, 43 and 44.
Run `python run_efficientnet_comparison.py`, then `python compare_efficientnet_e3.py` to select by mean validation F1-defect.
Run `python validate_efficientnet_e3.py` only after selection to evaluate the selected variant on the fixed test split.

## EfficientNetB4 v2

`train_efficientnet_b4_v2.py` is a standalone regularized version. It uses a BatchNormalization-Dense128-L2-Dropout head, AdamW, ReduceLROnPlateau and fine-tunes `block6+block7` with frozen backbone batch normalization.
The local runner trains E3 for seeds 42, 43 and 44, evaluates each best validation checkpoint on test, and writes mean/std summaries:

```bash
python run_efficientnet_b4_v2.py
```

The matching Kaggle notebook is `kaggle_efficientnet_transfer_learning_v2.ipynb`; it runs all three seeds in one notebook and saves outputs under `/kaggle/working/efficientnet_b4_v2_seed_<seed>/`.

## EfficientNetB4 v3 full experiment grid

Version 3 uses progressive fine-tuning: head, then `block7`, then `block6+block7`, with AdamW, L2 regularization, Dropout and ReduceLROnPlateau.
It runs all four E0-E3 scenarios and seeds 42, 43 and 44 (12 runs total):

```bash
python run_efficientnet_b4_v3.py
```

The local artifacts are saved under `artifacts/efficientnet_b4_v3/<experiment>/seed_<seed>/`.
The matching Kaggle notebook is `kaggle_efficientnet_transfer_learning_v3.ipynb`; its `/kaggle/working/efficientnet_b4_v3_output/` contains matching `artifacts/` and `results/` directories.
Validation summary selects one experiment before `evaluate_efficientnet_b4_v3.py` evaluates its three test runs.

## Setup and execution

Create an isolated environment with a Python version supported by TensorFlow on your platform.
Install the project dependencies from `requirements.txt`.
TensorFlow manages its Keras and serialization dependencies; CUDA libraries are not included in the project requirements.

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
```

Configure GPU drivers, acceleration packages and TensorFlow hardware settings locally according to the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).
The scripts do not set CPU thread counts or GPU memory allocation policy.
For example, Linux or WSL2 users with NVIDIA GPUs can install `tensorflow[and-cuda]` in their environment.
Seed and deterministic operations remain enabled for repeatable experiments.
Dependencies are resolved at installation time; record the installed versions alongside reported experiment results.

Run the full experiment grid: five models, four scenarios and three seeds.

```bash
python run_experiments.py
```

The fixed experiment grid, augmentation flags and class weights are defined in `experiments.py`.
Class weights use the verified training counts: 3599 normal and 1200 defect.
The launcher starts a fresh process for every run and skips completed runs.
The dataset is fixed and was checked once; training does not repeat data hashing, manifest generation or protocol checks.
There is no separate preprocessing command.
Training and evaluation call `data_loader.load_split()` directly and pass its arrays to `ImageDataGenerator.flow()`, following the supplied code templates.
The loader creates disposable float32 memory-map caches on first use, while normalization remains inside each model and augmentation runs during training.
Use a separate output directory when changing the experiment design.
Training logs are saved beside each run's model.
Individual training scripts can also be run directly.

```bash
EXPERIMENT=E3 SEED=42 python train_complex_cnn.py
```

Training writes `best.keras`, `history.csv`, `run.json` and validation predictions.
After every planned training run completes, `select_model.py` selects the model/scenario with the best mean validation F1-defect across seeds.
Then `validate.py` evaluates the predeclared test rows and `summarize_results.py` produces the main table.
The summarizer requires every test run to be complete; it does not infer the pipeline stage from existing files.
Test predictions do not feed back into model selection.

## Checks

```bash
python -m pip install pytest ruff
python -m pytest tests/test_pipeline.py -q
python -m ruff check *.py tests/test_pipeline.py
```

The integration tests check raw-pixel preservation, sigmoid F1, safe model reload, memory-mapped batches and validation-only selection through result summarization.

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
