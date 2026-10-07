# Augmentation and imbalance experiments

The five approved models use the same four training scenarios.

| Experiment | Augmentation | Class weights |
|---|---|---|
| E0 | Off | Off |
| E1 | On | Off |
| E2 | Off | On |
| E3 | On | On |

Each training script declares `ImageDataGenerator` directly, following the supplied Python templates.
There is no shared augmentation class or hidden configuration file.
The generator receives raw RGB pixels in `[0,255]`; normalization belongs to the model.
No offline augmented image files are created.
Validation and test receive no random transformations.

## Approved augmentation

| Setting | Value |
|---|---|
| Rotation | -15 to +15 degrees |
| Horizontal and vertical shifts | Up to 5% |
| Horizontal and vertical flips | Enabled |
| Brightness multiplier | 0.9 to 1.1 |
| Border fill | nearest |

The [original authors' public code](https://github.com/cvitlab/USK-COFFEE-DATASET-A-multi-class-dataset-composed-of-the-various-green-bean-arabica/blob/main/USK_Coffee_Code.ipynb) uses horizontal flipping during training.
[Pereira Neto et al.](https://sol.sbc.org.br/index.php/wvc/article/download/27537/27349) evaluate augmentation on four-class USK-Coffee and include flipping, rotation, crop, color changes and random erasing.
[Izza and Kusuma](https://ijettjournal.org/Volume-72/Issue-6/IJETT-V72I6P128.pdf) report flips, contrast changes and rotation for transformer training.
These sources motivate candidate transformations, but do not prove this exact augmentation configuration is optimal for binary classification.
The 5% shifts and 0.9-1.1 brightness factors are project design choices.
Crop, erasing, strong color shifts and synthesis are outside the approved main experiment matrix.
Training-image previews must confirm that transformations retain meaningful defect regions before full training begins.

## Class weights

Weights are calculated from the actual training labels as `N/(2*n_class)`.
The full dataset therefore assigns approximately 0.667 to normal and 2.000 to defect.
Class weights change training loss contributions without discarding majority-class images.
They do not change the validation/test distribution or weight the reported evaluation metrics.
Ordinary augmentation applies to both classes and does not balance class counts.
Oversampling, focal loss and synthetic-image generation are not included in the approved main matrix.

## Reproducibility

Each experiment starts a fresh model with seed 42, 43 or 44.
Model initialization, random generators and TensorFlow operations share the selected seed.
TensorFlow deterministic operations are enabled.
Three-seed standard deviation measures training variability on the same fixed split, not uncertainty across new datasets.
