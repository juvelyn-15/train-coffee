# USK-Coffee preprocessing

The dataset is USK-Coffee, introduced by Febriana, Muchtar, Dawood and Lin in 2022.
The [authors' dataset page](https://coffee.comvislab-usk.org/) explicitly supports merging longberry, peaberry and premium into normal, and keeping defect as the second class.

| Split | Normal (0) | Defect (1) | Total |
|---|---:|---:|---:|
| Train | 3599 | 1200 | 4799 |
| Validation | 1200 | 400 | 1600 |
| Test | 1199 | 400 | 1599 |

These counts describe the local dataset after the two previously documented duplicate paths were removed.
The EDA notebook's stored outputs describe the original 8000 files and must not be treated as current counts.

## Cleaning and manifest

`DataPreprocessor.build_file_table()` preserves the existing train/val/test folders and creates a deterministic manifest containing paths, original classes, binary labels and SHA-256 hashes.
Unexpected or missing class folders, empty binary splits and exact duplicates stop the workflow for review.
`train/peaberry/14.jpg` and `test/peaberry/1916.jpg` are excluded if present, without deleting either file.
Both paths are absent in the current local dataset.
A manifest fingerprint identifies the exact files used by every experiment.
The experiment launcher saves that manifest and freezes its fingerprint before training.

The original notebook identified a visually similar train/test pair even though its file hashes differed.
SHA-256 detects exact file duplicates, not acquisition-level dependence or visually similar images with different encodings.
A 63-bit perceptual-hash scan found several similar-shape cross-split candidates.
The twelve closest candidate pairs were inspected visually; hash similarity alone did not justify excluding more images.
This screening is not an exhaustive near-duplicate or physical-bean identity audit.
No bean-instance identifiers are available to prove independence at the physical-bean level.

## Image and label contract

`load_split()` returns RGB `uint8` arrays of shape `(N, 256, 256, 3)` in `[0,255]` and float binary labels of shape `(N,1)`.
Images with a different size or color mode raise an explicit error.
All 7998 current files were decoded successfully as 256x256 RGB during inspection.

Training and evaluation request a read-only float32 memory map of the same raw pixel values.
This avoids ImageDataGenerator creating a second full float32 copy in RAM on this 16 GB machine.
Caches are keyed by the content manifest fingerprint and written atomically under `artifacts/data_cache/`.

Normalization is saved inside each model and applies to train, validation, test and inference.
Simple and complex CNNs use `Rescaling(1/255)`.
ResNet18 uses the image converter supplied with its ImageNet preset, with a 256x256 output size.
ResNet50 uses its ImageNet RGB-to-BGR and mean-subtraction convention through a serializable layer.
EfficientNetB4 already contains its input preprocessing.
The validator must pass raw RGB pixels and must not divide them by 255 again.

## Evaluation boundary

Augmentation and class weights affect training only.
Validation and test preserve the natural class distribution.
The four experiments use identical data files and deterministic ordering within each split.
Test evaluation follows validation selection and does not influence the selected configuration or threshold.
