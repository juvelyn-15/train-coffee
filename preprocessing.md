# USK-Coffee preprocessing

The dataset is USK-Coffee, introduced by Febriana, Muchtar, Dawood and Lin in 2022.
The [authors' dataset page](https://coffee.comvislab-usk.org/) explicitly supports merging longberry, peaberry and premium into normal, and keeping defect as the second class.

| Split | Normal (0) | Defect (1) | Total |
|---|---:|---:|---:|
| Train | 3599 | 1200 | 4799 |
| Validation | 1200 | 400 | 1600 |
| Test | 1199 | 400 | 1599 |

These counts describe the local dataset after the two previously documented duplicate paths were removed.
The EDA notebook is a historical analysis of the original four-class, 8000-file dataset.
Its paths, class names and stored counts do not describe the current binary directory layout.

## Fixed dataset

The dataset will not change or receive additional images.
A one-time inspection decoded all 7998 JPEG files successfully as 256x256 RGB.
No exact file duplicates or identical decoded pixel arrays were found.
The split counts are listed above.
The two previously removed paths, `train/peaberry/14.jpg` and `test/peaberry/1916.jpg`, are absent.

The cleaned `data/` directory is the source of truth and is already organized for binary classification.
Within each original split, longberry, peaberry and premium were moved into `normal/`.
Normal filenames retain the original class as a prefix, such as `longberry_1.jpg`, to avoid collisions and preserve provenance.
The `defect/` directories were kept as they were.
A one-time SHA-256 comparison confirmed that all 7998 files remained byte-identical and in their original split after moving 5998 normal images.
The three former normal-class directories were removed after migration.

`data_loader.load_split(split)` reads only the requested split, with normal=0 and defect=1.
It sorts filenames within each class, retaining the original normal-class ordering through the filename prefixes.
There is no preprocessing command, class-merging logic or `build_file_table()` in the training pipeline.
Training does not repeat folder, duplicate or image-shape validation.
Dataset fingerprints, exported manifests and protocol locks are unnecessary for this fixed dataset and are not generated.

The original notebook identified a visually similar train/test pair even though its file hashes differed.
SHA-256 detects exact file duplicates, not acquisition-level dependence or visually similar images with different encodings.
A 63-bit perceptual-hash scan found several similar-shape cross-split candidates.
The twelve closest candidate pairs were inspected visually; hash similarity alone did not justify excluding more images.
This screening is not an exhaustive near-duplicate or physical-bean identity audit.
No bean-instance identifiers are available to prove independence at the physical-bean level.

## Image and label contract

`load_split()` returns `(images, labels, metadata)`.
Images are a read-only float32 memory map of shape `(N, 256, 256, 3)`, containing raw RGB values in `[0,255]`.
Labels are float32 binary values of shape `(N,1)`.
Metadata contains the relative path, binary class name and label in exactly the same order as the images.
It is used to associate validation and test predictions with filenames; it is not an intermediate table used to load images.

The float32 conversion does not normalize pixel values.
This avoids ImageDataGenerator creating a second full float32 copy in RAM.
Caches are named `train.npy`, `val.npy` and `test.npy` and written atomically under `artifacts/data_cache/`.
The loader creates each cache on first use and reuses it afterwards.
Caches are disposable loading aids, not preprocessed source data.
Obsolete caches were deleted during the directory migration.
If the fixed dataset is deliberately changed in future, delete its caches before loading it again.
Temporary test datasets keep their caches under their own dataset directory.
The loader follows the supplied templates by providing arrays to `ImageDataGenerator.flow()`.
It does not one-hot encode labels because the models use one sigmoid output and binary cross-entropy.

Normalization is saved inside each model and applies to train, validation, test and inference.
Simple and complex CNNs use `Rescaling(1/255)`.
ResNet18 uses the image converter supplied with its ImageNet preset, with a 256x256 output size.
ResNet50 uses its ImageNet RGB-to-BGR and mean-subtraction convention through the serializable layer in `model_inputs.py`.
EfficientNetB4 already contains its input preprocessing.
The validator must pass raw RGB pixels and must not divide them by 255 again.

## Evaluation boundary

Augmentation and class weights affect training only.
Validation and test preserve the natural class distribution.
The four experiments use identical data files and deterministic ordering within each split.
Training counts and inverse-frequency class weights are declared once in `experiments.py`, using the verified fixed dataset.
Changing this dataset requires updating these counts as well as deleting the caches.
Run `select_model.py`, then `validate.py`, then `summarize_results.py`.
Test evaluation follows validation selection and does not influence the selected configuration or threshold.

## Existing outputs

Previously generated prediction CSVs retain their original four-class filenames and class names.
They are historical outputs and were not edited during this refactor.
New runs export the binary directory paths and `normal`/`defect` class names.
Existing completed runs are still skipped by the launcher.
Use a fresh run directory if predictions with the new paths are required.

## Refactor verification

All 7998 source files were checked against their pre-migration SHA-256 values.
The new loader was exercised on all three real splits, and every cached image was compared with its decoded source JPEG.
Image dimensions, RGB mode, class counts and metadata-label alignment matched the fixed dataset contract.
All ten pipeline tests and the project's Ruff checks passed.
A smoke run executed the simple-CNN training script for one epoch using four real training images and four validation images, then saved and reloaded its model.
The unmodified `validate.py` evaluated that model in a fresh process on all 1599 test images and exported correctly aligned binary filenames and labels.
This smoke run verifies the pipeline wiring and does not constitute a model-quality experiment or a rerun of the 60-run experiment grid.

## Fixed-protocol simplification verification

The current counts were checked against all three dataset folders, and all 7998 image contents were compared with their Git blobs before committing the directory migration.
The shared E0-E3 configuration preserves the previous augmentation flags and inverse-frequency class weights exactly.
The simplified metric calculation matched the previous implementation on finite sigmoid predictions; NaN and infinity are still rejected.
Each ImageNet backbone was loaded to confirm that its declared final stage has trainable weights while batch normalization stays frozen.
Repeat this development check when changing the backbone architecture or its dependencies.
An E3 smoke run trained the simple CNN for one epoch on four real images, saved and reloaded it, then evaluated all 1599 real test images in a fresh process.
The selection/summary integration test uses synthetic predictions to confirm that test scores cannot change validation selection and missing test files stop summarization.
These checks verify execution and protocol preservation; they do not rerun the complete 60-run experiment grid.
