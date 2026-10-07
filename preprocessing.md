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

## Fixed dataset

The dataset will not change or receive additional images.
A one-time inspection decoded all 7998 JPEG files successfully as 256x256 RGB.
No exact file duplicates or identical decoded pixel arrays were found.
The split counts are listed above.
The two previously removed paths, `train/peaberry/14.jpg` and `test/peaberry/1916.jpg`, are absent.

`DataPreprocessor.build_file_table()` lists the existing JPEG files in deterministic order and maps defect to 1 and the other three classes to 0.
It does not repeat folder, class, duplicate or image-shape validation during training.
Dataset fingerprints, exported manifests and protocol locks are unnecessary for this fixed dataset and are no longer generated.

The original notebook identified a visually similar train/test pair even though its file hashes differed.
SHA-256 detects exact file duplicates, not acquisition-level dependence or visually similar images with different encodings.
A 63-bit perceptual-hash scan found several similar-shape cross-split candidates.
The twelve closest candidate pairs were inspected visually; hash similarity alone did not justify excluding more images.
This screening is not an exhaustive near-duplicate or physical-bean identity audit.
No bean-instance identifiers are available to prove independence at the physical-bean level.

## Image and label contract

`load_split()` returns RGB `uint8` arrays of shape `(N, 256, 256, 3)` in `[0,255]` and float binary labels of shape `(N,1)`.

Training and evaluation request a read-only float32 memory map of the same raw pixel values.
This avoids ImageDataGenerator creating a second full float32 copy in RAM.
Caches are named by split and row count and written atomically under `artifacts/data_cache/`.
Existing hash-named caches are ignored; the first run creates the new caches.
Temporary test datasets keep their caches under their own dataset directory.

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
