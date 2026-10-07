"""Load the fixed normal/defect dataset as arrays for ImageDataGenerator.flow()."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / 'data'
CLASSES = ('normal', 'defect')  # Binary labels: normal=0, defect=1.


def load_split(split, data_dir=DATA_DIR):
    """Return raw float32 RGB pixels, (N, 1) labels and aligned image metadata."""
    data_dir = Path(data_dir)
    paths = []
    labels = []
    for label, class_name in enumerate(CLASSES):
        files = sorted((data_dir / split / class_name).glob('*.jpg'))
        paths.extend(files)
        labels.extend([label] * len(files))

    # Disk-backed float32 avoids ImageDataGenerator copying the full dataset.
    cache_root = DATA_DIR.parent / 'artifacts' if data_dir == DATA_DIR else data_dir
    cache_dir = cache_root / 'data_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{split}.npy'
    if not cache_path.exists():
        temporary_path = cache_path.with_suffix(f'.{os.getpid()}.tmp')
        images = np.lib.format.open_memmap(
            temporary_path, mode='w+', dtype=np.float32,
            shape=(len(paths), 256, 256, 3),
        )
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                images[index] = np.asarray(image, dtype=np.uint8)
        images.flush()
        del images
        temporary_path.replace(cache_path)

    images = np.load(cache_path, mmap_mode='r')
    metadata = pd.DataFrame({
        'relative_path': [path.relative_to(data_dir).as_posix() for path in paths],
        'class_name': [path.parent.name for path in paths],
        'label': labels,
    })
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    return images, labels, metadata
