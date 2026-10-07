from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package='coffee')
class ResNet50Preprocessing(Layer):
    """The ImageNet ResNet50 input convention, saved with the model."""

    def call(self, images):
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(images)


class DataPreprocessor:
    DATA_DIR = Path(__file__).resolve().parent / 'data'
    SPLITS = ('train', 'val', 'test')
    NORMAL_CLASSES = ('longberry', 'peaberry', 'premium')
    DEFECT_CLASS = 'defect'
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
    DUPLICATE_PATHS = frozenset({'train/peaberry/14.jpg', 'test/peaberry/1916.jpg'})

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR

    def build_file_table(self):
        rows = []
        classes = (*self.NORMAL_CLASSES, self.DEFECT_CLASS)
        for split in self.SPLITS:
            split_dir = self.data_dir / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f'Missing split: {split_dir}')
            unexpected = {p.name for p in split_dir.iterdir() if p.is_dir()} - set(classes)
            if unexpected:
                raise ValueError(f'Unexpected classes in {split}: {unexpected}')
            for class_name in classes:
                class_dir = split_dir / class_name
                if not class_dir.is_dir():
                    raise FileNotFoundError(f'Missing class: {class_dir}')
                for path in sorted(class_dir.iterdir()):
                    if path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                        continue
                    relative_path = path.relative_to(self.data_dir).as_posix()
                    if relative_path in self.DUPLICATE_PATHS:
                        continue
                    rows.append({
                        'path': str(path), 'relative_path': relative_path,
                        'split': split, 'class_name': class_name,
                        'label': int(class_name == self.DEFECT_CLASS),
                        'sha256': sha256(path.read_bytes()).hexdigest(),
                    })
        table = pd.DataFrame(rows)
        if table.empty:
            raise ValueError('Dataset contains no images')
        duplicated = table[table.duplicated('sha256', keep=False)]
        if not duplicated.empty:
            raise ValueError('Exact duplicates require review: ' + ', '.join(duplicated.relative_path))
        for split in self.SPLITS:
            if set(table.loc[table.split == split, 'label']) != {0, 1}:
                raise ValueError(f'{split} must contain normal and defect images')
        return table

    @staticmethod
    def fingerprint(table):
        manifest = table[['relative_path', 'split', 'label', 'sha256']].to_csv(index=False)
        return sha256(manifest.encode()).hexdigest()

    @staticmethod
    def load_rgb(path):
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (256, 256):
                raise ValueError(f'Expected 256x256 RGB: {path}; got {image.size}, {image.mode}')
            return np.asarray(image, dtype=np.uint8)

    def load_split(self, table, split, memory_map=False):
        part = table[table.split == split]
        if part.empty:
            raise ValueError(f'Empty split: {split}')
        if memory_map:
            # ImageDataGenerator otherwise copies the entire uint8 dataset to float32.
            # A read-only disk-backed array keeps the same raw pixel values.
            cache_dir = Path(__file__).resolve().parent / 'artifacts' / 'data_cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f'{self.fingerprint(table)}_{split}.npy'
            if not cache_path.exists():
                import os
                temporary_path = cache_path.with_suffix(f'.{os.getpid()}.tmp')
                images = np.lib.format.open_memmap(
                    temporary_path, mode='w+', dtype=np.float32,
                    shape=(len(part), 256, 256, 3),
                )
                for index, path in enumerate(part.path):
                    images[index] = self.load_rgb(path)
                images.flush()
                del images
                temporary_path.replace(cache_path)
            images = np.load(cache_path, mmap_mode='r')
        else:
            images = np.stack([self.load_rgb(path) for path in part.path])
        labels = part.label.to_numpy(dtype=np.float32).reshape(-1, 1)
        return images, labels

    def preprocess(self):
        table = self.build_file_table()
        return tuple(self.load_split(table, split) for split in self.SPLITS)
