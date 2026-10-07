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

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR

    def build_file_table(self):
        rows = []
        for split in self.SPLITS:
            for class_name in (*self.NORMAL_CLASSES, self.DEFECT_CLASS):
                for path in sorted((self.data_dir / split / class_name).glob('*.jpg')):
                    rows.append({
                        'path': str(path),
                        'relative_path': path.relative_to(self.data_dir).as_posix(),
                        'split': split, 'class_name': class_name,
                        'label': int(class_name == self.DEFECT_CLASS),
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def load_rgb(path):
        with Image.open(path) as image:
            return np.asarray(image, dtype=np.uint8)

    def load_split(self, table, split, memory_map=False):
        part = table[table.split == split]
        if memory_map:
            # ImageDataGenerator otherwise copies the entire uint8 dataset to float32.
            # A read-only disk-backed array keeps the same raw pixel values.
            cache_root = Path(__file__).resolve().parent / 'artifacts' if self.data_dir == self.DATA_DIR else self.data_dir
            cache_dir = cache_root / 'data_cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            # The dataset is fixed; the row count separates full and smoke subsets.
            cache_path = cache_dir / f'{split}_{len(part)}.npy'
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
