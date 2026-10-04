from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class DataPreprocessor:
    DATA_DIR = Path("data")
    SPLITS = ("train", "val", "test")
    NORMAL_CLASSES = ("longberry", "peaberry", "premium")
    DEFECT_CLASS = "defect"
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
    DUPLICATE_PATHS = {
        "data/train/peaberry/14.jpg",
        "data/test/peaberry/1916.jpg",
    }

    def __init__(self):
        self.data_dir = self.DATA_DIR

    def remove_duplicates(self):
        for path in self.DUPLICATE_PATHS:
            path = Path(path)

            if path.exists():
                path.unlink()

    def build_file_table(self):
        known_classes = set(self.NORMAL_CLASSES) | {self.DEFECT_CLASS}
        rows = []

        for split in self.SPLITS:
            split_dir = self.data_dir / split

            if not split_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing split folder: {split_dir}"
                )

            for class_dir in sorted(
                p for p in split_dir.iterdir() if p.is_dir()
            ):
                class_name = class_dir.name.lower()

                if class_name not in known_classes:
                    raise ValueError(
                        f"Unexpected class folder '{class_dir.name}' "
                        f"in {split_dir}. "
                        f"Expected one of {sorted(known_classes)}"
                    )

                for path in sorted(class_dir.iterdir()):
                    if path.suffix.lower() in self.IMAGE_EXTENSIONS:
                        rows.append(
                            {
                                "path": str(path),
                                "split": split,
                                "class_name": class_name,
                                "label": int(
                                    class_name == self.DEFECT_CLASS
                                ),
                            }
                        )

        if not rows:
            raise FileNotFoundError(
                f"No images found under {self.data_dir}"
            )

        return pd.DataFrame(rows)

    @staticmethod
    def load_rgb(path):
        with Image.open(path) as image:
            return np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

    def load_split(self, df, split):
        part = df[df["split"] == split]

        X = np.stack([
            self.load_rgb(path)
            for path in part["path"]
        ])

        y = part["label"].to_numpy()

        return X, y

    def preprocess(self):
        self.remove_duplicates()

        df = self.build_file_table()

        train = self.load_split(df, "train")
        val = self.load_split(df, "val")
        test = self.load_split(df, "test")

        return train, val, test