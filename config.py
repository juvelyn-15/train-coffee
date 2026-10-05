"""Load human-editable TOML configurations for training and tuning."""

import tomllib
from pathlib import Path


CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(name):
    path = CONFIG_DIR / f"{name}.toml"
    with path.open("rb") as config_file:
        return tomllib.load(config_file)
