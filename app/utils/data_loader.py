import configparser
from pathlib import Path
from typing import Generator, Tuple

import cv2
import yaml


def load_config(config_path: str = "app/core/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

