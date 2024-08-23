from pathlib import Path
from typing import Dict, List, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import model

# Project Directories
PACKAGE_ROOT = Path(model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

