from pathlib import Path
from typing import Dict, List, Any

from pydantic import BaseModel, field_validator
from strictyaml import YAML, load

import regression_model

# Project Directories
PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config
    """
    package_name: str
    training_data_file: str
    pipeline_save_file: str


class ModelParams(BaseModel):
    objective: str
    metric: str
    boosting_type: str
    n_estimators: int
    verbosity: int
    n_jobs: int


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    trainig and feature engineering.
    """
    label: str
    seed: int
    folds: int
    k: int
    skip_hungary: bool
    group_col: str
    cluster_vars: List[str]
    date_var: str
    date_attrs: List[str]
    week_attr: bool
    cyclic_mapping: Dict[str, int]
    holiday_var: str
    holiday_name_var: str
    warehouse_var: str
    filter_cols: List[str]
    drop_cols: List[str]
    params_model: Dict[str, Any]
    score_threshold: float
 
    @field_validator('params_model')
    @classmethod
    def validate_params_model(cls, v):
        expected_types = {
            'objective': str,
            'metric': str,
            'boosting_type': str,
            'n_estimators': int,
            'verbosity': int,
            'n_jobs': int
        }
        
        for key, expected_type in expected_types.items():
            if key in v:
                try:
                    v[key] = expected_type(v[key])
                except ValueError:
                    raise ValueError(f"'{key}' debe ser de tipo {expected_type.__name__}")
        
        return v


class Config(BaseModel):
    """Master config object."""

    config_app: AppConfig
    config_model: ModelConfig
    
    
def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        config_app=AppConfig(**parsed_config.data),
        config_model=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()