from typing import Generator

#import pandas as pd
import pytest
from fastapi.testclient import TestClient
from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset

from app.main import app


@pytest.fixture()
def sample_input_data():
    data = load_dataset(file_name=config.config_app.training_data_file)
    data = data.drop("orders", axis=1)
    data["date"] = data["date"].astype(str)
    return data[:10]


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
