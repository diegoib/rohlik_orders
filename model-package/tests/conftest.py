import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    data = load_dataset(file_name=config.config_app.training_data_file)
    data = data.drop("orders", axis=1)
    return data[:10]