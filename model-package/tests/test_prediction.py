import math

import numpy as np

from regression_model.inference import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 6895
    expected_no_predictions = 10

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=1000)
