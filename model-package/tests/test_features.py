from datetime import datetime

from regression_model.config.core import config
from regression_model.processing.features import DateAttributes


def test_date_attributes_transformer(sample_input_data):
    # Given
    transformer = DateAttributes(
        variable=config.config_model.date_var,
        date_attrs=config.config_model.date_attrs,
        week_attr=config.config_model.week_attr,
    )
    assert sample_input_data["date"].iat[0] == datetime(2020, 12, 5)

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["day"].iat[0] == 5
    assert subject["month"].iat[0] == 12
    assert subject["year"].iat[0] == 2020
    assert subject["dayofweek"].iat[0] == 5
    assert subject["dayofyear"].iat[0] == 340
    assert subject["quarter"].iat[0] == 4
    assert subject["is_quarter_end"].iat[0] == False  # noqa: E712
