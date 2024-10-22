from datetime import datetime

from regression_model.config.core import config
from regression_model.processing.features import DateAttributes


def test_date_attributes_transformer(sample_input_data):
    # Given
    transformer = DateAttributes(
        variable=config.config_model.date_var, 
        date_attrs= config.config_model.date_attrs,
        week_attr=config.config_model.week_attr
    )
    assert sample_input_data["date"][:1] == datetime(2020, 12, 5)

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["day"][:1] == 5
    assert subject["month"][:1] == 12
    assert subject["year"][:1] == 2020
    assert subject["dayofweek"][:1] == 5
    assert subject["dayofyear"][:1] == 340
    assert subject["quarter"][:1] == 4
    assert subject["is_quarter_end"][:1] == False