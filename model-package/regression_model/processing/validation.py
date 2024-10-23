from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_score(score: int, threshold: float):
    """Check model performance against threshold"""
    if score <= threshold:
        return True
    return False


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    validated_data.dropna(subset=["date", "warehouse"], inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    try:
        input_data["date"] = pd.to_datetime(input_data["date"], errors="raise")
    except ValueError as err:
        raise ValueError(err, f"'Unable to parse date")

    validated_data = drop_na_inputs(input_data=input_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleOrderInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return input_data, errors


class OrderInputSchema(BaseModel):
    warehouse: str
    date: datetime
    holiday_name: Optional[str]
    holiday: Optional[int]
    shutdown: Optional[int]
    mini_shutdown: Optional[int]
    shops_closed: Optional[int]
    winter_school_holidays: Optional[int]
    school_holidays: Optional[int]
    blackout: Optional[int]
    mov_change: Optional[float]
    frankfurt_shutdown: Optional[int]
    precipitation: Optional[float]
    snow: Optional[float]
    user_activity_1: Optional[float]
    user_activity_2: Optional[float]


class MultipleOrderInputs(BaseModel):
    inputs: List[OrderInputSchema]
