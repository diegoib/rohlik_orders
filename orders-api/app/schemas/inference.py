from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel
from regression_model.processing.validation import OrderInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleOrderInputs(BaseModel):
    inputs: List[OrderInputSchema]

    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [
                    {
                        "warehouse": "Prague_1",
                        "date": "2020-12-05",
                        "holiday_name": None,
                        "holiday": 0,
                        "shutdown": 0,
                        "mini_shutdown": 0,
                        "shops_closed": 0,
                        "winter_school_holidays": 0,
                        "school_holidays": 0,
                        "blackout": 0,
                        "mov_change": 0,
                        "frankfurt_shutdown": 0,
                        "precipitation": 0,
                        "snow": 0,
                        "user_activity_1": 1722,
                        "user_activity_2": 32575,
                        "id": "Prague_1_2020-12-05",
                    }
                ]
            }
        }
