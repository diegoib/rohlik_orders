from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleOrderInputs(
            inputs=input_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return input_data, errors


class OrderInputSchema(BaseModel):
    warehouse: Optional[str]        
    date: Optional[str]
    orders: Optional[float]       
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