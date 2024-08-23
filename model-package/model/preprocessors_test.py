import pandas as pd
from sklearn.pipeline import Pipeline

from .preprocessors import (
    ClusterGrouper,
    ConsecutiveDays,
    ConsecutiveWeeks,
    DateAttributes,
    ShoppingIntensity,
    CyclicDateAttributes,
    TransformOHE,
    ProximityHolidays,
    City2Country,
    RowsFilter,
    ColsDropper,
)

from .config import (
    HOLIDAYS_VAR,
    WAREHOUSE_VAR,
    DATE_VAR,
    CLUSTER_VARS,
    DATE_ATTRS,
    CYCLIC_MAPPING,
    FILTER_COLS,
    DROP_COLS,
    SEED,
    K,
    SKIP_HUNGARY,
    ROOT,
)

data = pd.read_csv(
    ROOT / "train.csv",
    parse_dates=[DATE_VAR],
)

pipe = Pipeline([
    ("ClusterGrouper", ClusterGrouper(CLUSTER_VARS, K, SEED)),
    ("ConsecutiveDays", ConsecutiveDays(DATE_VAR)),
    ("ConsecutiveWeeks", ConsecutiveWeeks(DATE_VAR)),
    ("DateAttributes", DateAttributes(DATE_VAR, DATE_ATTRS)),
    ("CyclicDateAttributes", CyclicDateAttributes(CYCLIC_MAPPING)),
    ("ShoppingIntensity", ShoppingIntensity()),
    ("ProximityHolidays", ProximityHolidays(HOLIDAYS_VAR)),
    ("City2Country", City2Country(WAREHOUSE_VAR, SKIP_HUNGARY)),
    ("TransformOHE", TransformOHE(HOLIDAYS_VAR)),
    ("TransformOHE", TransformOHE(WAREHOUSE_VAR)),
    ("RowsFilter", RowsFilter(FILTER_COLS)),
    ("ColsDropper", ColsDropper(DROP_COLS))   
])

df_train = pipe.fit_transform(data)

print(df_train.head(10))
print(df_train.info())


