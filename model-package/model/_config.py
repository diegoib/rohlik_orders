from pathlib import Path

DATA_PATH = Path("data")
SEED = 42
FOLDS = 10
K = 3
SKIP_HUNGARY = True

CLUSTER_VARS = [
    "holiday", 
    "school_holidays", 
    "winter_school_holidays",
]

DATE_VAR = "date"

DATE_ATTRS = [
    ('day', 'day'),
    ('month', 'month'),
    ('year', 'year'),
    ('dayofweek', 'dayofweek'),
    ('dayofyear', 'dayofyear'),
    ('weekofyear', lambda x: x.isocalendar().week),
    ('quarter', 'quarter'),
    ('is_quarter_end', 'is_quarter_end')    
]

CYCLIC_MAPPING = {
    "month": 12,
    "dayofyear": 365,
    "day": 31,
    "weekofyear": 52,
    "dayofweek": 7,
    "quarter": 4,
}

HOLIDAY_VAR = "holiday"
HOLIDAY_NAME_VAR = "holiday_name"
WAREHOUSE_VAR = "warehouse"

FILTER_COLS = [
    "mini_shutdown", 
    "blackout", 
    "frankfurt_shutdown", 
    "shutdown",
]

DROP_COLS = [
    'snow', 
    'mini_shutdown', 
    'user_activity_2', 
    'user_activity_1', 
    'mov_change', 
    'blackout', 
    'precipitation', 
    'frankfurt_shutdown',
    'shutdown',
    "id",
    "date",
    "holiday_name",
    "warehouse",
    "month",
    "dayofyear",
    "day",
    "weekofyear",
    "dayofweek",
    "quarter",
]