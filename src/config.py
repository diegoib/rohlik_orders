from pathlib import Path

pd.set_option('display.max_columns', None)

ROOT = Path("../")
SEED = 42
FOLDS = 10

CICLIC_MAPPING = {
    "month": 12,
    "dayofyear": 365,
    "day": 31,
    "weekofyear": 52,
    "dayofweek": 7,
    "quarter": 4,
}

