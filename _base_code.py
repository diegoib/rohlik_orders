# ## **1. Imports**

import numpy as np
import pandas as pd

import catboost as ctb
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import warnings
from pathlib import Path

import wandb

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
warnings.filterwarnings('ignore')

# Config

pd.set_option('display.max_columns', None)

ROOT = Path("/kaggle/input/rohlik-orders-forecasting-challenge")
SEED = 42
FOLDS = 10

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb.api.key")

wandb.login(key=secret_value_0)

# ## **2. Data**

df_train = pd.read_csv(
    ROOT / "train.csv",
    parse_dates=["date"],
).drop("id", axis=1)

df_test = pd.read_csv(
    ROOT / "test.csv",
    parse_dates=["date"],
).drop("id", axis=1)

non_test_cols = set(df_train.columns).difference(set(df_test.columns))
non_test_cols.discard("orders")

print(non_test_cols)

# ## **3. Feature Engineering**

# **cluster**

kmeans = KMeans(n_clusters=3, random_state=SEED) 

df_train['cluster'] = kmeans.fit_predict(df_train[['holiday', 'school_holidays','winter_school_holidays']])
df_test['cluster'] = kmeans.predict(df_test[['holiday', 'school_holidays','winter_school_holidays']])

# **date features**

def consecutive_days(df_train, df_test):
    mapping_days = {
        d: i for i, d in enumerate(
            pd.date_range(start=df_train["date"].min(), end=df_test["date"].max(), freq="D")
        )}
    df_train["cons_day"] = df_train["date"].map(mapping_days)
    df_test["cons_day"] = df_test["date"].map(mapping_days)
    return df_train, df_test

def consecutive_weeks(df_train, df_test):
    df_all = pd.concat([df_train, df_test], axis=0)

    mapping_train_weeks = {
        w: i for i, w in enumerate(
            (df_all["date"].dt.year*100 + df_all["date"].dt.isocalendar().week).unique()
        )} 

    df_train["cons_week"] = (df_train["date"].dt.year*100 + df_train["date"].dt.isocalendar().week).map(mapping_train_weeks)
    df_test["cons_week"] = (df_test["date"].dt.year*100 + df_test["date"].dt.isocalendar().week).map(mapping_train_weeks)
    return df_train, df_test

df_train, df_test = consecutive_days(df_train, df_test)
df_train, df_test = consecutive_weeks(df_train, df_test)

def date_feats(df):
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
#     df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
#     df["is_month_start"] = df["date"].dt.is_month_start
#     df["is_month_end"] = df["date"].dt.is_month_end
#     df["is_quarter_start"] = df["date"].dt.is_quarter_start
    df["is_quarter_end"] = df["date"].dt.is_quarter_end
#     df["is_year_start"] = df["date"].dt.is_year_start
#     df["is_year_end"] = df["date"].dt.is_year_end

    df['total_holidays_month'] = df.groupby(['year', 'month'])['holiday'].transform('sum')
    return df.drop(["date"], axis=1)


df_train = date_feats(df_train)
df_test = date_feats(df_test)

def shop_intense_days(data):
    
    black_friday = ((data['day'] >= 22) & (data['day'] <= 28) & (data['month'] == 11) & (data['weekofyear'] == 4))
    cyber_monday = ((data['day'] >= 22) & (data['day'] <= 28) & (data['month'] == 11) & (data['weekofyear'] == 0))
    valentines_day = ((data['day'] == 14) & (data['month'] == 2))
    singles_day = ((data['day'] == 11) & (data['month'] == 11))
    christmas_eve = ((data['day'] == 24) & (data['month'] == 12))
    christmas_day = ((data['day'] == 25) & (data['month'] == 12))
    new_years_eve = ((data['day'] == 31) & (data['month'] == 12))
    new_years_day = ((data['day'] == 1) & (data['month'] == 1))
    boxing_day = ((data['day'] == 26) & (data['month'] == 12))
    easter_monday = ((data['day'] == 1) & (data['month'] == 4))  # placeholder
    summer_sales = (data['month'].isin([7, 8]))
    winter_sales = ((data['month'] == 1) | ((data['day'] >= 27) & (data['month'] == 12)))
    prime_day = ((data['day'] == 15) & (data['month'] == 7))  # placeholder
    green_monday = ((data['day'] >= 8) & (data['day'] <= 14) & (data['month'] == 12) & (data['weekofyear'] == 0))
    click_frenzy = (((data['day'] == 15) & (data['month'] == 5)) | ((data['day'] == 15) & (data['month'] == 11)))  # placeholder
    # Orthodox Christian events (using placeholder dates - these need to be adjusted yearly)
    orthodox_christmas = ((data['day'] == 7) & (data['month'] == 1))
    orthodox_new_year = ((data['day'] == 14) & (data['month'] == 1))
    orthodox_easter = ((data['day'] == 15) & (data['month'] == 4))  # placeholder
    orthodox_easter_monday = ((data['day'] == 16) & (data['month'] == 4))  # placeholder
    orthodox_pentecost = ((data['day'] == 3) & (data['month'] == 6))  # placeholder
    dormition_of_theotokos = ((data['day'] == 15) & (data['month'] == 8))
    nativity_of_theotokos = ((data['day'] == 8) & (data['month'] == 9))
    exaltation_of_the_cross = ((data['day'] == 14) & (data['month'] == 9))

    # # New dates added
    october_first = ((data['day'] == 1) & (data['month'] == 10))
    april_sixteenth = ((data['day'] == 16) & (data['month'] == 4))
    december_twenty_second = ((data['day'] == 22) & (data['month'] == 12))
    april_sixth = ((data['day'] == 6) & (data['month'] == 4))

    # Sum up all the boolean masks
    data['shopping_intensity'] = (
        black_friday.astype(int) +
        cyber_monday.astype(int) +
        valentines_day.astype(int) +
        singles_day.astype(int) +
        christmas_eve.astype(int) +
        christmas_day.astype(int) +
        new_years_eve.astype(int) +
        new_years_day.astype(int) +
        boxing_day.astype(int) +
        easter_monday.astype(int) +
        summer_sales.astype(int) +
        winter_sales.astype(int) +
        prime_day.astype(int) +
        green_monday.astype(int) +
        click_frenzy.astype(int) +
        # Orthodox events
        orthodox_christmas.astype(int) +
        orthodox_new_year.astype(int) +
        orthodox_easter.astype(int) +
        orthodox_easter_monday.astype(int) +
        orthodox_pentecost.astype(int) +
        dormition_of_theotokos.astype(int) +
        nativity_of_theotokos.astype(int) +
        exaltation_of_the_cross.astype(int)+
        # New dates added
        october_first.astype(int) +
        april_sixteenth.astype(int) +
        december_twenty_second.astype(int) +
        april_sixth.astype(int)
    )
    # # Create a boolean column for any shopping day
#     data['is_shopping_day'] = data['shopping_intensity'] > 0
    return data


df_train = shop_intense_days(df_train)
df_test = shop_intense_days(df_test)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T06:03:47.4931Z","iopub.execute_input":"2024-08-06T06:03:47.493443Z","iopub.status.idle":"2024-08-06T06:03:47.545414Z","shell.execute_reply.started":"2024-08-06T06:03:47.493414Z","shell.execute_reply":"2024-08-06T06:03:47.544246Z"}}
def create_sin_cos(df, var, t):
    df[f"{var}_sin"] = np.sin(2 * np.pi * df[var] / t)
    df[f"{var}_cos"] = np.cos(2 * np.pi * df[var] / t)
    return df.drop(var, axis=1)
    
def ciclic_feats(df, mapping=None):
    if mapping== None:
        mapping = {
            "month": 12,
            "dayofyear": 365,
            "day": 31,
            "weekofyear": 52,
            "dayofweek": 7,
            "quarter": 4,
        }

    for var, t in mapping.items():
        df = create_sin_cos(df, var, t)
    
    return df


df_train = ciclic_feats(df_train)
df_test = ciclic_feats(df_test)

# **holidays**

def transform_ohe(ohe, df, var):
    ohe_encoded = pd.DataFrame(
        ohe.transform(df[[var]]), 
        columns=ohe.get_feature_names_out([var])
    )
    return pd.concat([df, ohe_encoded], axis=1).drop(var, axis=1)

holidays = list(df_train["holiday_name"].unique())
holidays.remove(np.nan)

hol_ohe = OneHotEncoder(
    categories=[holidays],
    drop=None,
    sparse_output=False,
    handle_unknown="ignore",
)
hol_ohe.fit(df_train[["holiday_name"]])

df_train = transform_ohe(hol_ohe, df_train, "holiday_name")
df_test = transform_ohe(hol_ohe, df_test, "holiday_name")

# holiday before & after

def create_proximity_hols(df):
    df['holiday_before'] = df['holiday'].shift(1).fillna(0).astype(int)
    df['holiday_after'] = df['holiday'].shift(-1).fillna(0).astype(int)
    return df

df_train = create_proximity_hols(df_train)
df_test = create_proximity_hols(df_test)


def create_diff_hols(df_train, df_test):
    diff_cols = [c for c in df_train.columns if c not in df_test.columns]
    df_all = pd.concat([df_train, df_test], axis=0)
    
    for lag in [3, 7, 14]:
        df_all[f"holiday_lag_diff_{lag}"] = df_all.groupby('warehouse')['holiday'].diff(lag).fillna(0).astype(int)
    
    df_train = df_all[df_all["orders"].notnull()]
    df_test = df_all.loc[df_all["orders"].isnull()].drop(diff_cols, axis=1)
    
    return df_train, df_test

df_train, df_test = create_diff_hols(df_train, df_test)

# **warehouse**

# Dictionary mapping cities to their countries

def city2country(df, skip_hungary=True):
    df["germany"] = np.where(df["warehouse"].isin(["Munich_1", "Frankfurt_1"]), 1, 0)
    df["czech"] = np.where(df["warehouse"].isin(["Brno_1", "Prague_1", "Prague_2", "Prague_3"]), 1, 0)
    if skip_hungary is not True:
        df["hungary"] = np.where(df["warehouse"]=="Budapest_1", 1, 0)
    return df

df_train = city2country(df_train)
df_test = city2country(df_test)

# OHE warehouse

warehouses = list(df_train["warehouse"].unique())

wh_ohe = OneHotEncoder(
    categories=[warehouses],
    drop=None,
    sparse_output=False,
    handle_unknown="ignore",
)
wh_ohe.fit(df_train[["warehouse"]])

df_train = transform_ohe(wh_ohe, df_train, "warehouse")
df_test = transform_ohe(wh_ohe, df_test, "warehouse")


# ## **4. Feature Selection**

df_train[list(non_test_cols)].head()

cols_2_delete = ["snow", "precipitation", "user_activity_1", "user_activity_2", "mov_change"]

df_train = df_train.drop(cols_2_delete, axis=1)

print(f"Rows before: {df_train.shape[0]}")

other_cols = [c for c in non_test_cols if c not in cols_2_delete]
print(other_cols)

for c in other_cols:
    df_train = df_train[df_train[c] == 0]

df_train = df_train.drop(other_cols, axis=1)

print(f"Rows after: {df_train.shape[0]}")



df_train.head()
df_test.head()

# Columns Integrity

print("Cols in train not in test: {}".format(set(df_train.columns).difference(set(df_test.columns))))
print("Cols in test not in train: {}".format(set(df_test.columns).difference(set(df_train.columns))))

# Nulls

nulls = (df_train.isnull()).sum()
print(nulls[nulls > 0])

nulls = (df_test.isnull()).sum()
print(nulls[nulls > 0])

# Cols names

df_train.columns = [re.sub(" ", "_", c) for c in df_train.columns]
df_test.columns = [re.sub(" ", "_", c) for c in df_test.columns]


# ## **5. Model**

cv = GroupKFold(FOLDS)
cv_group = df_train["cons_week"].astype("str")

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T06:03:47.973978Z","iopub.execute_input":"2024-08-06T06:03:47.974336Z","iopub.status.idle":"2024-08-06T06:03:47.984585Z","shell.execute_reply.started":"2024-08-06T06:03:47.9743Z","shell.execute_reply":"2024-08-06T06:03:47.983455Z"}}
params = {
    "loss_function": "RMSE",
    "eval_metric": "MAPE",
    "iterations": 6000,
    "early_stopping_rounds": 100,
    "use_best_model": True,
    "random_state": SEED,
    "silent": True,
}

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T06:03:47.986348Z","iopub.execute_input":"2024-08-06T06:03:47.98677Z","iopub.status.idle":"2024-08-06T06:04:06.448476Z","shell.execute_reply.started":"2024-08-06T06:03:47.986729Z","shell.execute_reply":"2024-08-06T06:04:06.447307Z"}}
# Run description

run_notes = """
    catboost
"""

run = wandb.init(
    project = 'rohlik-orders',
    config = params,
    save_code = True,
    group = 'cv_model-lgbm',
    notes = run_notes,
)

label = df_train.pop("orders")

fitted_models = []
scores = []
oof = np.zeros(df_train.shape[0])

for idx_train, idx_valid in cv.split(df_train, groups=cv_group):
    
    X_train, y_train = df_train.iloc[idx_train], label.iloc[idx_train]
    X_valid, y_valid = df_train.iloc[idx_valid], label.iloc[idx_valid]
    
    model = ctb.CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)]
    )
    
    preds = model.predict(X_valid)
    oof[idx_valid] = preds
    scores.append(mean_absolute_percentage_error(y_valid, preds))
    fitted_models.append(model)    



print(f"CV scores: {[np.round(i, 4) for i in scores]}")
print(f"CV mean score: {np.mean(scores):.5f}")
print(f"CV CoefVar score: {np.mean(scores)/np.std(scores):.4f}")

wh_scores = {}

df_valid = pd.DataFrame({"orders": label, "preds": oof})

wh_cols = [c for c in df_train.columns if c.startswith("warehouse")]
df_valid["warehouse"] = (wh_ohe.inverse_transform(df_train[wh_cols])).reshape(-1)

for wh in df_valid["warehouse"]:
    wh_scores[f"cv {wh.lower()}"] = mean_absolute_percentage_error(
        df_valid.loc[df_valid["warehouse"] == wh, "orders"], 
        df_valid.loc[df_valid["warehouse"] == wh, "preds"]
    )

print("City CV scores: {}".format({k: np.round(v, 4) for k, v in wh_scores.items()}))

run.log({
    "cv mean score": np.mean(scores),
    "cv coefvar score": np.mean(scores)/np.std(scores),
    "folds": FOLDS,
    **wh_scores
})

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T06:06:29.182206Z","iopub.execute_input":"2024-08-06T06:06:29.182557Z","iopub.status.idle":"2024-08-06T06:06:33.942059Z","shell.execute_reply.started":"2024-08-06T06:06:29.182525Z","shell.execute_reply":"2024-08-06T06:06:33.940986Z"}}
run.finish()

# %% [markdown]
# ## **6. Predict & Submission**

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T06:06:33.943729Z","iopub.execute_input":"2024-08-06T06:06:33.944168Z","iopub.status.idle":"2024-08-06T06:06:34.045016Z","shell.execute_reply.started":"2024-08-06T06:06:33.944127Z","shell.execute_reply":"2024-08-06T06:06:34.043944Z"}}
preds = np.zeros(df_test.shape[0])

for f in range(FOLDS):
    preds += fitted_models[f].predict(df_test) / FOLDS

# ## **7. Plot results**

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T06:06:34.103125Z","iopub.execute_input":"2024-08-06T06:06:34.103505Z","iopub.status.idle":"2024-08-06T06:06:34.149063Z","shell.execute_reply.started":"2024-08-06T06:06:34.103474Z","shell.execute_reply":"2024-08-06T06:06:34.14789Z"}}
df_train = pd.read_csv(ROOT / "train.csv", parse_dates=["date"])
df_test = pd.read_csv(ROOT / "test.csv", parse_dates=["date"])
df_test["orders"] = preds

all_df = pd.concat([df_train, df_test], axis=0)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T06:06:34.15059Z","iopub.execute_input":"2024-08-06T06:06:34.151014Z","iopub.status.idle":"2024-08-06T06:06:34.162407Z","shell.execute_reply.started":"2024-08-06T06:06:34.150976Z","shell.execute_reply":"2024-08-06T06:06:34.161138Z"}}
cutoff = "2023-03"
all_df = all_df[all_df["date"] > cutoff]

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T06:06:34.163889Z","iopub.execute_input":"2024-08-06T06:06:34.164349Z","iopub.status.idle":"2024-08-06T06:06:35.119295Z","shell.execute_reply.started":"2024-08-06T06:06:34.164316Z","shell.execute_reply":"2024-08-06T06:06:35.118181Z"}}
fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(
    data=all_df,
    x="date",
    y="orders",
    hue="warehouse",
)
ax.axvline(df_test["date"].min(), ls="--", color="red")
plt.title("All warehouses")
plt.show()