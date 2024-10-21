import numpy as np
import pandas as pd
from regression_model.inference import make_prediction

df = pd.read_csv("model-package/data/train.csv")

X = df[-10:]
X = X.drop("orders", axis=1)

preds = make_prediction(input_data=X)

print(np.round(preds["predictions"]))
