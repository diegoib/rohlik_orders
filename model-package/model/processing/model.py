from typing import Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error


class VotingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, params: Dict[str, Any], folds: int, group_col: str):
        self.params = params
        self.fitted_models = []
        self.scores = []
        self.folds = folds
        self.group_col = group_col
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        
        cv = GroupKFold(self.folds)
        
        for idx_train, idx_valid in cv.split(X, groups=X[self.group_col]):
    
            X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
            X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
            
            model = CatBoostRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)]
            )
            
            preds = model.predict(X_valid)
            self.scores.append(mean_absolute_percentage_error(y_valid, preds))
            self.fitted_models.append(model) 
            
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = np.zeros(X.shape[0])

        for f in range(self.folds):
            preds += self.fitted_models[f].predict(X) / self.folds
            
        return preds