from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error


class VotingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, params: Dict[str, Any], folds: int, group_col: str, seed: int):
        self.params = params
        self.params.update({"random_state": seed})
        self.folds = folds
        self.group_col = group_col
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.scores = []
        self.fitted_models = []
        cv = GroupKFold(self.folds)
        
        for idx_train, idx_valid in cv.split(X, groups=X[self.group_col]):
    
            X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
            X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
            
            model = LGBMRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[early_stopping(100)]
            )
            
            preds = model.predict(X_valid)
            self.scores.append(mean_absolute_percentage_error(y_valid, preds))
            self.fitted_models.append(model) 
            
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = np.zeros(X.shape[0])

        for f in range(self.folds):
            preds += self.fitted_models[f].predict(X) / self.folds
            
        return preds
    
    def get_scores(self) -> Tuple[float, List[float]]:
        mean_cv = np.mean(self.scores)
        return mean_cv, self.scores
        