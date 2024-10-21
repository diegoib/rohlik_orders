from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


class ClusterGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str], k:int, seed:int):
        
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        
        self.k = k
        self.seed = seed
        self.variables = variables
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.seed)
        
    def fit(self, X:pd.DataFrame, y=None):
        self.kmeans.fit(X[self.variables])
        return self
    
    def transform(self, X) -> pd.DataFrame:
        X["cluster"] = self.kmeans.predict(X[self.variables])
        return X
    

class ConsecutiveDays(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str):
        
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        
        self.variable = variable
        
    def fit(self, X:pd.DataFrame, y=None):
        self.mapping_days = {
            d: i for i, d in enumerate(pd.date_range(
                start=X[self.variable].min(),
                end=X[self.variable].max(), freq="D"
                ))
            }
        
        self.last_date = X[self.variable].max()
        self.last_day = self.mapping_days[self.last_date]
        
        return self
        
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        
        if X[self.variable].min() > self.last_date:
            X["cons_day"] = self.last_day + 1
        else:
            X["cons_day"] = X[self.variable].map(self.mapping_days)
        
        return X
 
    
class ConsecutiveWeeks(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str):

        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        
        self.variable = variable
        
    def fit(self, X:pd.DataFrame, y=None):
        self.mapping_weeks = {
            w: i for i, w in enumerate(
                (X[self.variable].dt.year*100 + X[self.variable].dt.isocalendar().week).unique()
            )}             
            
        self.last_date = (X[self.variable].dt.year*100 + X[self.variable].dt.isocalendar().week).max()
        self.last_week = self.mapping_weeks[self.last_date]
        
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        if (X[self.variable].dt.year*100 + X[self.variable].dt.isocalendar().week).min() > self.last_date:
            X["cons_week"] = self.last_week + 1
        else:
            X["cons_week"] = (X["date"].dt.year*100 + X["date"].dt.isocalendar().week).map(self.mapping_weeks)
            
        return X
    
    
class DateAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str, date_attrs: List[str], week_attr: bool):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        
        if not isinstance(date_attrs, list):
            raise ValueError("date_attrs should be a list")
        
        if not isinstance(week_attr, bool):
            raise ValueError("week_attr should be a boolean")
        
        self.variable = variable
        self.date_attrs = date_attrs
        self.week_attr = week_attr
    
    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        for attr_name in self.date_attrs:
            X[attr_name] = getattr(X[self.variable].dt, attr_name)
        
        if self.week_attr:
            X["weekofyear"] = X[self.variable].dt.isocalendar().week
            
        X['total_holidays_month'] = X.groupby(['year', 'month'])['holiday'].transform('sum')
        
        return X
       
        
class ShoppingIntensity(BaseEstimator, TransformerMixin):
    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        special_days = {
            'black_friday': ((X['day'] >= 22) & (X['day'] <= 28) & (X['month'] == 11) & (X['weekofyear'] == 4)),
            'cyber_monday': ((X['day'] >= 22) & (X['day'] <= 28) & (X['month'] == 11) & (X['weekofyear'] == 0)),
            'valentines_day': (X['day'] == 14) & (X['month'] == 2),
            'singles_day': (X['day'] == 11) & (X['month'] == 11),
            'christmas_eve': (X['day'] == 24) & (X['month'] == 12),
            'christmas_day': (X['day'] == 25) & (X['month'] == 12),
            'new_years_eve': (X['day'] == 31) & (X['month'] == 12),
            'new_years_day': (X['day'] == 1) & (X['month'] == 1),
            'boxing_day': (X['day'] == 26) & (X['month'] == 12),
            'easter_monday': (X['day'] == 1) & (X['month'] == 4),
            'summer_sales': X['month'].isin([7, 8]),
            'winter_sales': (X['month'] == 1) | ((X['day'] >= 27) & (X['month'] == 12)),
            'prime_day': (X['day'] == 15) & (X['month'] == 7),
            'green_monday': ((X['day'] >= 8) & (X['day'] <= 14) & (X['month'] == 12) & (X['weekofyear'] == 0)),
            'click_frenzy': ((X['day'] == 15) & (X['month'] == 5)) | ((X['day'] == 15) & (X['month'] == 11)),
            'orthodox_christmas': (X['day'] == 7) & (X['month'] == 1),
            'orthodox_new_year': (X['day'] == 14) & (X['month'] == 1),
            'orthodox_easter': (X['day'] == 15) & (X['month'] == 4),
            'orthodox_easter_monday': (X['day'] == 16) & (X['month'] == 4),
            'orthodox_pentecost': (X['day'] == 3) & (X['month'] == 6),
            'dormition_of_theotokos': (X['day'] == 15) & (X['month'] == 8),
            'nativity_of_theotokos': (X['day'] == 8) & (X['month'] == 9),
            'exaltation_of_the_cross': (X['day'] == 14) & (X['month'] == 9),
            'october_first': (X['day'] == 1) & (X['month'] == 10),
            'april_sixteenth': (X['day'] == 16) & (X['month'] == 4),
            'december_twenty_second': (X['day'] == 22) & (X['month'] == 12),
            'april_sixth': (X['day'] == 6) & (X['month'] == 4)
        }
        
        X['shopping_intensity'] = sum(condition.astype(int) for condition in special_days.values())
        
        return X
    
    
class CyclicDateAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, mapping: Dict[str, int]):
        if not isinstance(mapping, dict):
            raise ValueError("mapping should be a dictionary")
        
        self.mapping = mapping
        
    def fit(self, X:pd.DataFrame, y=None):
        return self
        
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        for var, t in self.mapping.items():
            X[f"{var}_sin"] = np.sin(2 * np.pi * X[var] / t)
            X[f"{var}_cos"] = np.cos(2 * np.pi * X[var] / t)
            
        return X 
    
    
class TransformOHE(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        self.variable = variable
        
    def fit(self, X:pd.DataFrame, y=None):
        
        self.categories = list(X[self.variable].unique())
        if np.nan in self.categories:
            self.categories.remove(np.nan)
        
        self.ohe = OneHotEncoder(
            categories=[self.categories],
            drop=None,
            sparse_output=False,
            handle_unknown="ignore",
        )
        self.ohe.fit(X[[self.variable]])
        
        return self  
        
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        ohe_encoded = pd.DataFrame(
            self.ohe.transform(X[[self.variable]]), 
            columns=self.ohe.get_feature_names_out([self.variable])
        )
        return pd.concat([X, ohe_encoded], axis=1)     
    

class ProximityHolidays(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        self.variable = variable
        
    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X[f"{self.variable}_before"] = X[self.variable].shift(1).fillna(0).astype(int)
        X[f"{self.variable}_after"] = X[self.variable].shift(-1).fillna(0).astype(int)
        return X        
    

class City2Country(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str, skip_hungary: bool=True):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        if not isinstance(skip_hungary, bool):
            raise ValueError("skip_hungary should be a boolean")
        
        self.variable = variable
        self.skip_hungary = skip_hungary
        
    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X["germany"] = np.where(X[self.variable].isin(["Munich_1", "Frankfurt_1"]), 1, 0)
        X["czech"] = np.where(X[self.variable].isin(["Brno_1", "Prague_1", "Prague_2", "Prague_3"]), 1, 0)
        if not self.skip_hungary:
            X["hungary"] = np.where(X[self.variable]=="Budapest_1", 1, 0)
        
        return X
        
        
class RowsFilter(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variable should be a list")
        
        self.variables = variables
        
    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        for v in self.variables:
            X = X[X[v] == 0]
            
        return X
    
    
class ColsDropper(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variable should be a list")
        
        self.variables = variables
        
    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.drop(self.variables, axis=1)
        
        return X
    
    
class ColsDropper(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variable should be a list")
        
        self.variables = variables
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(self.variables, axis=1)
        
        return X