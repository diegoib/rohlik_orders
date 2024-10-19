from sklearn.pipeline import Pipeline

from regression_model.config.core import config
from regression_model.processing import model
from regression_model.processing import features as pp


orders_pipe = Pipeline(
    [
        (
            "ClusterGrouper", 
            pp.ClusterGrouper(
                variables=config.config_model.cluster_vars, 
                k=config.config_model.k, 
                seed=config.config_model.seed
            )
        ),
        (
            "ConsecutiveDays", 
            pp.ConsecutiveDays(
                variable=config.config_model.date_var
            )
        ),
        (
            "ConsecutiveWeeks", 
            pp.ConsecutiveWeeks(
                variable=config.config_model.date_var
            )
        ),
        (
            "DateAttributes", 
            pp.DateAttributes(
                variable=config.config_model.date_var, 
                date_attrs= config.config_model.date_attrs,
                week_attr=config.config_model.week_attr
            )
        ),
        (
            "CyclicDateAttributes", 
            pp.CyclicDateAttributes(
                mapping=config.config_model.cyclic_mapping
            )
        ),
        (
            "ShoppingIntensity", 
            pp.ShoppingIntensity()
        ),
        (
            "ProximityHolidays", 
            pp.ProximityHolidays(
                variable=config.config_model.holiday_var
            )
        ),
        (
            "City2Country", 
            pp.City2Country(
                variable=config.config_model.warehouse_var, 
                skip_hungary=config.config_model.skip_hungary
            )
        ),
        (
            "TransformOHE_Holiday", 
            pp.TransformOHE(
                variable=config.config_model.holiday_name_var
            )
        ),
        (
            "TransformOHE_Warehouse", 
            pp.TransformOHE(
                variable=config.config_model.warehouse_var
            )
        ),
        (
            "RowsFilter", 
            pp.RowsFilter(
                variables=config.config_model.filter_cols
            )
        ),
        (
            "ColsDropper", 
            pp.ColsDropper(
                variables=config.config_model.drop_cols
            )
        ),
        (
            "LGBMModel",
            model.VotingRegressor(
                params=config.config_model.params_model,
                folds=config.config_model.folds,
                group_col=config.config_model.group_col,
                seed=config.config_model.seed,
                
            )
        ),
    ]
)

