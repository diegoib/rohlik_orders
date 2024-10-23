from datetime import datetime, timedelta

from config.core import config
from pipeline import orders_pipe
from processing.data_manager import load_dataset, save_pipeline
from processing.validation import validate_score


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.config_app.training_data_file)

    now = datetime.now()
    datecut = now - timedelta(days=365)
    X_train = data[data[config.config_model.date_var] < datecut]

    # divide X and y
    y_train = X_train.pop(config.config_model.label)

    # fit model
    orders_pipe.fit(X_train, y_train)

    # persist trained model
    cv_score, _ = orders_pipe.named_steps["Model"].get_scores()
    validation = validate_score(cv_score, config.config_model.score_threshold)

    if validation:
        save_pipeline(pipeline_to_persist=orders_pipe)


if __name__ == "__main__":
    run_training()
