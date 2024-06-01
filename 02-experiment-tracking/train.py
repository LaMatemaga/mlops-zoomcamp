import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc_taxi_experiment")
    mlflow.sklearn.autolog()
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))  
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))


    with mlflow.start_run():

        mlflow.set_tag("developer", "cynthia")
        mlflow.set_tag("autolog", "true")

        mlflow.log_param("train-data-path", "./data/green_tripdata_2023-01.parquet")
        mlflow.log_param("valid-data-path", "./data/green_tripdata_2023-02.parquet")
        mlflow.log_param("test-data-path",  "./data/green_tripdata_2023-03.parquet")

        max_depth = 10
        random_state = 0
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

if __name__ == '__main__':
    run_train()