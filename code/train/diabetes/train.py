import os
import warnings
import sys

import pandas as pd
import numpy as np

import azureml.core
from azureml.core import Dataset, Run

import mlflow
import mlflow.sklearn

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib



import lightgbm as lgb


# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    run = Run.get_context()

    base_path = run.input_datasets['data']
    # final_df = pd.read_csv(base_path)
    final_df = base_path.to_pandas_dataframe()

    df = final_df
    x_df = df
    y_df = x_df.pop("Y") # target

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)
  
    # Create a run object in the experiment
    # run =  experiment.start_logging()
    with mlflow.start_run():

        # Log the algorithm parameters to the run
        # run.log('num_leaves', 31)
        # run.log('learning_rate', 0.05)
        # run.log('n_estimators', 20)

        num_leaves=31
        learning_rate=0.01
        n_estimators=20

        mlflow.log_param("num_leaves",num_leaves)
        mlflow.log_param("learning_rate",learning_rate)
        mlflow.log_param("n_estimators",n_estimators)

        # setup model, train and test
        gbm = lgb.LGBMRegressor(num_leaves=num_leaves,
                                learning_rate=learning_rate,
                                n_estimators=n_estimators)
        model_gbm = gbm.fit(x_train, y_train,
                eval_set=[(x_test, y_test)],
                eval_metric='l1',
                early_stopping_rounds=5)

        preds = model_gbm.predict(x_test)

        # Output the Mean Squared Error to the notebook and to the run
        print('Mean Squared Error is', mean_squared_error(y_test, preds))
        # run.log('mse', mean_squared_error(y_test, preds))
        mlflow.log_metric('mse', mean_squared_error(y_test, preds))

        #     # Save the model to the outputs directory for capture
        #     model_file_name = './outputs/model.pkl'

        #     joblib.dump(value = model_gbm, filename = model_file_name)

        mlflow.sklearn.log_model(model_gbm, "gbm_model")

        # upload the model file explicitly into artifacts 
        # run.upload_file(name = model_file_name, path_or_stream = model_file_name)
