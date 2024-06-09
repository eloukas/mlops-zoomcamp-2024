from typing import Tuple
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import pickle

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('Experiment1')
print('artifact uri:', mlflow.get_artifact_uri())

with mlflow.start_run():
    mlflow.log_param('SIZE', 100)        

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data: Tuple[DictVectorizer, LinearRegression], **kwargs):
    dv, lr = data

    with open('/home/src/mlops/homework_3_0/custom/dict_vectorizer', 'wb') as f_out:
        pickle.dump(dv, f_out)    
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, artifact_path="models")
        mlflow.log_param('intercept', lr.intercept_)
        mlflow.log_artifact('/home/src/mlops/homework_3_1/custom/dict_vectorizer')

    return dv, lr