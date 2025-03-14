import sqlite3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from model.TaxiModel import TaxiModel

import common
import os, pickle


DB_PATH = os.path.join('data', 'taxis.db') 
MODEL_PATH = common.CONFIG['paths']['model_path']

def load_train_data():
    print(f"Reading train data from the database: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    X = data_train.drop(columns=['trip_duration'])
    y = data_train['trip_duration']
    return X, y

def train_model(X, y):
    print(f"Building a model")
    num_features = ['abnormal_period', 'hour']
    cat_features = ['weekday', 'month']
    
    column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
    ('scaling', StandardScaler(), num_features)]
    )

    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])
    model = TaxiModel(pipeline)

    return model

def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
        print(f"Done")


if __name__ == "__main__":

    X, y = load_train_data()
    model = train_model(X, y)
    model = model.fit(X, y)
    persist_model(model, MODEL_PATH)

