import sqlite3
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

def preprocess_data(X):
    print(f"Preprocessing data")
    X = X.drop(columns=['id'])
    X = X.drop(columns=['dropoff_datetime'])
    X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
    X['pickup_date'] = X['pickup_datetime'].dt.date
    df_abnormal_dates = X.groupby('pickup_date').size()
    abnormal_dates = df_abnormal_dates[df_abnormal_dates < 6300]
    X['weekday'] = X['pickup_datetime'].dt.weekday
    X['month'] = X['pickup_datetime'].dt.month
    X['hour'] = X['pickup_datetime'].dt.hour
    X['abnormal_period'] = X['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)

    return X

def train_model(X, y):
    print(f"Building a model")
    num_features = ['abnormal_period', 'hour']
    cat_features = ['weekday', 'month']
    train_features = num_features + cat_features
    
    column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
    ('scaling', StandardScaler(), num_features)]
    )

    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(X[train_features], y)

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

    X_train, y_train = load_train_data()
    X = preprocess_data(X_train)
    model = train_model(X, y_train)
    persist_model(model, MODEL_PATH)

