import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
    y = data_train['trip_duration ']
    return X, y

def preprocess_data(X):
    print(f"Preprocessing data")
    X = X.drop(columns=['id'])
    X = X.drop(columns=['dropoff_datetime'])
    X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
    X['pickup_date'] = X['pickup_datetime'].dt.date
    df_abnormal_dates = X.groupby('pickup_date').size()
    abnormal_dates = df_abnormal_dates[df_abnormal_dates < 6300]
    dict_weekday = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    weekday = X['pickup_datetime'].dt.weekday.map(dict_weekday).rename('weekday')
    hourofday = X['pickup_datetime'].dt.hour.rename('hour')
    month = X.pickup_datetime.dt.month.rename('month')


    return X

X_train, y_train = load_train_data()

