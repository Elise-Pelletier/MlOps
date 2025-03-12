import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
import common
from model.train import preprocess_data
import os

DB_PATH = os.path.join('data', 'taxis.db') 
MODEL_PATH = common.CONFIG['paths']['model_path']

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model

def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    con = sqlite3.connect(path)
    data_test = pd.read_sql('SELECT * FROM test', con)
    con.close()
    X = data_test.drop(columns=['trip_duration'])
    y = data_test['trip_duration']
    return X, y

def evaluate_model(model, X, y):
    print(f"Evaluating the model")
    X_preprocessed = preprocess_data(X)
    y_pred = model.predict(X_preprocessed)
    score = mean_squared_error(y, y_pred)
    return score

if __name__ == "__main__":

    X_test, y_test = load_test_data(DB_PATH)
    model = load_model(MODEL_PATH)
    score_test = evaluate_model(model, X_test, y_test)
    print(f"Score on test data {score_test:.2f}")
