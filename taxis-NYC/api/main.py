import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3
import pandas as pd
from datetime import datetime
import os
import sys



import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.normpath(os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)
from model.TaxiModel import TaxiModel


import config
DB_PATH = config.CONFIG['paths']['db_path']
MODEL_PATH = config.CONFIG['paths']['model_path']
MODEL_CUSTOM_PATH = config.CONFIG['paths']['model_custom_path']


app = FastAPI()


class Taxi(BaseModel):
    pickup_datetime : datetime
    passenger_count : int
    pickup_longitude : float
    pickup_latitude : float
    dropoff_longitude : float
    dropoff_latitude : float
    store_and_fwd_flag : str



@app.post("/predict")
def predict(taxi: Taxi):

    # load model
    print(f"Loading the model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # get prediction
    input_data = pd.DataFrame([taxi.model_dump()])

    result = model.predict(input_data)[0]

    return {"result": result}

@app.post("/predict_custom")
def predict_custom(taxi: Taxi):

    # load model
    print(f"Loading the model from {MODEL_CUSTOM_PATH}")
    with open(MODEL_CUSTOM_PATH, "rb") as file:
        model = pickle.load(file)

    # get prediction
    input_data = pd.DataFrame([taxi.model_dump()])
    result = model.predict(input_data)[0]

    # return prediction
    return {"result": result}

@app.get("/taxis/randomtest")
def get_random_test_patient():
    print(f"Reading test data from the database: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    data_test = pd.read_sql('SELECT * FROM test ORDER BY RANDOM() LIMIT 1', con)
    con.close()
    X = data_test.drop(columns=['trip_duration'])
    y = data_test['trip_duration']

    return {"x": X.iloc[0], "y": y[0]}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)