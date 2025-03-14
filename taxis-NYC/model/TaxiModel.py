from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder


class TaxiModel:
    def __init__(self, model):
        self.model = model

    def __preprocess(self, X):

        X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
        X['pickup_date'] = X['pickup_datetime'].dt.date
        df_abnormal_dates = X.groupby('pickup_date').size()
        abnormal_dates = df_abnormal_dates[df_abnormal_dates < 6300]
        X['weekday'] = X['pickup_datetime'].dt.weekday
        X['month'] = X['pickup_datetime'].dt.month
        X['hour'] = X['pickup_datetime'].dt.hour
        X['abnormal_period'] = X['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
        
        num_features = ['abnormal_period', 'hour']
        cat_features = ['weekday', 'month']
        train_features = num_features + cat_features
        return X[train_features]
    
    
    def fit(self, X, y):
        X_processed = self.__preprocess(X)
        self.model.fit(X_processed, np.log1p(y).rename('log_'+y.name))
        return self
        

    
    def __postprocess(self, raw_output) :
        output = np.expm1(raw_output)
        return output
    

    def predict(self, X):
        X_processed = self.__preprocess(X)
        raw_output = self.model.predict(X_processed)
        return self.__postprocess(raw_output)