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
        return X
      
       
    
    def __postprocess(self, raw_output) :

        if raw_output < 60:
            return f"{int(raw_output)} secondes"
        elif raw_output < 3600:
            minutes = int(raw_output // 60)
            remaining_seconds = int(raw_output % 60)
            return f"{minutes} minutes et {remaining_seconds} secondes"
        else:
            hours = int(raw_output // 3600)
            remaining_seconds = int(raw_output % 3600)
            minutes = int(remaining_seconds // 60)
            remaining_seconds = int(remaining_seconds % 60)
            return f"{hours} heures, {minutes} minutes et {remaining_seconds} secondes"
            

    def predict(self, X):
        X_processed = self.__preprocess(X)
        raw_output = self.model.predict(X_processed)
        return self.__postprocess(raw_output)