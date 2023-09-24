import time
import requests
import apimoex
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

class GetModels:
    def __init__(self, ticker='GAZP'):
        self.ticker = ticker
        self.model = None
    
    def download_stocks(self, start='2000-10-01', end=None, interval=24):
        if end is None:
            end = datetime.now() + timedelta(days=30)
            
        with requests.Session() as session:
            data = apimoex.get_market_candles(session, self.ticker, start=start, end=end, interval=interval)
            df = pd.DataFrame(data)
            df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Value']
            df.set_index('Date', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Value']]
            return df

    def create_dataset(self, df, window=30, percent=0.85, column='Close'):
        SIZE = int(len(df) * percent)
        train_df = df[:SIZE]
        test_df = df[SIZE - 1:]

        x_train, y_train, x_test, y_test = [], [], [], []

        for i in range(len(train_df) - window - 1):
            temp = train_df[i: (i + window)][column]

            x_train.append(temp)
            y_train.append(train_df.iloc[i + window][column])

        for i in range(len(test_df) - window - 1):
            temp = test_df[i: (i + window)][column]

            x_test.append(temp)
            y_test.append(test_df.iloc[i + window][column])

        test_time = test_df.index.values[-len(x_test):]
        test_time = [int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp()) for date in test_time]

        return list(x_train), list(y_train), list(x_test), list(y_test), test_time
    
        
    def model_fit_predict(self, prediction_days=30, window=30):
        stock_dataframe = self.download_stocks()
        x_train, y_train, x_test, y_test, test_time = self.create_dataset(stock_dataframe.iloc[:-14, :])

        model = XGBRegressor(n_estimators=1000, max_depth=3)
        model.fit(x_train, y_train)
        self.model = model
        
        return model

def download_stocks(ticker='GAZP', start='2000-10-01', end=None, interval=24):
    if end is None:
        end = datetime.now() + timedelta(days=30)
        
    with requests.Session() as session:
        data = apimoex.get_market_candles(session, ticker, start=start, end=end, interval=interval)
        df = pd.DataFrame(data)
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Value']
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Value']]
        return df

def create_dataset(df, window=30, percent=0.85, column='Close'):
    SIZE = int(len(df) * percent)
    train_df = df[:SIZE]
    test_df = df[SIZE - 1:]

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(len(train_df) - window - 1):
        temp = train_df[i: (i + window)][column]

        x_train.append(temp)
        y_train.append(train_df.iloc[i + window][column])

    for i in range(len(test_df) - window - 1):
        temp = test_df[i: (i + window)][column]

        x_test.append(temp)
        y_test.append(test_df.iloc[i + window][column])

    test_time = test_df.index.values[-len(x_test):]
    test_time = [int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp()) for date in test_time]

    return list(x_train), list(y_train), list(x_test), list(y_test), test_time

df = download_stocks(ticker='TATN')
x_train, y_train, x_test, y_test, test_time = create_dataset(df.iloc[:-14, :])

import joblib

tick = ['SBER', 'GAZP']

for val in tick:
    model = GetModels(ticker=val).model_fit_predict()
    joblib.dump(model, open(f'model_{val}', "wb"))

models = []
for val in tick:
    model = joblib.load(f'model_{val}.pkl')
    models.append(model)

df_predictions = pd.DataFrame()
for i, model in enumerate(models, start=1):
    model_predictions = model.predict(x_test)
    df_predictions[f'model_{i}'] = model_predictions

# Calculate the average predictions
df_predictions['Average'] = np.mean(df_predictions.values, axis=1)
        
df_predictions.tail()