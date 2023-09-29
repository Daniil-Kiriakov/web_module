import requests
import apimoex
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error,
                             mean_absolute_error)
import joblib


def download_stocks(ticker, start='2000-10-01', end=datetime.now() + timedelta(days=30), interval=24):
    with requests.Session() as session:
        data = apimoex.get_market_candles(
            session, ticker, start=start, end=end, interval=interval)
        df = pd.DataFrame(data)
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Value']
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Value']]
        return df


def create_dataset(df, window=30, percent=0.85, column='Close'):
    SIZE = int(len(df) * percent)
    train_df, test_df = df[:SIZE], df[SIZE - 1:]

    def create_data_samples(data_frame):
        x, y = [], []
        for i in range(len(data_frame) - window - 1):
            temp = data_frame[i: (i + window)][column]
            x.append(temp)
            y.append(data_frame.iloc[i + window][column])
        return x, y

    x_train, y_train = create_data_samples(train_df)
    x_test, y_test = create_data_samples(test_df)

    test_time = test_df.index.values[-len(x_test):]
    test_time = [int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp())
                 for date in test_time]

    return list(x_train), list(y_train), list(x_test), list(y_test), test_time


# ticker_model = {
#     'GAZP': XGBRegressor(n_estimators=100, max_depth=3),
#     'SBER': XGBRegressor(n_estimators=500, max_depth=2),
#     'LKOH': XGBRegressor(n_estimators=300, max_depth=3),
# }


# for ticker, model in ticker_model.items():
#     df = download_stocks(ticker=ticker)
#     x_train, y_train, x_test, y_test, test_time = create_dataset(df)
#     model.fit(x_train, y_train)

#     print(f'SCORE: {mean_squared_error(y_test, model.predict(x_test))}')

#     model_filename = 'models/model_{}.pkl'.format(ticker)
#     joblib.dump(model, model_filename)
#     print('Model save')

#     loaded_model = joblib.load(model_filename)
#     y_pred_loaded = loaded_model.predict(x_test)
#     print(f'SCORE: {mean_squared_error(y_test, y_pred_loaded)}')

a = np.array([[1, 2, 2],
              [3, 4, 3],
              [4, 1, 2,],
              [1, 1, 1,]])

# print(type(np.mean(a, axis=1)))
# print(a)
# print(np.reshape([1,2,3,1,2,3], (2, 3)))
import os
print(os.listdir('models/'))

