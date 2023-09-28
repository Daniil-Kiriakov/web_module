from xgboost import XGBRegressor

from stock_predictor import StockPredictor
from stack_model import StackedXGBoostRegressor

import pandas as pd
import requests
from datetime import datetime, timedelta
import apimoex
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# stock_pred = StockPredictor(ticker)
# answer = stock_pred.model_fit_predict()

# for key, val in answer.items():
#     print(key, len(val), type(val))

def process_stock_data(ticker, start='2000-10-01', end=None, interval=24, window=30, percent=0.85, column='Close'):
    if end is None:
        end = datetime.now() + timedelta(days=30)

    with requests.Session() as session:
        data = apimoex.get_market_candles(session, ticker, start=start, end=end, interval=interval)
        df = pd.DataFrame(data)
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Value']
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Value']]

    SIZE = int(len(df) * percent)
    train_df = df[:SIZE]
    test_df = df[SIZE - 1:]

    x_train = [] 
    y_train = []

    x_test = [] 
    y_test = []

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


ticker = 'SBER'
x_train, y_train, x_test, y_test, test_time = process_stock_data(ticker)

    
base_models = [XGBRegressor(n_estimators=100, max_depth=3) for _ in range(3)]
final_model = XGBRegressor(n_estimators=500, max_depth=2)

stacked_model = StackedXGBoostRegressor(base_models, final_model)

StackedXGBoostRegressor.fit(X_train=x_train, y_train=y_train)
pred = StackedXGBoostRegressor.predict_(x_test)

print(mean_absolute_percentage_error(y_test, pred), mean_squared_error(y_test, pred))