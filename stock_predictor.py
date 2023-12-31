import requests
import apimoex
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import os
import joblib
from sklearn.metrics import (mean_absolute_percentage_error, 
                             mean_squared_error, 
                             mean_absolute_error)


class StockPredictor:
    def __init__(self, ticker='GAZP'):
        self.ticker = ticker
        self.model = None
    
    def download_stocks(self, start='2000-10-01', end=datetime.now() + timedelta(days=30), interval=24):
        with requests.Session() as session:
            data = apimoex.get_market_candles(session, self.ticker, start=start, end=end, interval=interval)
            df = pd.DataFrame(data)
            df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Value']
            df.set_index('Date', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Value']]
            return df

    def create_dataset(self, df, window=30, percent=0.85, column='Close'):
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
        test_time = [int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp()) for date in test_time]

        return list(x_train), list(y_train), list(x_test), list(y_test), test_time
    
    
    def get_metrics(self, y_test, predictions):
        MAE = mean_absolute_error(y_test, predictions)
        MAPE = mean_absolute_percentage_error(y_test, predictions) * 100
        RMSE = mean_squared_error(y_test, predictions, squared=False)
        return MAE, MAPE, RMSE
    
        
    def model_fit_predict(self, prediction_days=30, window=30):
        stock_dataframe = self.download_stocks()
        x_train, y_train, x_test, y_test, test_time = self.create_dataset(stock_dataframe.iloc[:-14, :])

        # model_filenames = os.listdir('models/')
        # test_predictions = []
        # for list_model in model_filenames:
        #     temp_model = joblib.load('models/{}'.format(list_model))
        #     test_predictions.append(temp_model.predict(x_test))
        model = XGBRegressor()
        model.fit(x_train, y_train)
        test_predictions = model.predict(x_test)
        
        # test_predictions = np.mean( np.reshape(test_predictions, (len(model_filenames), len(x_test))), axis=1)
        # print(np.reshape(test_predictions, (len(model_filenames), len(x_test)))[:3])
        # print(test_predictions[:3])
        
        MAE, MAPE, RMSE = self.get_metrics(y_test=y_test, predictions=test_predictions)

        '''prediction'''
        temp = stock_dataframe.iloc[-100:-15, :].copy()
        future_pred = []
        last_days = temp['Close'][-window:].tolist()

        for _ in range(prediction_days):
            last_days = last_days[-window:]

            temp_prediction = model.predict([last_days])

            # temp_pred = []
            # for list_model in model_filenames:
            #     temp_model = joblib.load('models/{}'.format(list_model))
            #     temp_pred.append(temp_model.predict(x_test))
                
            # temp_prediction = np.mean(temp_pred)
            temp_prediction[0] = np.random.normal(loc=temp_prediction[0], scale=temp_prediction[0] * 0.005, size=1)
            
            future_pred.append(temp_prediction[0])
            last_days.append(temp_prediction[0])
            
        real_dataframe = stock_dataframe.iloc[-15:, :].copy()
        time_real_prices = [int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp()) for date in real_dataframe.index.values]

        start_date = datetime.fromtimestamp(time_real_prices[0])
        time_future_predictions = [start_date + timedelta(days=i) for i in range(prediction_days)]
        time_future_predictions = [int(datetime.combine(date, datetime.min.time()).timestamp()) for date in time_future_predictions]

        '''real prices'''

  
        test_df = pd.DataFrame({'time': test_time[-150:], 'price': test_predictions[-150:]})
        fut_df = pd.DataFrame({'time': time_future_predictions, 'price': future_pred})
        real_df = pd.DataFrame({'time': time_real_prices, 'price': real_dataframe['Close'].values})

        # time	price_test	price_fut	price
        all_df = test_df.merge(fut_df, on='time', how='outer', suffixes=('_test', '_fut')) \
                        .merge(real_df, on='time', how='outer', suffixes=('', '_real'))
        all_df.fillna(-1, inplace=True)
                
        res = {
            # 'Model': model,
            'Ticker': self.ticker,
            'Metrics': [MAE, MAPE, RMSE],

            # 'Time_test_predictions': [datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d') for timestamp in all_df['time'].values],
            'Time_test_predictions': np.arange(len(all_df['time'].values)).tolist(),

            'Test_predictions': all_df['price_test'].tolist(),
            'Future_predictions': all_df['price_fut'].tolist(),
            'Real_prices': all_df['price'].tolist(),
        }
        
        return res