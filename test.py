from stock_predictor import StockPredictor

ticker = 'SBER'
stock_pred = StockPredictor(ticker)
answer = stock_pred.model_fit_predict()
for key, val in answer.items():
    print(len(val), key)
