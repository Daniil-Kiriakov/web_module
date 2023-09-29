from stock_predictor import StockPredictor

ticker = 'LKOH'
stock_pred = StockPredictor(ticker)
answer = stock_pred.model_fit_predict()

for key, val in answer.items():
    print('{0}  ||  {2}  ||  {1}   ||   {3}'.format(key, len(val), type(val), val[:4]))
    
    