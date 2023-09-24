from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import traceback
import numpy as np
import json

from stock_predictor import StockPredictor

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

ticker = None

class OutJson(BaseModel):
    ticker: str
    
    
@app.post("/set_ticker")
async def set_ticker(data: OutJson):
    try:
        global ticker
        json_compatible_item_data = jsonable_encoder(data)
        ticker = json_compatible_item_data['ticker']
        return get_predict_by_ticker()
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"Error": "Error in ticker or " + str(e)})
        


def get_predict_by_ticker():
    global ticker
    try:
        stock_pred = StockPredictor(ticker)
        answer = stock_pred.model_fit_predict()
        answer = {key: str(value) for key, value in answer.items()}
        # json_data = json.dumps(answer, indent=4)
        return JSONResponse(content=answer)
    
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"Error": "Error in ticker or " + str(e)})

    
@app.get("/", response_class=HTMLResponse)
async def get_ticker_form():
    with open("static/index.html", "r") as file:
        return file.read()
    

# if __name__ == "__main__":
#     import uvicorn
#     from worked_host import HOST, PORT
    
#     uvicorn.run(app, host=HOST, port=PORT)
# #     uvicorn main:app --reload