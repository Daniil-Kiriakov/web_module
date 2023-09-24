import time
import requests
import apimoex
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


import joblib
import pandas as pd
import numpy as np

# pre-trained models:
# model1, model2, .. modeln
# -> pd.DataFrame({pred1, pred2, .., target=avg(pred)})


class MetaModel:
    def __init__(self, num_models):
        self.num_models = num_models
        self.models = []

    def save_models(self):
        for i, model in enumerate(self.models, start=1):
            joblib.dump(model, f'model{i}.pkl')

    def load_models(self):
        self.models = []
        for i in range(1, self.num_models + 1):
            model = joblib.load(f'model{i}.pkl')
            self.models.append(model)

    def predict_and_record(self, X_test):
        predictions = []
        for i, model in enumerate(self.models, start=1):
            model_predictions = model.predict(X_test)
            predictions.append(model_predictions)
        
        # Calculate the average predictions
        average_predictions = np.mean(predictions, axis=0)
        
        # Create a DataFrame to store the individual predictions
        predictions_df = pd.DataFrame({'Model_{}'.format(i): model_pred for i, model_pred in enumerate(predictions, start=1)})
        predictions_df['Average'] = average_predictions
        
        return predictions_df

# Пример использования:
# Создаем экземпляр класса ModelEnsemble с 5 моделями
ensemble = MetaModel(num_models=5)

# Сохраняем модели
ensemble.save_models()

# Загружаем модели
ensemble.load_models()

# Предсказываем и записываем результаты
X_test = ...  # Замените на ваши тестовые данные
predictions_df = ensemble.predict_and_record(X_test)

# Теперь у вас есть DataFrame с предсказаниями каждой модели и средними значениями
print(predictions_df)
