import numpy as np
from xgboost import XGBRegressor

class StackedXGBoostRegressor:
    def __init__(self, base_models, final_model):
        self.base_models = base_models
        self.final_model = final_model

    def fit(self, X_train, y_train):
        # Обучаем базовые модели и сохраняем их предсказания
        self.base_model_predictions = []
        for model in self.base_models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_train)
            self.base_model_predictions.append(predictions)

        # Стекинг базовых моделей
        X_stacked = np.column_stack(self.base_model_predictions)
        self.final_model.fit(X_stacked, y_train)

    def predict(self, X_test):  # Исправлено имя метода на "predict"
        # Генерируем прогнозы базовых моделей
        base_model_predictions = []
        for model in self.base_models:
            predictions = model.predict(X_test)
            base_model_predictions.append(predictions)

        # Стекинг базовых моделей
        X_stacked = np.column_stack(base_model_predictions)

        # Получаем прогноз окончательной модели
        final_predictions = self.final_model.predict(X_stacked)
        return final_predictions
