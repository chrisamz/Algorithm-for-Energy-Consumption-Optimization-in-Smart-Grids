# evaluation.py

"""
Evaluation Module for Energy Consumption Optimization in Smart Grids

This module contains functions for evaluating the performance of forecasting models
and optimization algorithms using appropriate metrics.

Techniques Used:
- Forecasting Model Evaluation
- Optimization Algorithm Evaluation

Metrics Used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Energy Efficiency
- Cost Reduction
- Load Balancing
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def load_data(self, filepath):
        """
        Load test data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')

    def evaluate_forecasting_model(self, model_name, data, column, forecast_periods=30):
        """
        Evaluate the specified forecasting model using various metrics.
        
        :param model_name: str, name of the model to evaluate
        :param data: DataFrame, time series data
        :param column: str, column to evaluate
        :param forecast_periods: int, number of periods to forecast
        :return: dict, evaluation metrics
        """
        model = joblib.load(f'models/{model_name}_model.pkl')
        if model_name == 'prophet':
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = model.predict(future)
            predictions = forecast['yhat'][-forecast_periods:].values
        else:
            predictions = model.forecast(steps=forecast_periods)
        
        true_values = data[column][-forecast_periods:].values
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        return {'mae': mae, 'rmse': rmse}

    def evaluate_optimization_algorithm(self, actual_values, optimized_values):
        """
        Evaluate the specified optimization algorithm using various metrics.
        
        :param actual_values: list, actual energy values
        :param optimized_values: list, optimized energy values
        :return: dict, evaluation metrics
        """
        energy_efficiency = np.mean(optimized_values) / np.mean(actual_values)
        cost_reduction = np.sum(actual_values) - np.sum(optimized_values)
        load_balancing = np.std(optimized_values) / np.std(actual_values)
        
        return {
            'energy_efficiency': energy_efficiency,
            'cost_reduction': cost_reduction,
            'load_balancing': load_balancing
        }

if __name__ == "__main__":
    test_data_filepath = 'data/processed/preprocessed_energy_data_test.csv'
    target_column = 'energy_consumption'

    evaluator = ModelEvaluation()
    data = evaluator.load_data(test_data_filepath)
    
    # Evaluate forecasting models
    arima_metrics = evaluator.evaluate_forecasting_model('arima', data, target_column)
    sarima_metrics = evaluator.evaluate_forecasting_model('sarima', data, target_column)
    prophet_metrics = evaluator.evaluate_forecasting_model('prophet', data, target_column)
    lstm_metrics = evaluator.evaluate_forecasting_model('lstm', data, target_column)
    
    print("ARIMA Model Evaluation:", arima_metrics)
    print("SARIMA Model Evaluation:", sarima_metrics)
    print("Prophet Model Evaluation:", prophet_metrics)
    print("LSTM Model Evaluation:", lstm_metrics)

    # Example actual and optimized values for optimization algorithm evaluation
    actual_values = data[target_column].values[-30:]
    optimized_values = actual_values * 0.9  # Example optimized values (10% reduction)

    optimization_metrics = evaluator.evaluate_optimization_algorithm(actual_values, optimized_values)
    print("Optimization Algorithm Evaluation:", optimization_metrics)
