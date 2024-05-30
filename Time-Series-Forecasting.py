# time_series_forecasting.py

"""
Time Series Forecasting Module for Energy Consumption Optimization in Smart Grids

This module contains functions for building, training, and evaluating time series forecasting models
to predict future energy consumption based on historical data.

Techniques Used:
- ARIMA
- SARIMA
- LSTM
- Prophet

Metrics Used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fbprophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

class TimeSeriesForecasting:
    def __init__(self):
        """
        Initialize the TimeSeriesForecasting class.
        """
        self.models = {}

    def load_data(self, filepath):
        """
        Load time series data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        return data

    def train_arima(self, data, order=(5, 1, 0)):
        """
        Train an ARIMA model.
        
        :param data: Series, time series data
        :param order: tuple, order of the ARIMA model
        :return: ARIMA model
        """
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        self.models['arima'] = model_fit
        return model_fit

    def train_sarima(self, data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Train a SARIMA model.
        
        :param data: Series, time series data
        :param order: tuple, order of the SARIMA model
        :param seasonal_order: tuple, seasonal order of the SARIMA model
        :return: SARIMA model
        """
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        self.models['sarima'] = model_fit
        return model_fit

    def train_prophet(self, data):
        """
        Train a Prophet model.
        
        :param data: DataFrame, time series data with columns 'ds' (date) and 'y' (value)
        :return: Prophet model
        """
        model = Prophet()
        model.fit(data)
        self.models['prophet'] = model
        return model

    def train_lstm(self, data, n_lag=1, n_ahead=1, n_epochs=50, n_batch=1, n_neurons=50):
        """
        Train an LSTM model.
        
        :param data: Series, time series data
        :param n_lag: int, number of lag observations as input
        :param n_ahead: int, number of observations as output
        :param n_epochs: int, number of epochs for training
        :param n_batch: int, number of batches for training
        :param n_neurons: int, number of neurons in the LSTM layer
        :return: LSTM model
        """
        X, y = self.create_lagged_features(data, n_lag, n_ahead)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape=(1, n_lag)))
        model.add(Dense(n_ahead))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=2, shuffle=False)
        
        self.models['lstm'] = model
        return model

    def create_lagged_features(self, data, n_lag=1, n_ahead=1):
        """
        Create lagged features for time series data.
        
        :param data: Series, time series data
        :param n_lag: int, number of lag observations as input
        :param n_ahead: int, number of observations as output
        :return: tuple, input and output arrays for the model
        """
        X, y = [], []
        for i in range(len(data) - n_lag - n_ahead + 1):
            X.append(data[i:(i + n_lag)].values)
            y.append(data[(i + n_lag):(i + n_lag + n_ahead)].values)
        return np.array(X), np.array(y)

    def evaluate_model(self, model_name, data, n_ahead=1):
        """
        Evaluate the specified model using various metrics.
        
        :param model_name: str, name of the model to evaluate
        :param data: Series, time series data
        :param n_ahead: int, number of observations to predict ahead
        :return: dict, evaluation metrics
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        
        if model_name in ['arima', 'sarima']:
            predictions = model.forecast(steps=len(data))
        elif model_name == 'prophet':
            future = model.make_future_dataframe(periods=len(data))
            forecast = model.predict(future)
            predictions = forecast['yhat'][-len(data):].values
        elif model_name == 'lstm':
            X, y_true = self.create_lagged_features(data, n_lag=1, n_ahead=n_ahead)
            X = X.reshape(X.shape[0], 1, X.shape[1])
            predictions = model.predict(X)
            predictions = predictions.flatten()
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        mae = mean_absolute_error(data, predictions)
        rmse = np.sqrt(mean_squared_error(data, predictions))
        return {'mae': mae, 'rmse': rmse}

    def save_model(self, model_name, filepath):
        """
        Save the trained model to a file.
        
        :param model_name: str, name of the model to save
        :param filepath: str, path to save the model
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        joblib.dump(model, filepath)

    def load_model(self, model_name, filepath):
        """
        Load a trained model from a file.
        
        :param model_name: str, name to assign to the loaded model
        :param filepath: str, path to the saved model
        """
        self.models[model_name] = joblib.load(filepath)

if __name__ == "__main__":
    filepath = 'data/processed/preprocessed_energy_data.csv'
    
    ts_forecasting = TimeSeriesForecasting()
    data = ts_forecasting.load_data(filepath)
    energy_data = data['energy_consumption']
    
    # Train models
    arima_model = ts_forecasting.train_arima(energy_data)
    sarima_model = ts_forecasting.train_sarima(energy_data)
    prophet_data = data.reset_index().rename(columns={'timestamp': 'ds', 'energy_consumption': 'y'})
    prophet_model = ts_forecasting.train_prophet(prophet_data)
    lstm_model = ts_forecasting.train_lstm(energy_data)
    
    # Save models
    ts_forecasting.save_model('arima', 'models/arima_model.pkl')
    ts_forecasting.save_model('sarima', 'models/sarima_model.pkl')
    ts_forecasting.save_model('prophet', 'models/prophet_model.pkl')
    ts_forecasting.save_model('lstm', 'models/lstm_model.pkl')
    
    # Evaluate models
    metrics_arima = ts_forecasting.evaluate_model('arima', energy_data)
    metrics_sarima = ts_forecasting.evaluate_model('sarima', energy_data)
    metrics_prophet = ts_forecasting.evaluate_model('prophet', energy_data)
    metrics_lstm = ts_forecasting.evaluate_model('lstm', energy_data)
    
    print("ARIMA Model Evaluation:", metrics_arima)
    print("SARIMA Model Evaluation:", metrics_sarima)
    print("Prophet Model Evaluation:", metrics_prophet)
    print("LSTM Model Evaluation:", metrics_lstm)
