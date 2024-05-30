# energy_analytics.py

"""
Energy Analytics Module for Energy Consumption Optimization in Smart Grids

This module contains functions for analyzing energy consumption patterns and trends
to inform optimization decisions.

Techniques Used:
- Descriptive Analytics
- Predictive Analytics
- Anomaly Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class EnergyAnalytics:
    def __init__(self):
        """
        Initialize the EnergyAnalytics class.
        """
        pass

    def load_data(self, filepath):
        """
        Load energy data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')

    def descriptive_analysis(self, data):
        """
        Perform descriptive analysis on the data.
        
        :param data: DataFrame, input data
        :return: dict, descriptive statistics
        """
        summary = data.describe()
        print("Descriptive Analysis Summary:")
        print(summary)
        return summary

    def plot_time_series(self, data, column):
        """
        Plot the time series data.
        
        :param data: DataFrame, input data
        :param column: str, column to plot
        """
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data[column], label=column)
        plt.title(f"Time Series Plot of {column}")
        plt.xlabel("Timestamp")
        plt.ylabel(column)
        plt.legend()
        plt.show()

    def seasonal_decomposition(self, data, column, model='additive', period=365):
        """
        Perform seasonal decomposition on the data.
        
        :param data: DataFrame, input data
        :param column: str, column to decompose
        :param model: str, type of decomposition ('additive' or 'multiplicative')
        :param period: int, periodicity of the data
        :return: DecomposeResult, decomposition results
        """
        decomposed = seasonal_decompose(data[column], model=model, period=period)
        decomposed.plot()
        plt.show()
        return decomposed

    def anomaly_detection(self, data, column, contamination=0.05):
        """
        Perform anomaly detection on the data.
        
        :param data: DataFrame, input data
        :param column: str, column to detect anomalies in
        :param contamination: float, proportion of anomalies in the data
        :return: DataFrame, data with anomaly labels
        """
        model = IsolationForest(contamination=contamination)
        data['anomaly'] = model.fit_predict(data[[column]])
        anomalies = data[data['anomaly'] == -1]
        
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data[column], label=column)
        plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomaly')
        plt.title(f"Anomaly Detection in {column}")
        plt.xlabel("Timestamp")
        plt.ylabel(column)
        plt.legend()
        plt.show()
        
        return data

    def predictive_analysis(self, data, column, periods=30):
        """
        Perform predictive analysis on the data using Holt-Winters Exponential Smoothing.
        
        :param data: DataFrame, input data
        :param column: str, column to forecast
        :param periods: int, number of periods to forecast
        :return: DataFrame, forecasted data
        """
        model = ExponentialSmoothing(data[column], trend='add', seasonal='add', seasonal_periods=365)
        model_fit = model.fit()
        forecast = model_fit.forecast(periods)
        
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data[column], label='Actual')
        plt.plot(forecast.index, forecast, label='Forecast', color='red')
        plt.title(f"Predictive Analysis of {column}")
        plt.xlabel("Timestamp")
        plt.ylabel(column)
        plt.legend()
        plt.show()
        
        forecast_df = pd.DataFrame({'timestamp': forecast.index, column: forecast.values})
        return forecast_df

if __name__ == "__main__":
    filepath = 'data/processed/preprocessed_energy_data.csv'
    
    analytics = EnergyAnalytics()
    data = analytics.load_data(filepath)
    
    # Descriptive Analysis
    analytics.descriptive_analysis(data)
    
    # Plot Time Series
    analytics.plot_time_series(data, 'energy_consumption')
    
    # Seasonal Decomposition
    analytics.seasonal_decomposition(data, 'energy_consumption')
    
    # Anomaly Detection
    anomaly_data = analytics.anomaly_detection(data, 'energy_consumption')
    
    # Predictive Analysis
    forecast_data = analytics.predictive_analysis(data, 'energy_consumption')
    print("Forecasted Data:")
    print(forecast_data.head())
