import numpy as np
from YuleWalker import yule_walker

class ARModel:
    def __init__(self, p: int):
        '''
        Initialize AR(p) model

        Parameters:
            p (int): Order of the autoregressive model
        '''
        self.p = p
        self.phi = None
        self.variance = None
        self.fitted_vals = None
        self.residuals = None

    def fit(self, time_series: np.ndarray):
        '''
        Fit the AR(p) model to the given time series using Yule-Walker equations
        '''
        # Store the fitted parameters
        self.phi, self.variance = yule_walker(time_series, self.p)

        n = len(time_series)
        self.fitted_vals = np.zeros(n)
        self.residuals = np.zeros(n)

        for t in range(self.p, n):
            past_vals = time_series[t-self.p:t][::-1]
            self.fitted_vals[t] = np.dot(self.phi, past_vals)
            self.residuals[t] = time_series[t] - self.fitted_vals[t]

    def predict(self, past_vals: np.ndarray, steps: int = 1) -> np.ndarray:
        '''
        Forecast future values based on last 'p' values

        Parameters:
            past_vals (np.ndarray): The p most recent observations
            steps (int): How far ahead we forecast in steps

        Returns:
            np.ndarray: Forecasted values of length 'steps'
        '''

        if self.phi is None:
            raise RuntimeError("Model must be fitted before prediction.")
        
        forecast = []
        values = list(past_vals[-self.p:])

        for _ in range(steps):
            next_val = np.dot(self.phi, values[-self.p:][::-1])
            forecast.append(next_val)
            values.append(next_val)

        return np.array(forecast)