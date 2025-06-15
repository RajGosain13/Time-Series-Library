import numpy as np
from YuleWalker import yule_walker

class ARModel:
    def __init__(self, p: int, phi: np.ndarray = None, variance: float = None):
        '''
        Initialize AR(p) model

        Parameters:
            p (int): Order of the AR model
            phi (np.ndarray, optional): AR coefficients
            variance (float, optional): Variance of white noise
        '''
        self.p = p
        self.phi = phi
        self.variance = variance
        self.fitted_vals = None
        self.residuals = None

    def fit(self, time_series: np.ndarray):
        '''
        Fit the AR(p) model to the given time series using Yule-Walker equations

        Parameters:
            time_series (np.ndarray): Input time series to be fit
        '''

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
    
    def simulate(self, n: int) -> np.ndarray:
        '''
        Simulate a time series for an AR(p) model

        Parameters:
            n (int): Number of time steps to simulate

        Returns:
            np.ndarray: Simulated time series of length n
        '''
        if self.phi is None or self.variance is None:
            raise RuntimeError("phi and variance must be specified to simulate")
        
        time_series = np.zeros(n)
        noise = np.random.normal(0, np.sqrt(self.variance), size=n)

        for t in range(self.p, n):
            time_series[t] = np.dot(self.phi, time_series[t - self.p: t][::-1]) + noise[t]

        return time_series