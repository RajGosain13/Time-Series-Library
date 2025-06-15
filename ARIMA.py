import numpy as np
from ARMA import ARMAModel

class ARIMAModel:
    def __init__(self, p: int, d: int, q: int, phi: np.ndarray = None, theta: np.ndarray = None, variance: float = None):
        '''
        Initialize an ARIMA(p,d,q) model.

        Parameters:
            p (int): Order of the AR model
            d (int): Degree of differencing
            q (int): Order of the MA model
            phi (np.ndarray, optional): AR coefficients
            theta (np.ndarray, optional): MA coefficients
            variance (float, optional): Variance of white noise
        '''
        self.p = p
        self.d = d
        self.q = q
        self.model = ARMAModel(p, q, phi=phi, theta=theta, variance=variance)
        self.original_series = None
        self.differenced_series = None

    def difference(self, time_series: np.ndarray) -> np.ndarray:
        '''
        Differences 
        '''
        differenced = time_series.copy()
        for _ in range(self.d):
            differenced = np.differenced(differenced)
        return differenced

    def inverse_difference(self, diff_series: np.ndarray, last_values: np.ndarray) -> np.ndarray:
        '''
        '''
        result = diff_series.copy()
        for i in range(self.d):
            result = np.r_[last_values[-(i+1)], result].cumsum()
        return result
    
    def set_original_series(self, series: np.ndarray):
        self.original_series = series.copy()
        self.differenced_series = self.difference(series) 

    def predict(self, steps: int = 1) -> np.ndarray: 
        '''
        Predict a time series for an ARMA(p,q) model

        Parameters:
            steps (int): Number of time steps into the future to predict

        Returns:
            np.ndarray: Prediction time series of length steps
        '''
        if self.original_series is None:
            raise RuntimeError("Original series must be set before prediction.")

        if self.model.phi is None or self.model.theta is None:
            raise RuntimeError("Model must be initialized with parameters before prediction.")

        past_obs = list(self.differenced_series[-self.p:]) if self.p > 0 else []
        past_errors = list(self.model.residuals[-self.q:]) if self.q > 0 else []

        preds = []
        for _ in range(steps):
            ar = np.dot(self.model.phi, past_obs[-self.p:][::-1]) if self.p > 0 else 0
            ma = np.dot(self.model.theta, past_errors[-self.q:][::-1]) if self.q > 0 else 0
            pred = ar + ma
            preds.append(pred)
            past_obs.append(pred)
            past_errors.append(0) 

        forecast_vals = self.inverse_difference(np.array(preds), self.original_series[-self.d:])
        return forecast_vals
    
    def simulate(self, n: int) -> np.ndarray:
        '''
        Simulates a time series for an ARIMA(p,d,q) model

        Parameters:
            n (int): Number of time steps to simulate

        Returns:
            np.ndarray: Simulated time series of length n
        '''
        arma_sim = self.model.simulate(n)
        simulated = arma_sim.copy()
        for _ in range(self.d):
            simulated = simulated.cumsum()
        return simulated
    
    
