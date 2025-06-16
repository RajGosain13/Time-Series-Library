import numpy as np
from scipy.optimize import minimize

class MAModel:
    def __init__(self, q: int, theta: np.ndarray = None, variance: float = None):
        '''
        Initalize an MA(q) model

        Parameters:
            q (int): Order of the MA model
            theta (np.ndarray, optional): MA coefficients
            variance (float, optional): Variance of white noise
        '''
        self.q = q
        self.theta = theta
        self.variance = variance
        self.residuals = None

        if self.theta is not None and len(self.theta) != self.q:
            raise ValueError("Length of theta must match MA order q")
        
    def simulate(self, n: int) -> np.ndarray:
        '''
        Simulates a time series for an MA(q) model

        Parameters:
            n (int): Number of time steps to simulate

        Returns:
            np.ndarray: Simulated time series of length n
        '''

        if self.theta is None or self.variance is None:
            raise RuntimeError("theta and variance must be specified to simluate")
        
        e = np.random.normal(0, np.sqrt(self.variance), size=n + self.q)
        y = np.zeros(n)

        for t in range(n):
            past = e[t:t+self.q][::-1]
            y[t] = e[t+self.q] + np.dot(self.theta, past)

        return y
    
    def predict(self, past_errors: np.ndarray) -> float:
        '''
        Forecast future values based on last 'p' values

        Parameters:
            past_errors (np.ndarray): Array of the q most recent white noise terms

        Returns:
            float: Next step prediction
        '''

        if self.theta is None:
            raise RuntimeError("Model must be initalized with theta to predict")
        if len(past_errors) != self.q:
            raise ValueError("Length of past errors must match MA order q")
    
        return np.dot(self.theta, past_errors[::-1])
    
    def fit(self, time_series: np.ndarray):
        n = len(time_series)
        q = self.q

        def objective(theta):
            residuals = np.zeros(n)
            for t in range(q, n):
                ma = np.dot(theta, residuals[t-q:t][::-1]) if q > 0 else 0
                residuals[t] = time_series[t] - ma
            return np.sum(residuals[q:] ** 2)
        
        initial = np.zeros(q)
        result = minimize(objective, initial, method='BFGS')

        if not result.success:
            raise RuntimeError(f'MA fit failed: {result.message}')
        
        self.theta = result.x
        self.residuals = np.zeros(n)
        for t in range(q, n):
            ma = np.dot(self.theta, self.residuals[t-q:t][::-1]) if q > 0 else 0
            self.residuals[t] = time_series[t] - ma

        self.variance = np.var(self.residuals[q:])