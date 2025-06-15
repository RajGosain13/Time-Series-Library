import numpy as np

class ARMAModel:
    def __init__(self, p: int, q: int, phi: np.ndarray = None, theta: np.ndarray = None, variance: float = None):
        '''
        Initialize ARMA(p,q) model

        Parameters:
            p (int): Order of the AR model
            q (int): Order of the MA model
            phi (np.ndarray, optional): AR coefficients
            theta (np.ndarray, optional): MA coefficients
            variance (float, optional): Variance of white noise
        '''
        self.p = p
        self.q = q
        self.phi = phi
        self.theta = theta
        self.variance = variance
        self.residuals = None

        if self.phi is not None and len(self.phi) != p:
            raise ValueError("Length of phi must equal AR order p")
        
        if self.theta is not None and len(self.theta) != q:
            raise ValueError("Length of theta must equal MA order q")
        
    def simulate(self, n: int) -> np.ndarray:
        '''
        Simulates a time series for an ARMA(p,q) model

        Parameters:
            n (int): Number of time steps to simulate

        Returns:
            np.ndarray: Simulated time series of length n
        '''
        if self.phi is None or self.theta is None or self.variance is None:
            raise RuntimeError("phi, theta, and variance must be specified to simulate")
        
        max_lag = max(self.p, self.q)
        x = np.zeros(n + max_lag)
        e = np.random.normal(0, np.sqrt(self.variance), size=n+max_lag)

        for t in range(max_lag, n + max_lag):
            ar =np.dot(self.phi, x[t-self.p:t][::-1]) if self.p > 0 else 0
            ma = np.dot(self.theta, x[t-self.q:t][::-1]) if self.q > 0 else 0
            x[t] = ar + ma + e[t]

        return x[max_lag:]
    
    def predict(self, past_ar: np.ndarray, past_errors: np.ndarray, steps: int = 1) -> np.ndarray:
        '''
        Predict a time series for an ARMA(p,q) model

        Parameters:
            past_ar (np.ndarray): Previous observations of the AR portion
            past_errors (np.ndarray): Previous observations of the MA portion
            steps (int): Number of time steps into the future to predict

        Returns:
            np.ndarray: Prediction time series of length steps
        '''
        if self.phi is None or self.theta is None:
            raise RuntimeError("Model must be fitted or initalized before prediction")
        
        preds = []

        observed = list(past_ar[-self.p:])
        errors = list(past_errors[-self.q:])

        for _ in range(steps):
            ar =np.dot(self.phi, observed[-self.p:][::-1]) if self.p > 0 else 0
            ma = np.dot(self.theta, errors[-self.q:][::-1]) if self.q > 0 else 0
            pred = ar + ma
            preds.append(pred)
            # Assume error = 0 for future steps
            observed.append(pred)
            errors.append(0)

        return np.array(preds)