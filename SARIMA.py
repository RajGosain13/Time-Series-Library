import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAModel:
    def __init__(self, p, d, q, P=0, D=0, Q=0, s=0):
        self.p, self.d, self.q = p, d, q
        self.P, self.D, self.Q, self.s = P, D, Q, s
        self.model = None
        self.fitted_model = None
        self.original_series = None

    def fit(self, time_series: np.ndarray):
        self.original_series = np.asarray(time_series)
        self.model = SARIMAX(
            self.original_series,
            order=(self.p, self.d, self.q),
            seasonal_order=(self.P, self.D, self.Q, self.s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False)
        return self.fitted_model
    
    def predict(self, steps=1, alpha=0.05):
        if self.fitted_model is None:
            raise RuntimeError("Must fit model before predicting")
        forecast = self.fitted_model.get_forecast(steps=steps)
        return forecast.predicted_mean, forecast.conf_int(alpha=alpha)
    
    def simulate(self, n: int):
        if self.fitted_model is None:
            raise RuntimeError("Must fit model before simulating")
        sim = self.fitted_model.simulate(nsimulations=n)
        return sim
    
    def summary(self):
        if self.fitted_model is not None:
            return self.fitted_model.summary()
        return "Model not fitted yet"
    
    def residuals(self):
        if self.fitted_model is not None:
            return self.fitted_model.resid
        return None