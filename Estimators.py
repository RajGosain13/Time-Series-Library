import numpy as np
from Core import autocovariance_function

def method_of_moments_ar(time_series: np.ndarray, p: int):
    acvf = np.array([autocovariance_function(time_series, lag) for lag in range(p+1)])
    Gamma = np.array([[acvf[abs(i - j)] for j in range(p)] for i in range(p)])
    gamma = acvf[1:]
    phi = np.linalg.solve(Gamma, gamma)
    variance = acvf[0] - np.dot(phi, gamma)
    return phi, variance

def conditonal_least_squares_ar(time_series: np.ndarray, p: int):
    n = len(time_series)
    if n <= p:
        raise ValueError("Input time series too short for specified AR order")
    
    X = np.column_stack([time_series[i:n-p+i] for i in range(p)][::-1])
    Y = time_series[p:]
    phi = np.linalg.lstsq(X, Y, rcond=None)[0]
    residuals = Y - X @ phi
    variance = np.mean(np.power(residuals, 2))
    return phi, variance

def maximum_likelihood_ar(time_series: np.ndarray, p: int):
    n = len(time_series)
    X = np.column_stack([time_series[i:n-p+i] for i in range(p)][::-1])
    Y = time_series[p:]
    phi = np.linalg.lstsq(X, Y, rcond=None)[0]
    residuals = Y - X @ phi
    variance = np.sum(np.power(residuals, 2)) / n
    loglik = -0.5 * n * np.log(2 * np.pi * variance) - 0.5 * np.sum(np.power(residuals, 2)) / variance
    return phi, variance, loglik

