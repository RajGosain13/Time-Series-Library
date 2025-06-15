import numpy as np

def mean_function(time_series: np.ndarray) -> float:
    '''
    Finds the mean of a time series

    Parameters:
        time_series (np.ndarray): The input time series

    Returns:
        float: Mean of the time series
    '''

    return np.mean(time_series)

def variance_function(time_series: np.ndarray) -> float:
    '''
    Finds the variance of a time series

    Parameters:
        time_series (np.ndarray): The input time series

    Returns:
        float: Variance of the time series
    '''
    # Use ddof = 0 since this is population variance, not sample
    return np.var(time_series, ddof=0)

def autocovariance_function(time_series: np.ndarray, lag: int) -> float:
    '''
    Finds the autocovariance at the input lag

    Parameters:
        time_series (np.ndarray): The input time series
        lag (int): The lag at which we compute autocovariance
    
    Returns:
        float: Autocovariance at the given lag
    '''

    n = len(time_series)
    mu = np.mean(time_series)
    return np.sum((time_series[:n - lag] - mu) * (time_series[lag:] - mu)) / n

def autocorrelation_function(time_series: np.ndarray, lag: int) -> float:
    '''
    Finds the autocorrelation at the input lag

    Parameters:
        time_series (np.ndarray): The input time series
        lag (int): The lag at which we compute autocorrelation
    
    Returns:
        float: Autocorrelation at the given lag
    '''

    acvf0 = autocovariance_function(time_series, 0)
    if acvf0 == 0:
        return np.nan
    return autocovariance_function(time_series, lag) / acvf0

def __acf__(time_series: np.ndarray, lags: int) -> np.ndarray:
    return np.array([autocorrelation_function(time_series, k) for k in range(lags + 1)])

def pacf(time_series: np.ndarray, lags: int, return_all_phi: bool = False):
    '''
    Finds the partial autocorrelation at the given input lags

    Parameters:
        time_series (np.ndarray): The input time series
        lag (int): The lags at which we compute autocorrelation
        return_all_phi (bool): If True, returns all phi[k, j]
    '''
    acf_vals = __acf__(time_series, lags)
    pacf_vals = np.zeros(lags+1)
    phi = np.zeros((lags+1, lags+1))
    variance = np.zeros(lags+1)

    pacf_vals[0] = 1.0
    phi[1,1] = acf_vals[1]
    variance_function[1] = 1 - np.power(acf_vals, 2)
    pacf_vals[1] = phi[1,1,]

    for k in range(2, lags+1):
        sum_phi_gamma = sum(phi[k-1,j] * acf_vals[k-j] for j in range(1, k))
        phi[k, k] = (acf_vals[k] - sum_phi_gamma) / variance[k - 1]

        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

        variance[k] = variance[k - 1] * (1 - phi[k, k]**2)
        pacf_vals[k] = phi[k, k]\
        
    if return_all_phi:
        phi_table = {f'phi_{k}_{j}': phi[k, j] for k in range(1, lags + 1) for j in range(1, k + 1)}
        return pacf_vals, phi_table
    
    return pacf_vals
