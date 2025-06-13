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
    Find the autocovariance at the input lag

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
    Find the autocorrelation at the input lag

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

