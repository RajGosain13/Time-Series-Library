import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

def adf_test(time_series: np.ndarray, max_lag: int = None):
    result = adfuller(time_series, maxlag=max_lag, autolag='AIC')
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'ic_best': result[5]
    }

def ljung_box(time_series: np.ndarray, lags: int=20):
    lbvalue, pval = acorr_ljungbox(time_series, lags=lags, return_df=False)
    return {
        'Ljung-Box Statistic': lbvalue,
        'p-value': pval
    }

def shapiro_wilk(time_series: np.ndarray):
    stat, p = shapiro(time_series)
    return {
        'W statistic': stat, 
        'p-value': p
    }