import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from ARMA import ARMAModel
from Core import autocovariance_function
from scipy.stats import chi2

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

def white_noise_test(residuals: np.ndarray, lags: int=20):
    lb = ljung_box(residuals, lags)
    return {
        'statistic': lb['Ljung-Box Statistic'],
        'p-value': lb['p-value'],
        'conclusion': 'Reject H0 (not white noise)' if np.any(lb['p-value'] < 0.05) else 'Fail to reject H0 (white noise)'
    }

def barlett(acf_vals: np.ndarray, n: int, lags: int=20):
    confidence_intervals = []
    for h in range(1, lags + 1):
        se = np.sqrt((1+2*np.sum(acf_vals[1:h]**2)) / n)
        ci = (acf_vals[h] - 2 * se, acf_vals[h] + 2 * se)
        confidence_intervals.append(ci)

    return confidence_intervals

def eacf(time_series: np.ndarray, max_p: int=8, max_q: int=13):
    table = np.full((max_p + 1, max_q + 1), 'x', dtype=object)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARMAModel(p, q)
                model.fit(time_series)
                res_acf = [autocovariance_function(model.residuals[q:], lag) / np.var(model.residuals[q:]) for lag in range(1, q+2)]
                Q = len(model.residuals[q:]) * np.sum(np.square(res_acf))
                p_val = 1 - chi2.cdf(Q, df=q+1)
                table[p, q] = 'o' if p_val > 0.05 else 'x'
            except Exception:
                table[p,q] = '?'

    print("EACF Table (o: not significant, x: significant, ?: error)")
    print("     " + "  ".join([f"q={j}" for j in range(table.shape[1])]))
    for i, row in enumerate(table):
        print(f"p={i}  " + "  ".join(row))

    return table