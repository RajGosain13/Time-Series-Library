import numpy as np
import matplotlib.pyplot as plt

def plot_series(time_series: np.ndarray, title: str = 'Time Series', xlabel: str = 'Time', ylabel: str = 'Value'):
    plt.figure(figsize=(10, 4))
    plt.plot(time_series, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_acf(acf_vals: np.ndarray, title: str = "Autocorrelation Function"):
    lags = np.arange(len(acf_vals))
    plt.figure(figsize=(8, 4))
    plt.stem(lags, acf_vals, basefmt=" ", use_line_collection=True)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pacf(pacf_vals: np.ndarray, title: str = "Partial Autocorrelation Function"):
    lags = np.arange(len(pacf_vals))
    plt.figure(figsize=(8, 4))
    plt.stem(lags, pacf_vals, basefmt=" ", use_line_collection=True)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_forecast(true_series: np.ndarray, forecast: np.ndarray, stderr: float = None, title: str = "Forecast with 95% CI"):
    n_true = len(true_series)
    n_forecast = len(forecast)
    time = np.arange(n_true + n_forecast)

    plt.figure(figsize=(10, 4))
    plt.plot(time[:n_true], true_series, label="Observed", marker='o')
    plt.plot(time[n_true:], forecast, label="Forecast", marker='o', color='orange')

    if stderr is not None:
        ci_upper = forecast + 1.96 * stderr
        ci_lower = forecast - 1.96 * stderr
        plt.fill_between(time[n_true:], ci_lower, ci_upper, color='orange', alpha=0.3, label="95% CI")

    plt.axvline(x=n_true - 1, linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()