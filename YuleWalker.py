import numpy as np
from Core import autocovariance_function

def _build_toeplitz(acvf_vec: np.ndarray) -> np.ndarray:
    '''
    Construct our Gamma matrix, a Toeplitz matrix, from the vector of autocovariances

    Parameters:
        acvf_vec: Autocovariances for lags 0 to p, inclusive

    Returns:
        np.ndarray: Gamma matrix of shape p by p
    '''
    
    n = len(acvf_vec)
    return np.array([[acvf_vec[abs(i-j)] for j in range(n)] for i in range(n)])

def yule_walker(time_series: np.ndarray, p: int) -> tuple[np.ndarray, float]:
    '''
    Estimate the AR(p) parameters using the Yule-Walker equations

    Parameters:
        time_series (np.ndarray): The input time series
        p (int): Order of our AR process

    Returns:
        phi_vector (np.ndarray): Estimated AR coefficients (has length p)
        variance (float): Estimated white noise variance
    '''

    # Calculate the autocovariance of our time series for lags 0 through p
    acvf = np.array([autocovariance_function(time_series, lag) for lag in range(p+1)])

    # Use the equation Γ * Φ = γ for Yule-Walker equations, solve for Φ

    # Our Γ built as a Toeplitz matrix
    Gamma_matrix = _build_toeplitz(acvf)
    # Our γ vector of lags 1 to p
    gamma_vector = acvf[1:]
    # Solve Γ * Φ = γ
    phi_vector = np.linalg.solve(Gamma_matrix, gamma_vector)
    # Estimate our variance as σ² = γ(0) - φᵗγ
    variance = acvf[0] - np.dot(phi_vector, gamma_vector)

    return phi_vector, variance