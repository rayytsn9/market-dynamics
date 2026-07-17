import numpy as np


def mp_support(gamma, sigma2=1.0):
    a = sigma2 * (1 - np.sqrt(gamma))**2
    b = sigma2 * (1 + np.sqrt(gamma))**2
    return a, b

def mp_density(X, gamma, sigma2=1.0, a=None, b=None):
    if a is None or b is None:
        a, b = mp_support(gamma, sigma2)
    
    pdf = np.zeros(X.shape)
    support = (X >= a) & (X <= b)
    pdf[support] = (np.sqrt((b - X[support]) * (X[support] - a))) / \
                   (2 * np.pi * sigma2 * gamma * X[support])
    return pdf

def estimate_sigma2(eigvals: np.ndarray) -> float:
    """Estimates sigma^2 as mean of eigenvalues = trace(C)/m"""
    return np.mean(eigvals)

def get_outliers(eigvals: np.darray, b: float) -> np.darray:
    return eigvals[eigvals > b]