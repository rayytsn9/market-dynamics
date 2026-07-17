from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import numpy as np

def pca_statistics(X: np.ndarray, n_components: int) -> dict:
    pca = PCA(n_components=n_components) # fit PCA with n_outliers components
    projected = pca.fit_transform(X)  # shape (100, 3)
    PC = pca.components_.T  # shape (56, 3) PC 
    eigvals_pca = pca.explained_variance_  # shape (3,)

    return {'pca': pca, 'projected': projected, 'PC': PC, 'eigvals_pca': eigvals_pca}

def svd_statistics(X: np.ndarray, n_components: int) -> dict:

    svd = TruncatedSVD(n_components=n_components)
    projected = svd.fit_transform(X)      # shape (100, 3) — projected data
    PC = svd.components_.T                # shape (56, 3) — principal components
    eigvals_svd = svd.explained_variance_ # shape (3,) — eigenvalues

    return {'svd': svd, 'projected': projected, 'PC': PC, 'eigvals_svd': eigvals_svd}

def get_eigenpair(C: np.ndarray) -> tuple:
    """
    Computes eigenvalues and eigenvectors of covariance matrix C
    Returns eigvals, eigvecs sorted in descending order
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs

def extract_factors(eigvecs: np.ndarray, n_factors: int) -> np.ndarray:
    """
    Extracts top n_factors eigenvectors as principal components
    Returns PC matrix of shape (m, n_factors)
    """
    return eigvecs[:, :n_factors]

def project(X: np.ndarray, PC: np.ndarray) -> np.ndarray:
    """
    Projects data matrix X onto PC space
    X: (N, m), PC: (m, n_factors)
    Returns projected scores of shape (N, n_factors)
    """
    return X @ PC