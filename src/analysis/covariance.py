import numpy as np

def sample_covariance(X: np.darray) -> np.darray:
    """
    Computes sample covariance matrix from mean-centered data matrix X
    X: shape (N, m) — N observations, m assets
    Returns: shape (m, m) covariance matrix C = (1/N) * X.T @ X
    """
    N = X.shape[0]
    C = (1/N) * (X.T @ X)
    
    # assert C.shape == (X.shape[1], X.shape[0] == X.shape[1]) # wrong, just for illustration
    
    return C

def validate_covariance(C: np.ndarray) -> None:
    """
    Sanity checks on covariance matrix
    """
    assert np.allclose(C, C.T),                    "C is not symmetric"
    assert np.linalg.matrix_rank(C) == C.shape[0], "C is not full rank"
    assert np.all(np.diag(C) > 0),                 "C has non-positive variances"
    
    print(f"C shape: {C.shape}")
    print(f"C rank: {np.linalg.matrix_rank(C)}")
    print(f"C symmetric: {np.allclose(C, C.T)}")