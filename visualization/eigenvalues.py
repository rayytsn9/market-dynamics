import numpy as np
import matplotlib.pyplot as plt

def plot_eigenvalues(eigvals: np.ndarray, gamma: float, sigma2: float,
                     a: float, b: float, title: str = 'Eigenvalue Distribution vs Marchenko-Pastur Law') -> plt.Figure:
    """
    Plots eigenvalue histogram with MP density overlay
    """
    from analysis.mp import mp_density

    fig, ax = plt.subplots(figsize=(10, 6))

    # histogram
    ax.hist(eigvals, density=True, bins=50, alpha=0.7,
            color='steelblue', label='Empirical eigenvalues')

    # MP density
    x_grid = np.linspace(a * 0.5, b * 1.5, 2000)
    pdf    = mp_density(x_grid, gamma, sigma2, a, b)
    ax.plot(x_grid, pdf, 'r-', linewidth=2, label=f'MP fit (σ²={sigma2:.4f}, γ={gamma:.2f})')

    # support edges
    ax.axvline(b, color='green',  linestyle='--', linewidth=1.5, label=f'λ+ = {b:.4f}')
    ax.axvline(a, color='orange', linestyle='--', linewidth=1.5, label=f'λ- = {a:.4f}')

    # outliers
    outliers = eigvals[eigvals > b]
    for val in outliers:
        ax.axvline(val, color='red', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    return fig


def plot_scree(eigvals: np.ndarray, b: float, title: str = 'Scree Plot') -> plt.Figure:
    """
    Scree plot of eigenvalues with MP upper edge marked
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(eigvals)+1), eigvals, 'o-', markersize=4, color='steelblue')
    ax.axhline(b, color='green', linestyle='--', linewidth=1.5, label=f'λ+ = {b:.4f}')
    ax.fill_between(range(1, len(eigvals)+1), eigvals, b,
                    where=(eigvals > b), alpha=0.3, color='red', label='Signal')
    ax.fill_between(range(1, len(eigvals)+1), eigvals, 0,
                    where=(eigvals <= b), alpha=0.2, color='gray', label='Noise')

    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title)
    ax.legend()

    return fig