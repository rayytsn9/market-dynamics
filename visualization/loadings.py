import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_loading_bars(pc_loadings: pd.DataFrame, exposure_matrix: pd.DataFrame,
                      title: str = 'Factor Loadings by Asset') -> plt.Figure:
    """
    Bar chart of loadings per factor colored by exposure category
    """
    n_factors = pc_loadings.shape[1]
    fig, axes = plt.subplots(1, n_factors, figsize=(8 * n_factors, 6))

    if n_factors == 1:
        axes = [axes]

    for idx, col in enumerate(pc_loadings.columns):
        loadings = pc_loadings[col].sort_values(ascending=False)

        colors = []
        for coin in loadings.index:
            if exposure_matrix.loc[coin, col] == 'positive':
                colors.append('green')
            elif exposure_matrix.loc[coin, col] == 'negative':
                colors.append('red')
            else:
                colors.append('gray')

        axes[idx].bar(range(len(loadings)), loadings.values, color=colors)
        axes[idx].axhline(y= 0.05, color='green', linestyle='--', alpha=0.5, label='positive threshold')
        axes[idx].axhline(y=-0.05, color='red',   linestyle='--', alpha=0.5, label='negative threshold')
        axes[idx].axhline(y=0,     color='black',  linestyle='-',  alpha=0.3)
        axes[idx].set_xticks(range(len(loadings)))
        axes[idx].set_xticklabels(loadings.index, rotation=90, fontsize=7)
        axes[idx].set_title(col)
        axes[idx].set_ylabel('Loading')
        axes[idx].legend()

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_exposure_heatmap(exposure_matrix: pd.DataFrame,
                          pc_loadings: pd.DataFrame,
                          title: str = 'Factor Exposure Matrix') -> plt.Figure:
    """
    Heatmap of factor exposures per asset
    """
    # convert to numeric for heatmap
    # positive=1, neutral=0, negative=-1
    numeric = exposure_matrix.replace({'positive': 1, 'neutral': 0, 'negative': -1})
    numeric = numeric.loc[pc_loadings.iloc[:, 0].sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, len(numeric) * 0.3 + 2))

    im = ax.imshow(numeric.values.T, aspect='auto', cmap='RdYlGn',
                   vmin=-1, vmax=1)

    ax.set_xticks(range(len(numeric.index)))
    ax.set_xticklabels(numeric.index, rotation=90, fontsize=7)
    ax.set_yticks(range(len(numeric.columns)))
    ax.set_yticklabels(numeric.columns, fontsize=9)

    plt.colorbar(im, ax=ax, ticks=[-1, 0, 1],
                 label='negative | neutral | positive')
    ax.set_title(title)
    plt.tight_layout()

    return fig