import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

REGIME_COLORS = {'bull': 'green', 'bear': 'red', 'neutral': 'gray'}
DARK_BG = 'rgb(10, 10, 20)'

def _get_point_colors(regimes):
    return regimes['regime'].map(REGIME_COLORS).tolist()

def _get_hover_text(projected, regimes, dates, n_factors):
    texts = []
    for i, d in enumerate(dates):
        pc_str = '<br>'.join([f'PC{j+1}: {projected[i,j]:.3f}' for j in range(n_factors)])
        texts.append(f'{d}<br>{pc_str}<br>Regime: {regimes["regime"].iloc[i]}')
    return texts


def plot_1d(projected, regimes, index) -> go.Figure:
    dates        = index.strftime('%Y-%m-%d').tolist()
    point_colors = _get_point_colors(regimes)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=index,
        y=projected[:, 0],
        mode='lines+markers',
        marker=dict(color=point_colors, size=6),
        line=dict(color='steelblue', width=1),
        text=[f'{d}<br>PC1: {projected[i,0]:.3f}<br>Regime: {regimes["regime"].iloc[i]}'
              for i, d in enumerate(dates)],
        hoverinfo='text',
        name='PC1 Score'
    ))
    fig.add_hline(y=0, line=dict(color='white', width=1, dash='dash'), opacity=0.3)
    fig.update_layout(
        title='Factor Space (1D)',
        xaxis_title='Date',
        yaxis_title='PC1 (Market Mode)',
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color='white')
    )
    return fig


def plot_2d(projected, regimes, today, index) -> go.Figure:
    dates        = index.strftime('%Y-%m-%d').tolist()
    point_colors = _get_point_colors(regimes)
    hover_text   = _get_hover_text(projected, regimes, dates, 2)

    fig = go.Figure()

    # convex hull outline
    try:
        hull     = ConvexHull(projected[:, :2])
        hull_pts = np.append(hull.vertices, hull.vertices[0])
        fig.add_trace(go.Scatter(
            x=projected[hull_pts, 0],
            y=projected[hull_pts, 1],
            mode='lines',
            line=dict(color='lightblue', width=1.5),
            opacity=0.4,
            name='Hull',
            hoverinfo='skip'
        ))
    except Exception as e:
        print(f"2D hull failed: {e}")

    # historical points
    fig.add_trace(go.Scatter(
        x=projected[:, 0],
        y=projected[:, 1],
        mode='markers',
        marker=dict(color=point_colors, size=6, opacity=0.6),
        text=hover_text,
        hoverinfo='text',
        name='Trading Days'
    ))

    # today
    fig.add_trace(go.Scatter(
        x=[float(today['PC1'])],
        y=[float(today['PC2'])],
        mode='markers',
        marker=dict(size=12, color='yellow', symbol='diamond',
                    line=dict(color='black', width=2)),
        text=[f"TODAY<br>Regime: {today['regime']}<br>Confidence: {float(today['confidence']):.1%}"],
        hoverinfo='text',
        name='Today'
    ))

    fig.add_hline(y=0, line=dict(color='white', width=1, dash='dash'), opacity=0.2)
    fig.add_vline(x=0, line=dict(color='white', width=1, dash='dash'), opacity=0.2)
    fig.update_layout(
        title='Factor Space (2D)',
        xaxis_title='PC1 (Market Mode)',
        yaxis_title='PC2',
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color='white')
    )
    return fig


def plot_3d(projected, regimes, today, index) -> go.Figure:
    dates        = index.strftime('%Y-%m-%d').tolist()
    point_colors = _get_point_colors(regimes)
    hover_text   = _get_hover_text(projected, regimes, dates, 3)

    fig = go.Figure()

    # convex hull mesh
    try:
        hull = ConvexHull(projected[:, :3])
        fig.add_trace(go.Mesh3d(
            x=projected[:, 0],
            y=projected[:, 1],
            z=projected[:, 2],
            i=[s[0] for s in hull.simplices],
            j=[s[1] for s in hull.simplices],
            k=[s[2] for s in hull.simplices],
            opacity=0.08,
            color='lightblue',
            hoverinfo='skip',
            name='Hull'
        ))
    except Exception as e:
        print(f"3D hull failed: {e}")

    # historical points
    fig.add_trace(go.Scatter3d(
        x=projected[:, 0],
        y=projected[:, 1],
        z=projected[:, 2],
        mode='markers',
        marker=dict(size=4, color=point_colors, opacity=0.6),
        text=hover_text,
        hoverinfo='text',
        name='Trading Days'
    ))

    # today
    if all(f'PC{i+1}' in today for i in range(3)):
        fig.add_trace(go.Scatter3d(
            x=[float(today['PC1'])],
            y=[float(today['PC2'])],
            z=[float(today['PC3'])],
            mode='markers',
            marker=dict(size=12, color='yellow', symbol='diamond',
                        line=dict(color='black', width=2)),
            text=[f"TODAY<br>Regime: {today['regime']}<br>Confidence: {float(today['confidence']):.1%}"],
            hoverinfo='text',
            name='Today'
        ))

    fig.update_layout(
        title='Crypto Factor Space (3D)',
        scene=dict(
            xaxis_title='PC1 (Market Mode)',
            yaxis_title='PC2',
            zaxis_title='PC3',
            bgcolor=DARK_BG,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            zaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
        ),
        paper_bgcolor=DARK_BG,
        font=dict(color='white'),
        height=800
    )
    return fig


def plot_factor_space(projected, scores, regimes, today, n_factors, index) -> go.Figure:
    """
    Main entry point — dispatches to correct plot based on n_factors
    """
    if n_factors == 1:
        return plot_1d(projected, regimes, index)
    elif n_factors == 2:
        return plot_2d(projected, regimes, today, index)
    else:
        return plot_3d(projected, regimes, today, index)