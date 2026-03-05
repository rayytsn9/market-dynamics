import streamlit as st
import pandas as pd
import datetime as dt
import config

from data.fetcher import instantiate_exchange, get_top_m_symbols, construct_dataset
from data.processor import build_panel, mean_center
from analysis.covariance import sample_covariance
from analysis.mp import estimate_sigma2, mp_support, get_outliers
from analysis.pca import get_eigenpair, extract_factors, project
from interpretation.regimes import build_scores, classify_regimes, get_today_regime
from visualization.projection import plot_factor_space

st.set_page_config(page_title='Crypto Factor Space', layout='wide')
st.title('Crypto Factor Space — Live')

@st.cache_data(ttl=86400)
def run_pipeline():
    now   = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(hours=24 * (config.N + 1))
    start = start.replace(second=0, microsecond=0, minute=0)
    since = int(start.timestamp() * 1000)

    exchange = instantiate_exchange()
    symbols  = get_top_m_symbols(exchange, config.M_BUFFER, config.STABLECOINS)

    if config.CSV_FILE is not None:
        df       = pd.read_csv(config.CSV_FILE, index_col=0)
        raw_data = {col: df[col] for col in df.columns}
    else:
        raw_data = construct_dataset(exchange, symbols, config.N, since=since)

    panel  = build_panel(raw_data, config.N, config.M)
    X      = mean_center(panel)
    C      = sample_covariance(X)

    eigvals, eigvecs = get_eigenpair(C)
    gamma   = config.M / config.N
    sigma2  = estimate_sigma2(eigvals)
    a, b    = mp_support(gamma, sigma2)
    outliers = get_outliers(eigvals, b)
    n_factors = len(outliers)

    PC        = extract_factors(eigvecs, n_factors)
    projected = project(X, PC)

    index   = pd.to_datetime(panel.index, unit='ms')
    scores  = build_scores(projected, index, n_factors)
    regimes = classify_regimes(scores)
    today   = get_today_regime(scores, regimes)

    return projected, scores, regimes, today, n_factors, index

# ── Run Pipeline ─────────────────────────────────────
with st.spinner('Fetching data...'):
    projected, scores, regimes, today, n_factors, index = run_pipeline()

# ── Plot ─────────────────────────────────────────────
fig = plot_factor_space(projected, scores, regimes, today, n_factors, index)
st.plotly_chart(fig, use_container_width=True)

# ── Today's Regime ───────────────────────────────────
col1, col2, col3 = st.columns(3)
regime_emoji = {'bull': '🟢', 'bear': '🔴', 'neutral': '⚪'}
col1.metric("Regime",     f"{regime_emoji[today['regime']]} {today['regime'].upper()}")
col2.metric("Confidence", f"{float(today['confidence']):.1%}")
col3.metric("PC1 Score",  f"{float(today['PC1']):.4f}")