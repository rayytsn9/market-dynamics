# Market Dynamics

A quantitative research pipeline project for exploring latent structure and regime behavior in financial markets using Principal Component Analysis.

The project currently focuses on transforming multivariate market data into a lower-dimensional representation, separating meaningful components from noise, interpreting factor loadings, and visualizing market observations in principal-component space. Eventually, the project aims to be automated to produce dynamic analysis of market data. 

## Current Focus

* Market data collection and preprocessing
* Covariance and correlation analysis
* Principal Component Analysis
* Eigenvalue and explained-variance analysis
* Marchenko–Pastur noise filtering
* Principal-component loading interpretation
* Market regime exploration
* Interactive visualization and dashboard development

## Research Pipeline

```text
Market Data
    ↓
Data Processing / Feature Engineering
    ↓
Covariance Estimation
    ↓
PCA and Noise Analysis
    ↓
Loading Interpretation
    ↓
Regime Exploration
    ↓
Visualization and Dashboard
```

## Project Structure

```text
market-dynamics/
├── analysis/
│   ├── covariance.py        # Covariance and correlation analysis
│   ├── mp.py                # Marchenko–Pastur noise analysis
│   └── pca.py               # PCA computation and transformations
│
├── data/
│   ├── fetcher.py           # Market data collection
│   └── processor.py         # Data cleaning and preprocessing
│
├── interpretation/
│   ├── loadings.py          # Principal-component loading interpretation
│   └── regimes.py           # Market regime analysis
│
├── visualization/
│   ├── eigenvalues.py       # Eigenvalue and explained-variance plots
│   ├── loadings.py          # Loading visualizations
│   └── projection.py        # PCA projection visualizations
│
├── notebooks/
│   ├── ccxt.ipynb           # Data exploration and experimentation
│   └── *.csv                # Local research datasets
│
├── dashboard.py             # Interactive dashboard
├── main.py                  # Main research workflow
├── config.py                # Project configuration
├── utils.py                 # Shared utility functions
├── test_functions.py        # Tests
├── requirements.txt         # Python dependencies
└── README.md
```

## Methodology

### Current Phase

**Research Question**

> **What feature representation best captures the latent macro market state for regime modeling and downstream trading decisions?**

Given the macro asset universe

```text
[SPY, QQQ, IWM, TLT, VIX, DXY]
```

construct a feature representation of the macro market state.

### Research Inspiration

The current feature engineering phase is inspired in part by the following paper:

> **Key Technical Indicators for Stock Market Prediction**  
> *Machine Learning with Applications* (2025)  
> https://www.sciencedirect.com/science/article/pii/S2666827025000143

The paper investigates the predictive power of **88 technical indicators** categorized into interpretable classes (e.g., **momentum, trend, volatility, and volume**) using supervised machine learning models for S&P 500 price forecasting.

While the paper focuses on **supervised price prediction**, market-dynamics adopts a different objective.

---

### Working Hypothesis

> **Technical indicators identified as informative for supervised price prediction also contain sufficient information to characterize latent macro market regimes in an unsupervised setting.**

Rather than directly predicting future prices, market-dynamics investigates whether these technical indicators can be transformed into a meaningful representation of the macro market state.

---

## Current Research Pipeline

```text
Macro Assets
        │
        ▼
Feature Engineering
        │
        ├── Curated Features
        └── TA-Lib Feature Space
                │
                ▼
(Optional PCA / Statistical Processing)
                │
                ▼
Market State Representation
                │
                ▼
Unsupervised Regime Detection
                │
                ▼
Discrete Market Regimes
                │
                ▼
Markov Transition Model
```

The output of this pipeline is **not** a trading signal or price prediction.

Instead, market-dynamics estimates the current macro market regime and its transition dynamics.

---

## Long-Term Vision

The estimated market regime serves as **context** for downstream trading strategies rather than acting as a trading strategy itself.

Future strategy pipelines will follow the architecture:

```text
Current Market Regime
        │
        ▼
Trading Strategy
        │
        ▼
Hypothesis Testing / Statistical Validation
        │
        ▼
Trading Decision
```

This separation allows multiple independent trading strategies to leverage a common market state estimation engine while keeping market state estimation, strategy generation, and statistical validation as distinct components of the system.


## Roadmap

### Phase 1 — Data Foundation
* [x] Market data collection
* [x] Data preprocessing framework
* [x] Feature engineering framework

### Phase 2 — Feature Engineering Research
* [ ] Curated macro feature engineering
* [ ] Kitchen Sink (TA-Lib) feature generation
* [ ] Feature validation & statistical diagnostics
* [ ] Compare curated vs. TA-Lib feature spaces
* [ ] Finalize market state representation


## Philosophy

The project prioritizes understanding market structure before attempting prediction.

Its purpose is not to label every statistical pattern as a trading signal, but to build an interpretable research foundation for studying common market factors, dimensionality, noise, and changing market conditions.

## Status

This project is under active development.

Current work is centered on PCA-based market structure analysis and the interpretation of potential market regimes.
