# Market Dynamics

A quantitative research project focused on estimating **latent macro market regimes** through statistical representation learning and unsupervised learning.

The project investigates whether technical indicators shown to be informative for supervised price prediction also contain sufficient information to characterize latent market regimes in an unsupervised setting.

Beginning with a macro asset universe consisting of **SPY, QQQ, IWM, TLT, VIX, and DXY**, market-dynamics constructs a comprehensive, literature-informed feature space before applying statistical representation learning techniques (e.g., PCA and related methods) to discover meaningful market structure.

The resulting latent market representation is clustered into discrete market regimes and modeled as a Markov transition process. Rather than directly predicting asset prices, the long-term objective is to estimate the current macro market state and provide it as contextual information for downstream trading strategies, where individual strategies will be statistically validated within the estimated market regime.

## Current Focus

- Literature-informed feature engineering
- Technical indicator generation
- Feature validation and statistical diagnostics
- Statistical representation learning
- Latent market state representation
- Unsupervised market regime detection
- Markov regime transition modeling
- Interactive visualization and dashboard development

## Current Research Pipeline

```text
Macro Asset Universe
        │
        ▼
Technical Indicator Generation
(Literature + TA-Lib)
        │
        ▼
Feature Validation
        │
        ▼
Statistical Representation Learning
(Standardization, PCA, MP Filtering, etc.)
        │
        ▼
Latent Market State Representation
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

## Project Structure

```text
market-dynamics/
├── notebooks/
│   └── ccxt.ipynb                  # Exploratory research and experimentation
│
├── src/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── covariance.py           # Covariance and correlation analysis
│   │   ├── mp.py                   # Marchenko–Pastur noise analysis
│   │   └── pca.py                  # PCA computation and transformations
│   │
│   └── data/
│       ├── __init__.py
│       ├── fetcher.py              # Market data collection
│       ├── models.py               # Data models and schema definitions
│       ├── processor.py            # Data preprocessing and indicator generation
│       └── utils.py                # Shared data-layer utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Fixtures and dynamic pytest test generation
│   ├── test_data.py                # Data pipeline tests
│   └── test_ta_lib.py              # TA-Lib indicator mapping and execution tests
│
├── .gitignore
├── config.py                       # Project-wide configuration
├── main.py                         # Main research workflow / entry point
├── pytest.ini                      # Pytest configuration and custom markers
├── requirements.txt                # Python dependencies
└── README.md
```

## Methodology & Research Architecture

### Core Research Question
> **What unsupervised feature representation best captures the latent macro market state for robust regime modeling?**

Given the macro asset universe: `[SPY, QQQ, IWM, TLT, VIX, DXY]`, this project systematically evaluates high-dimensional feature spaces to isolate clean macro market state representations.

### Research Inspiration & Hypothesis
This architecture replicates and extends the feature space introduced by Mostafavi & Hooman (2025) in *Machine Learning with Applications*:
* [Key Technical Indicators for Stock Market Prediction (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2666827025000143)

While the original paper utilizes **88 technical indicators** (spanning momentum, trend, volatility, and volume) for *supervised price forecasting*, **market-dynamics** shifts the paradigm to an unsupervised setting based on a central working hypothesis:

> **Working Hypothesis:** Technical indicators identified as highly informative for supervised price prediction contain sufficient structural information to characterize latent macro market regimes in an unsupervised representation space.

## Roadmap

### Phase 1 — Data Foundation
- [x] Market data collection
- [x] Data preprocessing framework
- [x] Feature engineering framework

### Phase 2 — Feature Engineering Research
- [ ] Literature review on technical indicator representations
- [ ] Construct comprehensive technical indicator feature space
- [ ] Feature validation & statistical diagnostics
- [ ] Dimensionality reduction research (PCA, MP filtering, etc.)
- [ ] Finalize market state representation


## Philosophy

market-dynamics treats market state estimation as a problem separate from trading.

Rather than directly predicting prices or generating trading signals, the project first seeks to estimate the latent macro market regime. This estimated market state serves as contextual information for downstream trading strategies, allowing market understanding, strategy generation, and statistical validation to remain independent components of the research pipeline.

## Status

Current work is focused on constructing a literature-informed feature space and investigating statistical methods for learning latent macro market representations.
