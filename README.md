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

## Methodology & Research Architecture

### Core Research Question
> **What unsupervised feature representation best captures the latent macro market state for robust regime modeling?**

Given the macro asset universe: `[SPY, QQQ, IWM, TLT, VIX, DXY]`, this project systematically evaluates high-dimensional feature spaces to isolate clean macro market state representations.

### Research Inspiration & Hypothesis
This architecture replicates and extends the feature space introduced by Mostafavi & Hooman (2025) in *Machine Learning with Applications*:
* [Key Technical Indicators for Stock Market Prediction (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2666827025000143)

While the original paper utilizes **88 technical indicators** (spanning momentum, trend, volatility, and volume) for *supervised price forecasting*, **market-dynamics** shifts the paradigm to an unsupervised setting based on a central working hypothesis:

> **Working Hypothesis:** Technical indicators identified as highly informative for supervised price prediction contain sufficient structural information to characterize latent macro market regimes in an unsupervised representation space.

---

## Unsupervised Feature Evaluation Framework

To determine which of the 88 technical indicators provide the highest structural signal for macro regime modeling without relying on a noisy price-prediction target, the pipeline executes a two-track unsupervised evaluation:

### Track 1: Individual Asset Analysis
* **Objective:** Isolate a single asset (e.g., `SPY`) and its corresponding 88 technical indicators.
* **Methodology:** Apply unsupervised representation learning and clustering models directly to the local asset feature matrix.
* **Evaluation:** Identify which specific indicator categories (momentum, trend, volatility, volume) drive local cluster boundaries and regime formation for that independent asset.

### Track 2: Cross-Asset Macro Analysis
* **Objective:** Concatenate the technical indicator spaces of all 6 macro assets simultaneously to construct a high-dimensional, 528-feature macro state matrix.
* **Methodology:** Run Principal Component Analysis (PCA) across the entire systemic matrix to observe cross-asset variance capture.
* **Evaluation:** Analyze the absolute mathematical values of the **Principal Component Loadings Matrix** to objectively rank which indicator categories dominate systemic, global macroeconomic structural shifts.

---

## Mathematical Evaluation Metrics (Unsupervised Validation)

Because this pipeline estimates macro market regimes and transition dynamics rather than standalone trading signals, feature representation quality is validated using purely unsupervised mathematical metrics:

1. **Cluster Separability:** Evaluating feature subsets by measuring resulting cluster boundaries via **Silhouette Scores** and **Davies-Bouldin Indices**.
2. **Regime Longevity & Persistence:** Constructing a discrete, first-order **Markov Transition Matrix** to mathematically evaluate the temporal stability ($P_{ii}$ diagonal persistence) of the mapped states.

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

Current work is focused on constructing a literature-informed feature representation of the macro market and investigating whether unsupervised representation learning can recover meaningful market regimes.
