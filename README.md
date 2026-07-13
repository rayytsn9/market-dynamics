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

This project will evaluate two methodologies for feature engineering before selecting the final representation.

---

### Method 1 — Curated Feature Engineering

Research and engineer a set of features that are hypothesized to capture meaningful characteristics of the macro market.

Examples of latent concepts include:

- Trend
- Momentum
- Volatility
- Risk Sentiment
- Relative Strength
- Cross-Asset Relationships

The objective is to build a compact, interpretable feature set driven by financial intuition.

---

### Method 2 — Kitchen Sink (TA-Lib)

Generate a comprehensive feature space by computing as many relevant technical indicators as possible using TA-Lib.

Rather than manually selecting indicators, this approach allows the data to determine which combinations of features contain meaningful structure.

The resulting feature space will then be analyzed using statistical techniques such as:

- Covariance analysis
- Principal Component Analysis (PCA)
- Marchenko–Pastur filtering
- Other dimensionality reduction or covariance-cleaning methods

The objective is to determine whether a broad, data-driven representation captures the macro market state more effectively than manually engineered features.

---

### Method 3 — Comparative Evaluation

Compare both feature representations using quantitative and qualitative evaluation criteria, including:

- Regime quality
- Regime stability
- Interpretability
- Downstream usefulness for trading decisions

The methodology demonstrating the strongest empirical performance will become the feature engineering pipeline for market-dynamics.

---

### Future Pipeline

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
      Statistical Processing
 (Scaling, PCA, MP, etc.)
                │
                ▼
      Market State Representation
                │
                ▼
         Regime Detection
                │
                ▼
      Transition Modeling
                │
                ▼
      Decision Support System
```

## Installation

Clone the repository:

```bash
git clone https://github.com/rayytsn9/market-dynamics.git
cd market-dynamics
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main research workflow:

```bash
python main.py
```

Launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

Run the tests:

```bash
pytest
```

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
