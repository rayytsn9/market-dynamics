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
Data Processing
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

Principal Component Analysis is used to identify the dominant sources of variation across a collection of financial assets.

Given a standardized return matrix (X), PCA decomposes its covariance or correlation matrix into eigenvalues and eigenvectors:

[
\Sigma = V \Lambda V^\top
]

where:

* (\Lambda) contains the variance explained by each principal component
* (V) contains the corresponding component loadings
* projecting (X) onto the leading eigenvectors produces a lower-dimensional representation of market behavior

The project also uses the Marchenko–Pastur distribution to examine whether observed eigenvalues represent meaningful shared structure or are consistent with random-matrix noise.

The resulting components and projections can then be studied for clusters, transitions, and other patterns associated with changing market conditions.

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

* [x] Market data collection
* [x] Data preprocessing
* [x] Covariance analysis
* [x] PCA implementation
* [x] Eigenvalue visualization
* [x] Principal-component loading analysis
* [x] PCA projection visualization
* [ ] Refine Marchenko–Pastur noise filtering
* [ ] Formalize regime identification
* [ ] Evaluate regime stability over time
* [ ] Add additional assets and market features
* [ ] Improve automated testing
* [ ] Expand the interactive dashboard

## Philosophy

The project prioritizes understanding market structure before attempting prediction.

Its purpose is not to label every statistical pattern as a trading signal, but to build an interpretable research foundation for studying common market factors, dimensionality, noise, and changing market conditions.

## Status

This project is under active development.

Current work is centered on PCA-based market structure analysis and the interpretation of potential market regimes.
