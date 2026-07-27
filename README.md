# Systematic Equity Alpha Generation using Machine Learning

An end-to-end quantitative research framework for forecasting **cross-sectional daily stock returns** in the S&P 500 using machine learning. The project combines point-in-time fundamental data from SEC EDGAR XBRL filings with technical market features to generate alpha forecasts and evaluate systematic investment strategies under realistic trading assumptions.

---

## Overview

This repository implements the complete quantitative research pipeline:

* Point-in-time data collection and preprocessing
* Feature engineering from accounting and market data
* Machine learning alpha models
* Walk-forward validation
* Portfolio construction
* Transaction cost modelling
* Performance attribution and risk analytics

The objective is not simply to maximise predictive accuracy, but to evaluate whether machine learning signals translate into economically meaningful investment performance after realistic implementation constraints.

---

## Research Pipeline

```text
SEC EDGAR XBRL
        │
        ▼
Feature Engineering
(100+ Fundamental & Technical Features)
        │
        ▼
Cross-sectional ML Models
(Ridge • LightGBM • CatBoost • XGBoost)
        │
        ▼
Daily Alpha Forecasts
        │
        ▼
Portfolio Construction
(Long-Only / Long-Short)
        │
        ▼
Walk-Forward Backtest
        │
        ▼
Performance & Risk Analysis
```

---

## Dataset

### Universe

* S&P 500 constituents
* Daily observations
* 2007–2026
* Daily portfolio rebalancing

### Data Sources

* SEC EDGAR XBRL financial statements
* Daily adjusted market prices
* Trading volume
* Market capitalisation
* Sector classifications

All accounting variables are processed using a **strict point-in-time methodology**, ensuring that only information available at each decision date is used during model training and portfolio construction.

---

## Feature Engineering

The framework constructs more than **100 predictive features**, including:

### Fundamental

* Valuation ratios
* Profitability metrics
* Earnings quality
* Growth indicators
* Leverage
* Liquidity
* Cash flow metrics

### Market

* Momentum
* Mean reversion
* Volatility
* Volume
* Relative strength
* Price trend statistics

Features are standardised cross-sectionally and prediction scores are ranked **within sectors**, reducing unintended sector exposures during portfolio construction.

---

## Models

The repository currently implements

* Ridge Regression
* LightGBM
* CatBoost
* XGBoost

The tree-based models capture nonlinear relationships between accounting information and future stock returns, while Ridge regression provides an interpretable linear benchmark.

---

## Validation Methodology

To avoid look-ahead bias, all experiments follow an **expanding-window walk-forward framework**.

* Models are trained only on information available up to the prediction date.
* Retraining is performed annually using the expanding historical sample.
* Portfolios are rebalanced daily using out-of-sample forecasts.

Model performance is evaluated through cross-sectional prediction accuracy rather than in-sample fit.

---

## Portfolio Construction

Two investment strategies are implemented.

### Long-Only

* Select top-ranked securities
* Daily rebalancing
* Fully invested portfolio

### Long-Short

* Long top decile
* Short bottom decile
* Dollar-neutral construction
* Daily rebalancing

Supported weighting schemes include:

* Equal Weight
* Risk Parity
* Score Weighting
* Score per Volatility
* Mean-Variance (Markowitz) Optimisation

---

## Transaction Costs

Portfolio turnover is explicitly modelled.

Transaction costs are applied according to


Net Return = Gross Return - c × Turnover

where

* (c) is the transaction cost per trade,
* turnover is computed from changes in portfolio weights.

This allows gross and net performance to be compared under realistic implementation assumptions.

---

## Evaluation Metrics

### Predictive Performance

* Information Coefficient (IC)
* Rank IC
* Annual IC analysis

### Portfolio Performance

* Annual return
* Annual volatility
* Sharpe ratio
* Maximum drawdown
* Daily turnover
* Gross and net cumulative returns

---

## Repository Structure

```text
data/
│
├── raw/
├── processed/

models/
│
├── ridge.py
├── lightgbm.py
├── catboost.py
└── xgboost.py

features/
│
├── fundamentals.py
├── technical.py

portfolio/
│
├── weighting.py
├── backtest.py

notebooks/

results/

paper/
```

---

## Current Results

Out-of-sample testing over 2017–2025 shows:

* Mean Information Coefficient ≈ **2%**
* Consistent positive predictive performance across market regimes
* Strong gross risk-adjusted returns for both long-only and market-neutral portfolios
* Significant performance degradation after realistic transaction costs, highlighting the importance of turnover-aware portfolio construction

The accompanying research paper discusses these results in detail and analyses the trade-off between predictive power and implementation costs.

---

## Future Work

Planned extensions include

* Bayesian hyperparameter optimisation
* Feature selection with SHAP values
* Turnover-constrained portfolio optimisation
* Dynamic position sizing
* Alternative data integration
* Multi-horizon forecasting
* Transformer-based architectures
* Graph neural networks

---

## References

* Fama, E. F. & French, K. R. (1993). *Common Risk Factors in the Returns on Stocks and Bonds.*
* Gu, Kelly & Xiu (2020). *Empirical Asset Pricing via Machine Learning.*
* LightGBM
* CatBoost
* XGBoost
