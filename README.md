# Cross-Sectional Alpha Generation Pipeline

An end-to-end quantitative research framework designed to predict cross-sectional returns on S\&P 500 equities using  using fundamental and technical alpha factors spanning value, quality, momentum, volatility, volume, and market structure with regression models. This architecture is optimized specifically to maximize the **Rank Information Coefficient (Rank IC)** while minimizing turnover and tracking error.


## 🚀 Architecture Overview

This repository implements a robust, modular machine learning pipeline built to process financial time-series data without looking-ahead or data leakage. 

* **Objective Function**: Continuous Cross-Sectional Regression (Targeting Spearman's Rank Correlation / IC).
* **Validation Strategy**: Strict Out-of-Fold (OOF) validation using `TimeSeriesSplit` to mimic a realistic out-of-sample forward-test.
* **Feature Engineering Pipeline**: 
  * Adaptive handling of structural features (e.g., sectors and industries) using robust encoders.
  * Robust, median-imputed data scaling wrapped inside sklearn pipelines to eliminate data contamination during validation folds.

## 🧠 Models Implemented

The architecture leverages an ensemble approach, combining tree-based architectures with linear regularization models:
1. **LightGBM Regressor**: Optimized for low-latency non-linear feature interactions with gradient-based one-side sampling.
2. **CatBoost Regressor**: Configured specifically to natively process categorical asset identifiers (sectors/industries) without symmetric bias.
3. **Ridge Regression**: Serves as a robust linear baseline, enforcing L2 regularization to counter cross-sectional multi-collinearity.

## 📈 Metric Evaluation: Mean Cross-Sectional Rank IC

Unlike traditional machine learning models tracking global MSE or Accuracy, this pipeline evaluates performance on a **per-date cross-section**. This isolation represents true trading desk performance by measuring the monotonic relationship between the signal's ranking and actual forward asset performance.

$$\text{Rank IC}_t = \rho_{\text{Spearman}}(\hat{y}_{t}, y_{t})$$

## 🛠️ Tech Stack
* **Languages**: Python
* **Machine Learning**: LightGBM, CatBoost, Scikit-Learn
* **Data Processing**: NumPy, Pandas, SciPy (Signal/Stats)
