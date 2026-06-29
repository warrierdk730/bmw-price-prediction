# 🚗 BMW Used Car Price Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Machine learning project to predict used BMW car prices using regression models — achieving R² of 0.9553 with Random Forest on 10,781 real UK listings across 24 models, 1996–2020.**

---

## 📌 Project Overview

| Detail | Value |
|--------|-------|
| **Dataset** | BMW Used Car Sales — Kaggle (UK market) |
| **Period** | 1996 – 2020 |
| **Records** | 10,781 listings |
| **BMW Models** | 24 (1 Series to X7, M-series, i3, i8, Z-series) |
| **Target Variable** | Car price (£) |
| **Best Model** | Random Forest — R² 0.9553, MAE £1,454 |
| **Baseline Model** | Linear Regression — R² 0.9037, MAE £2,286 |

---

## 🎯 Problem Statement

Buyers and sellers of used BMWs often struggle to determine a **fair market price**.  
Asking prices vary widely based on age, mileage, engine size, model, and transmission — making it difficult to spot overpriced or underpriced listings without a data-driven benchmark.

This project builds a **price prediction model** that estimates the fair value of any used BMW based on its specifications.

---

## 🔑 Key Results

| Model | R² Score | MAE | Notes |
|-------|----------|-----|-------|
| Linear Regression | 0.9037 | £2,286 | Strong baseline — 90% variance explained |
| Random Forest (200 trees) | **0.9553** | **£1,454** | Best performance |
| Tuned RF (500 trees) | 0.9442 | £1,583 | Reduced overfitting |
| 5-Fold CV Mean R² | **0.9321 ± 0.019** | — | Confirms robust generalisation |

> **The final model predicts used BMW prices with an average error of just £1,454** — less than 7% of the median car price (£20,462).

---

## 🔍 Key Findings

- 📅 **Year of registration is the #1 price driver** — accounts for ~26% of feature importance. Depreciation is the single biggest factor in used BMW pricing.
- 🔧 **Engine size ranks 3rd** — larger engines (M-series, X5, 7 Series) command significant premiums.
- 🛠️ **Feature engineering added real value** — `car_age`, `mileage_per_year`, and `engine_age_interaction` all rank in the top 10 most important features.
- 🌲 **Random Forest significantly outperforms Linear Regression** — R² improves from 0.90 to 0.96, confirming non-linear depreciation curves in the data.
- 🚘 **Semi-Auto transmission commands a £5,000–£8,000 premium** over equivalent manual cars.
- 💰 **Model accuracy varies by price segment** — predictions are more reliable for budget/economy cars (≤ £30K, 81.9% of dataset) than for luxury models (> £30K).
- 🏆 **Top 3 most expensive models on average:** X7 (£69,843), 8 Series (£63,998), M5 (£57,760).

---

## 🛠️ Feature Engineering

Three new features were engineered to improve model performance:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `car_age` | `2024 - year` | More intuitive for depreciation than raw year |
| `mileage_per_year` | `mileage / car_age` | Captures usage intensity — a 5yr/100K car differs from 5yr/20K |
| `engine_age_interaction` | `engineSize × car_age` | Larger older engines depreciate faster |

All three ranked in the **top 10 most important features** in the final model.

---

## 📁 Project Structure

```
bmw-price-prediction/
│
├── bmw.csv                        # Raw dataset (10,781 UK listings)
├── BMW_Price_Prediction.ipynb     # Full analysis notebook (10 sections)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 📓 Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Data Loading | Imports, load dataset, shape overview |
| 2 | Data Cleaning | Engine size = 0 fix, outlier removal, quality report |
| 3 | Exploratory Data Analysis | Price distribution, price by model, scatter plots, correlation heatmap |
| 4 | Feature Engineering | car_age, mileage_per_year, engine_age_interaction with visualisations |
| 5 | Baseline — Linear Regression | R² 0.90, predicted vs actual, residual plot |
| 6 | Random Forest Model | R² 0.9553, predicted vs actual, residual comparison |
| 7 | Tuning & Cross-Validation | 500-tree tuned RF, 5-fold CV, model comparison chart |
| 8 | Feature Importance | Top 15 features bar chart, cumulative importance curve |
| 9 | Price Segment Analysis | Separate models for ≤£30K vs >£30K segments |
| 10 | Key Findings & Conclusions | Summary table, business applications, future improvements |

---

## 📊 Charts Generated

| Chart | Description |
|-------|-------------|
| `01_distributions.png` | Price and mileage distribution histograms |
| `02_price_bands_models.png` | Price band breakdown + average price by model |
| `03_price_relationships.png` | Price vs year/mileage scatter + boxplots by transmission & fuel |
| `04_correlation_heatmap.png` | Correlation matrix — numeric features |
| `05_engineered_features.png` | Engineered features vs price scatter plots |
| `06_linear_regression_eval.png` | LR predicted vs actual + residual plot |
| `07_random_forest_eval.png` | RF predicted vs actual + residual plot |
| `08_model_comparison.png` | R² comparison across all models |
| `09_feature_importance.png` | Top 15 features + cumulative importance |
| `10_segment_analysis.png` | Performance comparison by price segment |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **Pandas** | Data loading, cleaning, feature engineering |
| **NumPy** | Numerical operations |
| **Matplotlib** | Chart rendering |
| **Seaborn** | Statistical visualisations (boxplots, heatmaps) |
| **Scikit-learn** | Linear Regression, Random Forest, cross-validation, metrics |
| **Jupyter Notebook** | Analysis environment |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/warrierdk730/bmw-price-prediction.git
cd bmw-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook BMW_Price_Prediction.ipynb
```

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
jupyter
```

---

## 💡 Skills Demonstrated

- ✅ Real-world **data cleaning** — handling engine size anomalies and price outliers
- ✅ **Exploratory Data Analysis** — distributions, correlations, categorical comparisons
- ✅ **Feature engineering** — creating meaningful derived features from raw data
- ✅ **Model comparison** — Linear Regression vs Random Forest, with residual analysis
- ✅ **Hyperparameter tuning** — reducing overfitting through constrained tree depth
- ✅ **Cross-validation** — 5-fold CV for robust performance estimation
- ✅ **Feature importance analysis** — understanding what drives used car prices
- ✅ **Price segmentation** — separate model evaluation by market segment
- ✅ **Data storytelling** — every chart and result explained with business context

---

## 🔜 Future Improvements

- Add **colour** and **number of previous owners** — both known to significantly affect used car prices
- Train on **post-2022 data** to capture post-COVID used car market dynamics
- Build a **Streamlit price estimator app** — input car specs, get instant predicted price
- Try **XGBoost or LightGBM** — gradient boosting often outperforms Random Forest on tabular regression
- Add **confidence intervals** to predictions for buyer/seller risk assessment

---

## 👤 Author

**Dileep Kumar Warrier**
- 📧 warrierdk.1985@gmail.com
- 💼 [LinkedIn](https://in.linkedin.com/in/warrier-dk-70b68420)
- 🐙 [GitHub](https://github.com/warrierdk730)

---

## 📄 Dataset Source

[BMW Used Car Dataset — Kaggle](https://www.kaggle.com/)  
*UK market listings from 1996 to 2020.*

---

*⭐ If you found this project useful, please star the repository!*
