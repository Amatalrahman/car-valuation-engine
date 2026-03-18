# 🚗 AutoValuate — AI Car Price Prediction

A machine learning project that predicts used car market values using **188,000+ vehicle records**. Built with LightGBM, tuned with Optuna, and explained with SHAP.

---

## 🌐 Live App

**Anyone can use the app instantly — no installation, no account needed:**

👉 **[https://car-valuation-engine-pqr5cynftm27jrewuflb7h.streamlit.app/](https://car-valuation-engine-pqr5cynftm27jrewuflb7h.streamlit.app/)**

Just open the link, configure your vehicle in the sidebar, and get an instant price estimate with a full breakdown.

---

## 📸 App Preview

<!-- Add your screenshot here -->
<img width="1918" height="912" alt="Image" src="https://github.com/user-attachments/assets/c5b5e9e8-0571-4c3a-a175-f5ad1970ed86" />

<img width="374" height="911" alt="Image" src="https://github.com/user-attachments/assets/5f5acbec-fedb-4f26-8a3a-1446ad3d598b" />


<img width="1372" height="493" alt="Image" src="https://github.com/user-attachments/assets/185411c4-c03e-47fa-876e-4a6b4c1b1bd3" />

---

## 📂 Project Structure

```
├── 01_EDA.ipynb                     # Data loading & exploratory analysis
├── 02_Preprocessing_Features.ipynb  # Cleaning & feature engineering
├── 03_Modeling.ipynb                # LightGBM training, tuning & SHAP
├── app.py                           # Streamlit app
├── requirements.txt                 # Deployment dependencies
└── README.md
```

---

## ⚙️ Pipeline Overview

| Phase | Description |
|---|---|
| **EDA** | Price distributions, brand analysis, correlation heatmap |
| **Preprocessing** | Null imputation, outlier removal, engine text parsing |
| **Feature Engineering** | 22 features — car age, log mileage, brand tier, HP ratios |
| **Modeling** | LightGBM + Optuna tuning (50 trials) + 5-fold CV |
| **Explainability** | SHAP bar, beeswarm & dependence plots |
| **App** | Real-time prediction with depreciation curve & factor breakdown |

---

## 📊 Dataset

**Kaggle Playground Series S4E9 — Used Car Price Regression**
- 188,000 records · 12 raw features · Target: `price` (USD)
- Source: [kaggle.com/competitions/playground-series-s4e9](https://www.kaggle.com/competitions/playground-series-s4e9)

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

`Python` · `LightGBM` · `Optuna` · `SHAP` · `Streamlit` · `Plotly` · `Pandas` · `scikit-learn`
