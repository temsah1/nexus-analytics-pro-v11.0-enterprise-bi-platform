# NEXUS-Analytics-Pro-v11.0-Enterprise-BI-Platform
Enterprise-grade Business Intelligence (BI) &amp; AI platform featuring Time-Series Forecasting, ML Profit Optimization, RFM Segmentation, and Market Basket Analysis.
<h1 align="center">🚀 NEXUS Analytics Pro v11.0 — Enterprise BI Platform</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E.svg" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Forecasting-Prophet-00BFFF.svg" alt="Prophet">
  <img src="https://img.shields.io/badge/Database-SQLite-003B57.svg" alt="SQLite">
</p>

<p align="center">
  <strong>An ultra-modern, AI-powered Business Intelligence platform designed to transform raw data into actionable enterprise insights.</strong>
</p>

---

## 📌 Project Overview
**NEXUS Analytics Pro** is a comprehensive Data Analytics and Machine Learning SaaS application. It allows businesses to securely upload massive datasets (up to 1GB) and instantly run advanced algorithms without writing a single line of code. From **Time-Series Forecasting** to **Market Basket Analysis** and **Customer Segmentation**, NEXUS acts as an automated data scientist for your organization.

## ✨ Core Features & Modules

* **🔐 Secure Mega Admin Dashboard:**
  * Complete User Management (Create, Delete, Promote, Password Resets).
  * System Activity Logging and Login tracking stored securely in `SQLite`.
  * Configurable system settings directly from the UI.
* **🔮 Demand Forecasting (Prophet & Statsmodels):**
  * Accurately predicts future trends (30-180 days) using Facebook's Prophet and Exponential Smoothing algorithms.
* **🤖 AI Profit Optimizer (Ensemble Learning):**
  * Utilizes a highly robust `VotingRegressor` combining Random Forest, Gradient Boosting, XGBoost, and Ridge to predict and optimize profitability.
  * Features **Explainable AI (XAI)** with Feature and Permutation Importance charts.
* **👥 Customer Intelligence (RFM & Clustering):**
  * Generates automated **RFM (Recency, Frequency, Monetary)** scoring and segmentation (e.g., 'Champions', 'At Risk').
  * Advanced unsupervised learning: K-Means, DBSCAN, and Hierarchical clustering with PCA scatter projections and Silhouette scoring.
* **🛒 Market Basket Analysis:**
  * Uses the `Apriori` algorithm to extract association rules (Lift, Support, Confidence), revealing which products are frequently bought together.
* **🔍 Advanced Anomaly Detection:**
  * Implements `Isolation Forest` to automatically flag outliers, financial anomalies, or potential fraud in the dataset.
* **📄 Automated Executive Reports:**
  * 1-Click Markdown generation summarizing KPIs and actionable AI recommendations.

## 🛠️ Technology Stack
* **Frontend:** Streamlit with Custom CSS (Ultra Modern Glassmorphism UI)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (RandomForest, IsolationForest, DBSCAN, VotingRegressor)
* **Time Series:** Prophet, Statsmodels
* **Association Rules:** mlxtend (Apriori)
* **Visualizations:** Plotly Express & Plotly Graph Objects
* **Database & Security:** SQLite3, Hashlib

## 🚀 Installation & Local Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/nexus-analytics-pro.git](https://github.com/yourusername/nexus-analytics-pro.git)
cd nexus-analytics-pro
