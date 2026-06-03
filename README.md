# 🚚 Food Delivery ETA Prediction and MLOps Platform

## Overview

An end-to-end Machine Learning and MLOps project that predicts food delivery times using historical delivery data. The project follows industry-standard practices for data preprocessing, model development, experiment tracking, versioning, deployment, and CI/CD automation.

**Key Result:** Achieved **0.823 Test R²** and **3.17 MAE** using a stacked ensemble model optimized with Optuna.

---

## Problem Statement

Accurate delivery time estimation is crucial for customer satisfaction and operational efficiency in food delivery platforms. This project aims to build a scalable and production-ready system capable of predicting delivery times with high accuracy.

---

## Dataset

* **45K+ delivery records**
* Numerical and categorical features related to orders, delivery partners, distance, traffic, weather, and restaurant information.

---

## Workflow

```text
Data Collection
      │
      ▼
Data Cleaning & Preprocessing
      │
      ▼
EDA & Feature Engineering
      │
      ▼
Model Training & Tuning
      │
      ▼
Stacking Regressor
      │
      ▼
MLflow + DVC + DagsHub
      │
      ▼
Flask API
      │
      ▼
Docker
      │
      ▼
GitHub Actions CI/CD
```

---

## Exploratory Data Analysis

* Univariate and bivariate analysis
* Missing value analysis
* Outlier detection and treatment
* Feature-target relationship analysis
* Correlation analysis and visualization

---

## Model Development

### Baseline Models

* Linear Regression
* Random Forest Regressor

### Final Model

**Stacking Regressor**

* Base Models:

  * LightGBM
  * Random Forest
* Meta Model:

  * Linear Regression

### Optimization

* Optuna Hyperparameter Tuning
* Cross Validation

---

## Model Performance

| Metric      | Score |
| ----------- | ----- |
| Train R²    | 0.849 |
| Test R²     | 0.823 |
| Train MAE   | 2.91  |
| Test MAE    | 3.17  |
| Mean CV MAE | 3.17  |

---

## MLOps Features

### Experiment Tracking

* MLflow
* Model Registry
* Artifact Logging

### Data Versioning

* DVC
* DagsHub Integration

### CI/CD

* GitHub Actions

### Containerization

* Docker

---

## Deployment

The trained model is deployed as a **Flask REST API**, enabling real-time delivery time predictions through API endpoints.

```text
User Input → Flask API → Preprocessing Pipeline → Model → ETA Prediction
```

---

## Project Structure

```text
├── notebooks/
├── src/
│   ├── components/
│   ├── pipelines/
│   ├── utils/
│   ├── logger/
│   └── exception/
├── artifacts/
├── app.py
├── Dockerfile
├── dvc.yaml
├── requirements.txt
└── .github/workflows/
```

---

## Tech Stack

**Python • Pandas • NumPy • Scikit-Learn • LightGBM • Optuna • MLflow • DVC • DagsHub • Flask • Docker • GitHub Actions • Matplotlib • Seaborn**

---

## Key Highlights

✅ End-to-end ML pipeline on **45K+ records**
✅ Stacking Ensemble with **LightGBM + Random Forest + Linear Regression**
✅ **0.823 Test R²** and **3.17 MAE**
✅ Experiment tracking using MLflow
✅ Data versioning with DVC and DagsHub
✅ Dockerized Flask API deployment
✅ Automated CI/CD with GitHub Actions

---

**Ayush Devan**

Machine Learning • Data Science • MLOps • Data Engineering
