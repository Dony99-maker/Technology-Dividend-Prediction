# Predicting Dividend Changes in Technology Companies

## Project Overview
An MSc Data Analytics dissertation combining finance and machine learning to predict whether technology companies will increase their dividend payments using historical stock prices and trading volumes.

## Dataset
- Data source: Yahoo Finance
- Companies analysed: 39
- Historical data: 2010–2025

## Machine Learning Models
- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosting

## Methodology
- Financial data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- SMOTE class balancing
- Chronological cross-validation
- Hyperparameter tuning using RandomizedSearchCV
- Explainable AI using SHAP

## Model Performance

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 73.07% | 0.7930 |
| Decision Tree | 61.54% | 0.6529 |
| Random Forest | 57.69% | 0.6704 |
| Gradient Boosting | 55.13% | 0.5989 |

## Key Findings
- Logistic Regression achieved the best predictive performance.
- Price Range Ratio was the most influential feature.
- Simpler models outperformed more complex models on this dataset.

## Project Files
- data-collection.ipynb — Financial data collection
- dividend_dataset.csv — Project dataset
- dividend_test_dataset.csv — Testing dataset
- train_with smote.ipynb — Model training with SMOTE
- train_without_smote.ipynb — Model training without SMOTE
- dividend_prediction_app.py — Desktop prediction application
- Dissertation report and project presentation

## Technical Skills
Python, Pandas, NumPy, Scikit-learn, Matplotlib, SHAP, Financial Analytics, Predictive Modelling and Machine Learning.

## Business Applications
- Financial forecasting
- Dividend policy analysis
- Investment research
- Financial decision support

## Limitations
The study uses a limited sample of technology companies and three engineered predictors. Further validation is required before practical investment use.

## Author
Dony Francis
MSc Data Analytics | MBA Finance & Operations

This project is for academic and research purposes and does not constitute investment advice.
