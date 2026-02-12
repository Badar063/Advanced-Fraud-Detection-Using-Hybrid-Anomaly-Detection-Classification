Advanced Fraud Detection Using Hybrid Anomaly Detection & Classification
Project Overview

This project presents a state-of-the-art fraud detection pipeline leveraging both unsupervised and supervised learning on synthetic financial transaction data. The approach combines anomaly detection methods (Autoencoder, Isolation Forest) with supervised classification models (XGBoost, Random Forest), incorporating advanced techniques such as SMOTE for imbalance handling, hyperparameter tuning, SHAP-based interpretability, and threshold optimization.

The dataset includes 50 synthetic credit card transactions, simulating real-world fraud patterns (≈20% fraud rate).

Dataset (Synthetic)

transactions.csv – 50 credit card transactions labeled for fraud

Key features: transaction amount, merchant category, card-present flag, distance from home, time since last transaction, hour/day, etc.

Engineered features: log-transformed amount, amount per distance, merchant fraud risk score, customer-level aggregates, autoencoder reconstruction error, isolation forest anomaly score

Techniques & Purpose
Technique	Purpose
Deep Autoencoder	Unsupervised anomaly detection via reconstruction error
Isolation Forest	Unsupervised outlier detection
SMOTE	Handle class imbalance for supervised learning
XGBoost / Random Forest	Supervised fraud classification
GridSearchCV	Hyperparameter tuning for optimal model performance
Precision-Recall & ROC	Evaluation metrics for imbalanced datasets
SHAP	Model interpretability and feature importance
Threshold Tuning	Optimize F1-score via probability cutoff adjustment
Installation

Install required dependencies:

pip install -r requirements.txt
