# Project 1 — Customer Segmentation \& Retention Analysis

## Overview

End-to-end data science project analyzing customer behavior for a telecom company.
Segments customers using RFM analysis and K-Means clustering, predicts churn using
Logistic Regression, and estimates Customer Lifetime Value (CLV).

## Business Questions Answered

* Who should get retention offers?
* Who are the high-value loyal customers?
* Who is likely to churn regardless of intervention?

## Tools Used

* Python 3.11
* pandas, numpy, scikit-learn
* matplotlib, seaborn
* Jupyter Notebook

## Dataset

Telco Customer Churn Dataset — https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Key Results

* Identified 4 customer segments via K-Means clustering
* Churn prediction ROC-AUC: 0.84 (Logistic Regression)
* Cluster 3 (At Risk): 48% churn rate — primary retention target
* Cluster 1 (Champions): 15% churn rate — highest CLV segment

## Project Structure

Customer-Segmentation-\&-Retention-Analysis/
├── data/
│   └── WA\_Fn-UseC\_-Telco-Customer-Churn.csv
├── project1\_customer\_segmentation.ipynb
└── README.md

