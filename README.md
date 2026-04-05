# Digital-Health-Project
# 🫀 Digital Health Project — CVD Detection Pipeline

Welcome to the **HSB Digital Health Project**!  

This repository contains practical tools and scripts for building machine learning models to predict **Raised Blood Pressure** using real-world health survey data.

### What's Inside

- **`data extraction.py`**  
  A clean and reusable script that handles data cleaning, clinical exclusions, and preprocessing — even when column names differ across datasets.

- **`CVD_detection_full_pipeline.ipynb`**  
  The main Jupyter notebook with a complete end-to-end pipeline:
  - Exploratory Data Analysis
  - Smart preprocessing
  - Handling class imbalance with SMOTE-ENN and SMOTE-Tomek
  - Training several models (Logistic Regression, SVM, KNN, Random Forest, XGBoost)
  - Threshold tuning to get the best F1-score for the positive class
  - Feature selection using SelectKBest and RFE
  - Comparison with a no-resampling baseline

### Quick Start

```bash
# Create and activate environment
conda create --name ml-env python=3.11
conda activate ml-env

# Install required packages
conda install numpy pandas scikit-learn matplotlib seaborn jupyter xgboost imbalanced-learn
