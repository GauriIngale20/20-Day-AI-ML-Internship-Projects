# Titanic Survival Prediction – Tabular Machine Learning Project

## 1. Project Overview

This project implements an end-to-end tabular machine learning workflow to predict whether a passenger survived the Titanic disaster.

It demonstrates data quality analysis, exploratory data analysis, preprocessing, model comparison, evaluation, and error analysis using Python and Scikit-learn.

## 2. Project Objectives

- Analyze and understand the Titanic dataset.
- Identify missing values and duplicate records.
- Perform exploratory data analysis.
- Build a leakage-aware preprocessing pipeline.
- Compare multiple classification algorithms.
- Evaluate the selected model on an unseen test dataset.
- Analyze incorrect predictions.
- Create a reproducible ML workflow.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Pytest
- Jupyter Notebook

## 4. Dataset Information

- Dataset: Titanic
- Total Records: 891
- Total Columns: 15
- Target Column: `survived`
- Problem Type: Binary Classification

Target values:
- 0 = Did not survive
- 1 = Survived

## 5. Project Structure

```text
Day-10/
│
├── data/
│   └── raw/
│       └── titanic.csv
│
├── models/
│   └── best_model.joblib
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── data_quality_report.md
│   ├── eda_observations.md
│   ├── model_comparison.csv
│   ├── metrics.json
│   ├── error_analysis.csv
│   └── model_review.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   └── visualization.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_pipeline.py
│
├── config.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore