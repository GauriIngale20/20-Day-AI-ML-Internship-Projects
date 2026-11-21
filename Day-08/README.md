# Day 08: Evaluation Metrics and Imbalanced Data

## Project Overview

This project focuses on evaluating machine learning classification models using different performance metrics and understanding how classification thresholds affect model predictions.

The Titanic dataset is used to compare Logistic Regression, Balanced Logistic Regression, and Decision Tree models.

## Objectives

- Understand Precision, Recall, F1-Score, and Accuracy.
- Analyze confusion matrices.
- Compare ROC-AUC and Precision-Recall performance.
- Understand class imbalance and class weighting.
- Perform threshold analysis.
- Study the impact of False Positives and False Negatives.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Pytest
- Jupyter Notebook

## Project Structure

```text
Day-08/
│
├── data/
│   └── raw/
│       └── titanic.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── balanced_logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── test_data.pkl
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── model_metrics.csv
│   ├── metrics_report.md
│   ├── threshold_results.csv
│   ├── threshold_analysis.md
│   └── error_cost_analysis.md
│
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── threshold_analysis.py
│   └── visualization.py
│
├── tests/
│   └── test_evaluation.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The Titanic dataset contains 891 passenger records and 12 columns.

Target variable: `Survived`

- 0 = Did not survive
- 1 = Survived

Class distribution:

- Non-survivors: 549 (61.62%)
- Survivors: 342 (38.38%)

The dataset has a moderate class imbalance.

## Models Implemented

1. Logistic Regression
2. Balanced Logistic Regression using `class_weight="balanced"`
3. Decision Tree Classifier

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures how many positive predictions are correct |
| Recall | Measures how many actual positive cases are identified |
| F1-Score | Balances Precision and Recall |
| ROC-AUC | Measures the model's ability to distinguish between classes |
| Average Precision | Summarizes the precision-recall curve |

Confusion matrices are also used to analyze True Positives, True Negatives, False Positives, and False Negatives.

## Threshold Analysis

The models are evaluated at thresholds 0.3, 0.4, 0.5, 0.6, and 0.7.

Best F1-score among the tested thresholds:

| Model | Selected Threshold | F1-Score |
|---|---:|---:|
| Logistic Regression | 0.4 | 0.7552 |
| Balanced Logistic Regression | 0.5 | 0.7552 |
| Decision Tree | 0.3 | 0.7020 |

The selected threshold is based on the highest F1-score among the tested values. A different threshold may be appropriate when precision or recall is more important.

## Visualizations

The project generates:

- Confusion Matrix Comparison
- ROC Curve
- Precision-Recall Curve
- Model Performance Comparison
- Threshold Analysis Plots

All generated figures are stored in `reports/figures/`.

## False Positive and False Negative Analysis

In the Titanic project:

- False Positive: The model predicts survival, but the passenger did not survive.
- False Negative: The model predicts non-survival, but the passenger survived.

A medical screening example is also included to explain why the cost of false negatives can be higher in some real-world applications.

## How to Run the Project

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the project modules

```powershell
python -m src.data_loader
python -m src.train
python -m src.evaluate
python -m src.visualization
python -m src.threshold_analysis
```

### 4. Run tests

```powershell
python -m pytest tests/ -v
```

## Test Results

All 5 automated tests passed successfully.

## Learning Outcomes

- Learned how to evaluate classification models using multiple metrics.
- Understood the effect of class imbalance.
- Compared regular and balanced Logistic Regression.
- Analyzed confusion matrices and classification errors.
- Understood how threshold selection affects Precision and Recall.
- Learned to select evaluation metrics based on the problem requirements.

## Conclusion

This project demonstrates that model evaluation should not depend only on accuracy. Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall analysis provide a more complete understanding of classification performance.

Threshold selection should depend on the practical cost of different prediction errors.