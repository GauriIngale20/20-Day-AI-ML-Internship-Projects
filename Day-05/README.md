# Day 05: First Baseline Machine Learning Model

## Overview

This project focuses on building and evaluating
baseline classification models using Python and
Scikit-learn.

The Breast Cancer Wisconsin Dataset is used to
predict whether a tumor is benign or malignant.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

Day-05/
│
├── data/
│   └── breast_cancer.csv
│
├── notebooks/
│   └── baseline_model.ipynb
│
├── src/
│
├── reports/
│   ├── baseline_metrics.csv
│   └── baseline_model_report.md
│
├── .gitignore
├── README.md
└── requirements.txt

## Models Implemented

1. DummyClassifier
2. Logistic Regression

## Dataset Splitting

- Training: 70%
- Validation: 15%
- Testing: 15%
- Random State: 42
- Stratified Sampling

## Results

| Model | Accuracy |
|-------|----------|
| Dummy Classifier | 62.35% |
| Logistic Regression (Validation) | 96.47% |
| Logistic Regression (Test) | 97.67% |

## Key Learning Outcomes

- Understanding features and target variables
- Training baseline classification models
- Using DummyClassifier for benchmarking
- Implementing Scikit-learn Pipelines
- Evaluating models using classification metrics
- Maintaining reproducibility

## How to Run

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Open the notebook:

   notebooks/baseline_model.ipynb

5. Run all notebook cells.

## Conclusion

The Logistic Regression baseline achieved 97.67%
test accuracy and outperformed the DummyClassifier.

This project establishes a foundation for future
machine learning experiments.

