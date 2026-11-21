# Day 07: Logistic Regression vs Decision Tree

## Project Description

This project compares two supervised machine learning classification algorithms using the Titanic dataset.

The main objective is to understand model performance, overfitting, regularization through tree-depth control, and 5-Fold Cross-Validation.

## Objectives

- Implement Logistic Regression.
- Implement Decision Tree Classifier.
- Understand overfitting and model generalization.
- Apply 5-Fold Cross-Validation.
- Compare classification evaluation metrics.
- Visualize model performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Structure

```text
Day-07/
│
├── data/
│   └── raw/
│       └── titanic.csv
│
├── notebooks/
│   └── classification_models.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
│
├── models/
│   ├── logistic_regression.pkl
│   └── decision_tree.pkl
│
├── reports/
│   ├── model_comparison.md
│   └── figures/
│       ├── accuracy_comparison.png
│       ├── metrics_comparison.png
│       └── cross_validation_comparison.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Machine Learning Models

1. Logistic Regression
2. Decision Tree Classifier

## Dataset

The Titanic dataset contains 891 passenger records.

Target variable: `Survived`

- 0 = Did not survive
- 1 = Survived

## Model Performance

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Testing Accuracy | 80.45% | 76.54% |
| Precision | 79.31% | 75.47% |
| Recall | 66.67% | 57.97% |
| F1 Score | 72.44% | 65.57% |
| Mean CV Accuracy | 79.63% | 80.34% |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python -m src.train
```

### 3. Evaluate the models

```bash
python -m src.evaluate
```

### 4. Generate visualizations

```bash
python -m src.visualization
```

## Conclusion

Logistic Regression achieved better testing accuracy and F1 score than Decision Tree on the selected test split.

The project demonstrates classification model comparison, cross-validation, and overfitting analysis.

