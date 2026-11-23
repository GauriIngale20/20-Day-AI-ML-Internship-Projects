# Titanic Survival Prediction - Model Review

## 1. Project Objective

The objective of this project is to predict whether a Titanic passenger survived based on passenger information.

## 2. Dataset

- Dataset: Titanic
- Total Records: 891
- Original Features: 14
- Target Column: survived
- Task Type: Binary Classification

## 3. Data Quality

- Missing values were identified.
- Age and Embarked contain missing values.
- Duplicate records were checked.
- Target leakage columns were excluded from model features.

## 4. Data Preprocessing

- Numerical missing values were handled using median imputation.
- Categorical missing values were handled using most frequent imputation.
- Numerical features were standardized.
- Categorical features were one-hot encoded.
- Preprocessing was included in a Scikit-learn Pipeline.

## 5. Model Development

Three classification algorithms were compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The best model was selected using validation F1-score.

## 6. Model Evaluation

The selected model was evaluated on the separate test dataset.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Refer to `metrics.json` for actual test results.

## 7. Error Analysis

False Positive and False Negative predictions were analyzed.

Errors were also examined by gender and passenger class.

Refer to `error_analysis.csv` for detailed results.

## 8. Limitations

- The dataset contains only 891 passenger records.
- Some important passenger information is missing.
- Historical Titanic data may not represent modern survival situations.
- Model performance may change on unseen datasets.
- Correlation does not necessarily imply causation.

## 9. Possible Improvements

- Hyperparameter tuning using cross-validation.
- Feature engineering using family size and passenger titles.
- Comparing additional classification algorithms.
- Evaluating different classification thresholds.
- Testing on additional relevant datasets.

## 10. Conclusion

This project demonstrates a reproducible tabular machine learning workflow, covering data quality, EDA, preprocessing, data splitting, model comparison, evaluation, and error analysis.
