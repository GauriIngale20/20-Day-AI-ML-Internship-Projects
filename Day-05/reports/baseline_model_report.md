# Day 05: First Baseline Machine Learning Model

## 1. Objective

The objective of this project is to train a baseline
classification model using Scikit-learn and evaluate
its performance before applying any hyperparameter tuning.

## 2. Dataset Information

- Dataset: Breast Cancer Wisconsin Dataset
- Total Records: 569
- Total Features: 30
- Target Column: diagnosis
- Problem Type: Binary Classification

## 3. Data Splitting Strategy

The dataset was divided using stratified sampling.

| Dataset | Percentage | Records |
|---------|------------|---------|
| Training | 70% | 398 |
| Validation | 15% | 85 |
| Testing | 15% | 86 |

Random State: 42

## 4. Models Used

### A. Dummy Classifier

DummyClassifier was used as a baseline benchmark.
It predicts the most frequent class without learning
patterns from the input features.

### B. Logistic Regression

Logistic Regression was used as the actual baseline
classification model.

StandardScaler was included in a Scikit-learn Pipeline
to scale features without data leakage.

## 5. Validation Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Dummy Classifier | 0.6235 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.9647 | 1.0000 | 0.9062 | 0.9508 |

## 6. Final Test Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.9767 |
| Precision | 0.9688 |
| Recall | 0.9688 |
| F1 Score | 0.9688 |

## 7. Observations

1. DummyClassifier achieved 62.35% accuracy.
2. Logistic Regression performed significantly better
   than the DummyClassifier.
3. Logistic Regression achieved 96.47% validation accuracy.
4. The final test accuracy was 97.67%.
5. The pipeline helped maintain proper preprocessing
   without fitting the scaler on validation or test data.

## 8. Conclusion

The baseline Logistic Regression model achieved
97.67% accuracy on the test dataset.

The results demonstrate that Logistic Regression
learned useful patterns from the dataset and performed
better than the DummyClassifier benchmark.

This baseline will be used for comparison in future
machine learning tasks.

## 9. Limitations

- No hyperparameter tuning was performed.
- Results are based on one train-validation-test split.
- This educational model is not intended for clinical
  diagnosis or medical decision-making.