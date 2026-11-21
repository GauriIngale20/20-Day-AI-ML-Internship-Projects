# Day 07: Logistic Regression vs Decision Tree

## 1. Project Overview

The objective of this project is to compare two supervised machine learning classification algorithms: Logistic Regression and Decision Tree.

The Titanic dataset is used to predict whether a passenger survived or not.

## 2. Dataset Information

- Dataset: Titanic Dataset
- Total Records: 891
- Total Columns: 12
- Target Column: Survived
- Training Records: 712
- Testing Records: 179
- Selected Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

## 3. Algorithms Used

### Logistic Regression
Logistic Regression is a supervised machine learning algorithm used for classification. It predicts the probability of a passenger surviving based on the input features.

### Decision Tree
Decision Tree is a supervised machine learning algorithm that makes predictions using a tree-like structure of decision rules.

## 4. Evaluation Metrics

- Accuracy: Measures the percentage of correct predictions.
- Precision: Measures how many predicted survivors actually survived.
- Recall: Measures how many actual survivors were correctly identified.
- F1 Score: Provides a balance between precision and recall.
- Cross-Validation: Evaluates model performance across multiple data splits.

## 5. Model Comparison

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Training Accuracy | 80.76% | 86.52% |
| Testing Accuracy | 80.45% | 76.54% |
| Precision | 79.31% | 75.47% |
| Recall | 66.67% | 57.97% |
| F1 Score | 72.44% | 65.57% |
| Mean CV Accuracy | 79.63% | 80.34% |

## 6. Overfitting Analysis

The Decision Tree achieved 86.52% training accuracy but only 76.54% testing accuracy. This difference suggests possible overfitting.

Logistic Regression achieved 80.76% training accuracy and 80.45% testing accuracy. The smaller difference indicates more consistent performance on training and testing data.

## 7. Cross-Validation

Stratified 5-Fold Cross-Validation was applied to both models.

- Logistic Regression Mean CV Accuracy: 79.63%
- Decision Tree Mean CV Accuracy: 80.34%

Decision Tree achieved slightly higher mean cross-validation accuracy.

## 8. Conclusion

Logistic Regression performed better on the separate testing dataset, achieving 80.45% accuracy and a 72.44% F1 score.

Decision Tree achieved slightly higher cross-validation accuracy but showed a larger gap between training and testing accuracy.

Therefore, Logistic Regression is the better-performing model on this particular test split.
