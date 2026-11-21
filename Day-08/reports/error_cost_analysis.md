# False Positive and False Negative Cost Analysis

## 1. Introduction

In machine learning classification, predictions can be correct or incorrect. Two important types of classification errors are False Positive (FP) and False Negative (FN).

Understanding these errors helps us select suitable evaluation metrics and classification thresholds.

## 2. Confusion Matrix

| Actual / Predicted | Predicted Not Survived (0) | Predicted Survived (1) |
|---|---:|---:|
| Actually Not Survived (0) | True Negative (TN) | False Positive (FP) |
| Actually Survived (1) | False Negative (FN) | True Positive (TP) |

## 3. False Positive (FP)

A False Positive occurs when the model predicts a positive class, but the actual class is negative.

In our Titanic project:
- The model predicts that a passenger survived.
- In reality, the passenger did not survive.

**Possible impact:** This produces an incorrect survivor prediction and can lead to an overestimation of the number of survivors.

## 4. False Negative (FN)

A False Negative occurs when the model predicts a negative class, but the actual class is positive.

In our Titanic project:
- The model predicts that a passenger did not survive.
- In reality, the passenger survived.

**Possible impact:** This incorrectly excludes an actual survivor from the predicted survivor group and can lead to an underestimation of the number of survivors.

## 5. Real-World Example: Medical Screening

Consider a medical screening system that predicts whether a person may have a disease.

- **False Positive:** The system predicts that a healthy person may have the disease. This may result in unnecessary medical tests, expenses, and anxiety.
- **False Negative:** The system predicts that a person does not have the disease even though the person actually has it. This may delay diagnosis and treatment.

In this example, false negatives can be more costly because a person with a disease may not receive timely medical attention.

## 6. Effect of Classification Threshold

The classification threshold determines when a model predicts the positive class.

- A lower threshold generally increases recall and may increase false positives.
- A higher threshold generally increases precision and may increase false negatives.

Therefore, threshold selection depends on the cost of different types of errors.

## 7. Threshold Selection in Our Project

Based on our threshold analysis:

| Model | Selected Threshold | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.4 | 0.7297 | 0.7826 | 0.7552 |
| Balanced Logistic Regression | 0.5 | 0.7297 | 0.7826 | 0.7552 |
| Decision Tree | 0.3 | 0.6463 | 0.7681 | 0.7020 |

The selected thresholds provide the highest F1-score among the tested thresholds for each model.

## 8. Conclusion

False Positives and False Negatives represent different types of classification errors. Their importance depends on the application.

Precision helps control false positives, while recall helps identify more actual positive cases. The F1-score balances precision and recall.

Selecting an appropriate threshold helps achieve a suitable balance between these errors based on the requirements of the problem.