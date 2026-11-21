# Day 08 - Model Evaluation Report

## Model Performance Comparison

| Model                        |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   PR-AUC |   TN |   FP |   FN |   TP |
|:-----------------------------|-----------:|------------:|---------:|-----------:|----------:|---------:|-----:|-----:|-----:|-----:|
| logistic_regression          |     0.8045 |      0.7931 |   0.6667 |     0.7244 |    0.8437 |   0.7865 |   98 |   12 |   23 |   46 |
| balanced_logistic_regression |     0.8045 |      0.7297 |   0.7826 |     0.7552 |    0.8464 |   0.7882 |   90 |   20 |   15 |   54 |
| decision_tree                |     0.7654 |      0.7547 |   0.5797 |     0.6557 |    0.7971 |   0.7636 |   97 |   13 |   29 |   40 |

## Evaluation Metrics

- Accuracy: Overall correct predictions.
- Precision: Correct positive predictions among predicted positives.
- Recall: Correct positive predictions among actual positives.
- F1-Score: Harmonic mean of precision and recall.
- ROC-AUC: Ability to distinguish between classes.
- PR-AUC: Area under the Precision-Recall curve.

## Confusion Matrix

- TN: True Negative
- FP: False Positive
- FN: False Negative
- TP: True Positive

## Conclusion

The three classification models were evaluated on the same
test dataset. Model performance should be compared using
multiple metrics rather than accuracy alone.
