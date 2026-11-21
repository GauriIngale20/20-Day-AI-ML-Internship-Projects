# Day 08 - Threshold Analysis Report

## Threshold Performance Comparison

| Model                        |   Threshold |   Accuracy |   Precision |   Recall |   F1-Score |   TN |   FP |   FN |   TP |
|:-----------------------------|------------:|-----------:|------------:|---------:|-----------:|-----:|-----:|-----:|-----:|
| logistic_regression          |         0.3 |     0.7598 |      0.6548 |   0.7971 |     0.719  |   81 |   29 |   14 |   55 |
| logistic_regression          |         0.4 |     0.8045 |      0.7297 |   0.7826 |     0.7552 |   90 |   20 |   15 |   54 |
| logistic_regression          |         0.5 |     0.8045 |      0.7931 |   0.6667 |     0.7244 |   98 |   12 |   23 |   46 |
| logistic_regression          |         0.6 |     0.7989 |      0.8511 |   0.5797 |     0.6897 |  103 |    7 |   29 |   40 |
| logistic_regression          |         0.7 |     0.7877 |      0.8974 |   0.5072 |     0.6481 |  106 |    4 |   34 |   35 |
| balanced_logistic_regression |         0.3 |     0.7207 |      0.5979 |   0.8406 |     0.6988 |   71 |   39 |   11 |   58 |
| balanced_logistic_regression |         0.4 |     0.7598 |      0.6548 |   0.7971 |     0.719  |   81 |   29 |   14 |   55 |
| balanced_logistic_regression |         0.5 |     0.8045 |      0.7297 |   0.7826 |     0.7552 |   90 |   20 |   15 |   54 |
| balanced_logistic_regression |         0.6 |     0.7933 |      0.7581 |   0.6812 |     0.7176 |   95 |   15 |   22 |   47 |
| balanced_logistic_regression |         0.7 |     0.8045 |      0.8542 |   0.5942 |     0.7009 |  103 |    7 |   28 |   41 |
| decision_tree                |         0.3 |     0.7486 |      0.6463 |   0.7681 |     0.702  |   81 |   29 |   16 |   53 |
| decision_tree                |         0.4 |     0.7765 |      0.7377 |   0.6522 |     0.6923 |   94 |   16 |   24 |   45 |
| decision_tree                |         0.5 |     0.7654 |      0.7547 |   0.5797 |     0.6557 |   97 |   13 |   29 |   40 |
| decision_tree                |         0.6 |     0.7654 |      0.7547 |   0.5797 |     0.6557 |   97 |   13 |   29 |   40 |
| decision_tree                |         0.7 |     0.7654 |      0.8462 |   0.4783 |     0.6111 |  104 |    6 |   36 |   33 |

## Threshold Analysis Observations

### Logistic Regression

- Best F1-score threshold: 0.4
- Precision: 0.7297
- Recall: 0.7826
- F1-score: 0.7552

### Balanced Logistic Regression

- Best F1-score threshold: 0.5
- Precision: 0.7297
- Recall: 0.7826
- F1-score: 0.7552

### Decision Tree

- Best F1-score threshold: 0.3
- Precision: 0.6463
- Recall: 0.7681
- F1-score: 0.7020


## Threshold Selection Explanation

A lower classification threshold generally increases
the number of positive predictions and can improve recall.

A higher threshold generally reduces positive predictions
and can improve precision.

The best threshold depends on the problem and the cost
of false positive and false negative predictions.

The threshold with the highest F1-score provides one
possible balance between precision and recall. It is
not automatically the best threshold for every use case.

## Conclusion

Threshold selection changes the classification results.
Therefore, model evaluation should consider multiple
thresholds instead of relying only on the default 0.5.
