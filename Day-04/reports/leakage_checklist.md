# Data Leakage and Duplicate Contamination Checklist

## 1. Objective
The objective of this checklist is to identify and prevent data leakage during machine learning.

## 2. Data Leakage Checklist

| Sr. No. | Checklist | Status |
|---|---|---|
| 1 | Training and testing data kept separate | Completed |
| 2 | Duplicate records checked | Completed |
| 3 | Duplicate records across train and test checked | Completed |
| 4 | Stratified splitting applied | Completed |
| 5 | Reproducibility checked using random_state=42 | Completed |
| 6 | Preprocessing should be fitted only on training data | To be followed |
| 7 | Testing data should not be used during model training | To be followed |
| 8 | Target-related information should not be used as an input feature | To be followed |

## 3. Duplicate Audit Results

- Total duplicate records: 0
- Training duplicates: 0
- Validation duplicates: 0
- Testing duplicates: 0
- Duplicate records across training and testing: 0

## 4. Common Causes of Data Leakage

1. Using testing data during model training.
2. Applying scaling or normalization before splitting the dataset.
3. Using future information to predict past events.
4. Keeping duplicate records in training and testing datasets.
5. Including target-related information in input features.

## 5. Prevention Methods

1. Split the dataset before preprocessing.
2. Fit scalers and imputers only on training data.
3. Keep testing data separate until final evaluation.
4. Remove duplicate records before splitting when appropriate.
5. Use a proper train-validation-test strategy.

## 6. Conclusion

The dataset audit found no duplicate records or overlap between training and testing datasets. Stratified splitting maintained a similar class distribution. Proper preprocessing and data separation must be followed to prevent data leakage.