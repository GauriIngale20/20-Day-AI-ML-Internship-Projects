# Train, Validation and Test Split Strategy

## 1. Objective
The objective is to divide the dataset into training, validation, and testing sets for reliable machine learning evaluation.

## 2. Dataset Details
- Dataset: Breast Cancer Wisconsin Dataset
- Total Records: 569
- Total Features: 30
- Target Column: diagnosis

## 3. Splitting Strategy

The dataset was divided into three parts:

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 70% | 398 |
| Validation | 15% | 85 |
| Testing | 15% | 86 |
| Total | 100% | 569 |

## 4. Splitting Technique
- Used train_test_split() from Scikit-learn.
- Applied stratified splitting to maintain class distribution.
- Used random_state=42 for reproducibility.

## 5. Class Distribution
- Training: Benign 62.8%, Malignant 37.2%
- Validation: Benign 62.4%, Malignant 37.6%
- Testing: Benign 62.8%, Malignant 37.2%

## 6. Conclusion
The dataset was successfully divided into training, validation, and testing sets. Stratified splitting maintained a similar class distribution across all three sets. Reproducibility was verified using random_state=42.