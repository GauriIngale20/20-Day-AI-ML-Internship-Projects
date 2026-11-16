# Preprocessing Model Comparison Report

## Project: Titanic Survival Prediction

### Objective
To compare baseline preprocessing with a full preprocessing pipeline using Logistic Regression.

### Dataset
- Dataset: Titanic
- Total Records: 891
- Selected Features: 7
- Training Records: 712
- Testing Records: 179

### Preprocessing Techniques

**Baseline Pipeline**
- Median Imputation
- Most Frequent Imputation
- One-Hot Encoding

**Full Pipeline**
- Median Imputation
- Most Frequent Imputation
- One-Hot Encoding
- Standard Scaling

### Model Comparison

| Model | Accuracy |
|---|---:|
| Baseline Pipeline | 80.45% |
| Full Preprocessing Pipeline | 80.45% |

### Observation
Both models achieved the same accuracy of 80.45%. Scaling did not improve accuracy in this experiment.

### Conclusion
The full preprocessing pipeline provides a reusable workflow for handling missing values, encoding categorical features and scaling numerical features.