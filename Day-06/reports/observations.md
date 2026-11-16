# Day 06 – Project Observations

## Dataset Observations
- The Titanic dataset contains 891 records and 12 columns.
- Missing values were found in Age, Cabin and Embarked.
- Cabin contained the highest number of missing values.

## Preprocessing Observations
- Median imputation handled missing numerical values.
- Most frequent imputation handled missing categorical values.
- One-Hot Encoding converted categorical features into numerical values.
- StandardScaler scaled numerical features.
- ColumnTransformer combined different preprocessing operations.

## Model Observations
- Logistic Regression was used for classification.
- Baseline accuracy: 80.45%.
- Full pipeline accuracy: 80.45%.
- Both models achieved the same accuracy.

## Final Observation
The reusable preprocessing pipeline successfully handled missing values, categorical encoding and numerical scaling without fitting preprocessing steps on the test data.