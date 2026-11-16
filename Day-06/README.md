# Day 06: Data Preprocessing and Feature Engineering

## Project Overview
This project focuses on building a reusable preprocessing pipeline using Python and Scikit-learn.

The Titanic dataset is used to perform data cleaning, imputation, categorical encoding and numerical scaling.

## Objectives
- Handle missing values.
- Apply numerical and categorical preprocessing.
- Perform One-Hot Encoding and Standard Scaling.
- Use ColumnTransformer and Pipeline.
- Prevent data leakage.
- Compare baseline and full preprocessing models.

## Dataset
- Dataset: Titanic
- Records: 891
- Columns: 12
- Selected Features: 7
- Target: Survived

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Structure
```text
Day-06/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── preprocessing_feature_engineering.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── train.py
├── reports/
│   ├── figures/
│   ├── preprocessing_comparison.md
│   └── observations.md
├── models/
├── README.md
└── requirements.txt
```

## Preprocessing Techniques
1. Median Imputation
2. Most Frequent Imputation
3. One-Hot Encoding
4. Standard Scaling
5. ColumnTransformer
6. Scikit-learn Pipeline

## Model Comparison

| Model | Accuracy |
|---|---:|
| Baseline Pipeline | 80.45% |
| Full Preprocessing Pipeline | 80.45% |

## Results
Both models achieved approximately 80.45% accuracy. Scaling did not improve accuracy in this experiment.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the training script from the Day-06 directory:

```bash
python -m src.train
```

Open the notebook:

```bash
jupyter notebook
```

## Conclusion
A reusable preprocessing pipeline was developed using Scikit-learn to handle missing values, encode categorical features and scale numerical data.