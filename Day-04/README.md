# Day 04: Statistics, Splits, and Data Leakage

## Project Overview
This project focuses on statistical analysis, dataset splitting, reproducibility, and identifying data leakage in machine learning.

## Objectives
- Understand mean, median, variance, and standard deviation.
- Analyze numerical data distributions.
- Create training, validation, and testing datasets.
- Apply stratified splitting.
- Ensure reproducibility using random_state.
- Identify duplicate contamination and data leakage risks.

## Dataset
**Breast Cancer Wisconsin Dataset**

- Total Records: 569
- Numerical Features: 30
- Target Column: diagnosis
- Source: Scikit-learn built-in dataset

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
Day-04/
├── data/
│   └── breast_cancer.csv
├── notebooks/
│   └── statistics_splits_leakage.ipynb
├── reports/
│   ├── plots/
│   ├── split_strategy.md
│   ├── leakage_checklist.md
│   └── statistics_summary.md
├── .gitignore
├── README.md
└── requirements.txt
```

## Key Tasks Completed
1. Dataset loading and inspection.
2. Statistical analysis.
3. Data distribution visualization.
4. Train, validation, and test splitting.
5. Reproducibility verification.
6. Duplicate contamination audit.
7. Data leakage checklist and reports.

## Split Strategy

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Stratified splitting was used with `random_state=42`.

## Audit Results
- Total duplicate records: 0
- Training duplicates: 0
- Validation duplicates: 0
- Testing duplicates: 0
- Train-test overlapping records: 0

## Conclusion
This project demonstrates how to perform statistical analysis, create reliable dataset splits, maintain reproducibility, and identify data leakage risks before building machine learning models.