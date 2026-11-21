# Day 09: Error Analysis and Explainability

## Project Overview
This project performs error analysis and model explainability on the Titanic survival prediction dataset.

The objective is to understand where machine learning models make incorrect predictions and evaluate their performance across different passenger groups.

## Objectives
- Analyze False Positives and False Negatives.
- Compare classification model performance.
- Evaluate model performance across meaningful groups.
- Understand feature importance using permutation importance.
- Propose improvements based on model errors.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Pytest

## Dataset
Titanic dataset containing passenger information such as:
- Passenger class
- Sex
- Age
- Number of siblings and spouses
- Number of parents and children
- Fare
- Embarked port

Test dataset contains 179 records.

## Models Used
1. Logistic Regression
2. Balanced Logistic Regression
3. Decision Tree

## Overall Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 |
| Balanced Logistic Regression | 0.804 | 0.730 | 0.783 | 0.755 |
| Decision Tree | 0.765 | 0.755 | 0.580 | 0.656 |

Balanced Logistic Regression achieved the highest F1 Score of 0.755.

## Error Analysis
The project identifies:
- False Positives
- False Negatives

Error examples are saved in `reports/error_analysis_examples.csv`.

## Group-wise Analysis
Model performance is evaluated across:
- Sex
- Passenger Class
- Age Groups

Results are saved in `reports/group_metrics.csv`.

## Feature Importance
Permutation importance is used to understand how much each feature contributes to model performance.

The Sex feature showed the highest importance in this experiment.

## Visualizations
Generated plots:
1. F1 Score by Sex
2. F1 Score by Passenger Class
3. F1 Score by Age Group
4. False Positive vs False Negative Comparison
5. Permutation Feature Importance

All plots are stored in `reports/figures/`.

## Proposed Improvements
1. Investigate groups with lower F1 scores and collect or review more representative training examples.
2. Experiment with different prediction thresholds and compare precision and recall.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run error analysis:

```bash
python -m src.error_analysis
```

Generate visualizations:

```bash
python -m src.visualization
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Testing
All 5 automated tests passed successfully.

## Conclusion
This project demonstrates how error analysis and explainability help understand machine learning model behavior beyond overall accuracy.
