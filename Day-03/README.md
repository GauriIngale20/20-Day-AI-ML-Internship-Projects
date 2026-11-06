# Day 03 – Exploratory Data Analysis (EDA)

## Project Overview
This project focuses on Exploratory Data Analysis (EDA) using Python, Pandas, NumPy, Matplotlib, and Seaborn.

The analysis is performed on an employee dataset to understand data distributions, department-wise employee counts, salary patterns, and relationships between numerical variables.

## Objectives
- Understand the structure of the dataset.
- Perform data cleaning and preparation.
- Calculate summary statistics.
- Analyze employee age and salary distributions.
- Compare employee counts across departments.
- Identify relationships using correlation analysis.
- Create meaningful data visualizations.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure
```text
Day-03/
├── data/
│   └── employees.csv
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── reports/
│   ├── eda_observations.md
│   └── plots/
│       ├── age_distribution.png
│       ├── department_employee_count.png
│       ├── salary_distribution.png
│       ├── salary_by_department.png
│       └── correlation_heatmap.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── visualization.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Visualizations
1. Age Distribution Histogram
2. Department-wise Employee Count
3. Salary Distribution Histogram
4. Salary Distribution Across Departments (Box Plot)
5. Correlation Heatmap

## Key Findings
- The dataset contains 99 employee records.
- Average employee age is approximately 32 years.
- IT has the highest employee count with 27 employees.
- Average employee salary is approximately ₹47,948.
- Age, salary, and experience show very weak linear correlations.

## Learning Outcomes
- Learned data cleaning and preprocessing using Pandas.
- Practiced descriptive statistical analysis.
- Created visualizations using Matplotlib and Seaborn.
- Understood correlation analysis.
- Developed the ability to identify meaningful patterns in data.

## How to Run
1. Clone or download the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Open the Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `notebooks/exploratory_data_analysis.ipynb` and run the cells.

## Conclusion
This project demonstrates the process of exploring and understanding employee data through statistical analysis and visualization.
