
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data(df):
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["experience"] = pd.to_numeric(df["experience"], errors="coerce")
    return df


def plot_age_distribution(df):
    df = prepare_data(df)

    plt.figure(figsize=(8, 5))
    plt.hist(df["age"].dropna(), bins=8, edgecolor="black")
    plt.title("Age Distribution of Employees")
    plt.xlabel("Age")
    plt.ylabel("Number of Employees")
    plt.tight_layout()
    plt.show()


def plot_department_count(df):
    df = prepare_data(df)

    department_counts = df["department"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(department_counts.index, department_counts.values,
            edgecolor="black")
    plt.title("Employee Count by Department")
    plt.xlabel("Department")
    plt.ylabel("Number of Employees")
    plt.tight_layout()
    plt.show()


def plot_salary_distribution(df):
    df = prepare_data(df)

    plt.figure(figsize=(8, 5))
    plt.hist(df["salary"].dropna(), bins=8, edgecolor="black")
    plt.title("Salary Distribution of Employees")
    plt.xlabel("Salary")
    plt.ylabel("Number of Employees")
    plt.tight_layout()
    plt.show()


def plot_salary_by_department(df):
    df = prepare_data(df)

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="department", y="salary")
    plt.title("Salary Distribution Across Departments")
    plt.xlabel("Department")
    plt.ylabel("Salary")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    df = prepare_data(df)

    numeric_df = df[["age", "salary", "experience"]]
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Employee Data")
    plt.tight_layout()
    plt.show()
