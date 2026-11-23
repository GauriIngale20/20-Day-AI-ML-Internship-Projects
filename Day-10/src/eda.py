import pandas as pd

from src.data_loader import load_data
from config import REPORT_DIR


def perform_eda(df):

    print("\n========== EXPLORATORY DATA ANALYSIS ==========")

    # Survival distribution
    survival = df["survived"].value_counts()

    print("\n1. Survival Distribution:")
    print(survival)

    # Survival percentage
    print("\n2. Survival Percentage:")
    print((df["survived"].value_counts(normalize=True) * 100).round(2))

    # Survival by gender
    print("\n3. Survival by Gender:")
    print(df.groupby("sex", observed=True)["survived"].mean().mul(100).round(2))

    # Survival by passenger class
    print("\n4. Survival by Passenger Class:")
    print(df.groupby("pclass", observed=True)["survived"].mean().mul(100).round(2))

    # Age analysis
    print("\n5. Age Statistics:")
    print(df["age"].describe())

    # Fare analysis
    print("\n6. Fare Statistics:")
    print(df["fare"].describe())

    # Survival by embarkation port
    print("\n7. Survival by Embarkation Port:")
    print(df.groupby("embarked", observed=True)["survived"].mean().mul(100).round(2))

    # Numeric correlation
    print("\n8. Numeric Correlation:")
    print(df.select_dtypes(include="number").corr().round(2))


def save_eda_observations(df):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    female_survival = df.loc[df["sex"] == "female", "survived"].mean() * 100
    male_survival = df.loc[df["sex"] == "male", "survived"].mean() * 100

    class_survival = (
        df.groupby("pclass", observed=True)["survived"].mean() * 100
    )

    overall_survival = df["survived"].mean() * 100

    observations = [
        f"Overall survival rate was {overall_survival:.2f}%.",
        f"Female survival rate was {female_survival:.2f}%.",
        f"Male survival rate was {male_survival:.2f}%.",
        f"First-class survival rate was {class_survival.loc[1]:.2f}%.",
        f"Third-class survival rate was {class_survival.loc[3]:.2f}%.",
    ]

    report_path = REPORT_DIR / "eda_observations.md"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("# Titanic Dataset - EDA Observations\n\n")

        for number, observation in enumerate(observations, start=1):
            file.write(f"{number}. {observation}\n\n")

    print(f"\nEDA report saved: {report_path}")


if __name__ == "__main__":

    data = load_data()

    perform_eda(data)

    save_eda_observations(data)
