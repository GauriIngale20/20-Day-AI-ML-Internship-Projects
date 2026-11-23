import pandas as pd

from src.data_loader import load_data
from config import REPORT_DIR


def analyze_data_quality(df):

    print("\n========== DATA QUALITY ANALYSIS ==========")

    print("\n1. Dataset Shape:")
    print(df.shape)

    print("\n2. Missing Values:")

    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100

    missing_report = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": missing_percentage
    })

    print(missing_report[missing_report["Missing_Count"] > 0])

    print("\n3. Duplicate Records:")
    print(df.duplicated().sum())

    print("\n4. Data Types:")
    print(df.dtypes)

    print("\n5. Unique Values:")
    print(df.nunique())

    print("\n6. Statistical Summary:")
    print(df.describe(include="all").T)

    return missing_report


def save_quality_report(df, missing_report):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "data_quality_report.md"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("# Titanic Dataset - Data Quality Report\n\n")

        file.write(f"Dataset Shape: {df.shape}\n\n")

        file.write("## Missing Values\n\n")
        file.write(missing_report.to_markdown())

        file.write("\n\n## Duplicate Records\n\n")
        file.write(str(df.duplicated().sum()))

        file.write("\n\n## Data Types\n\n")
        file.write(df.dtypes.to_frame("Data Type").to_markdown())

    print(f"\nReport saved successfully: {report_path}")


if __name__ == "__main__":

    data = load_data()

    missing_report = analyze_data_quality(data)

    save_quality_report(data, missing_report)
