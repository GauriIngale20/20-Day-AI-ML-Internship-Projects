from pathlib import Path
import pandas as pd


# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = BASE_DIR / "data" / "raw" / "titanic.csv"


def load_data():
    """Load the Titanic dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    return df


def inspect_data(df):
    """Inspect dataset information."""

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())


def analyze_class_distribution(df):
    """Analyze target class distribution."""

    target = "Survived"

    if target not in df.columns:
        raise ValueError(
            "Target column 'Survived' not found!"
        )

    print("\n========== CLASS DISTRIBUTION ==========")

    counts = df[target].value_counts().sort_index()

    percentages = (
        df[target].value_counts(normalize=True)
        .sort_index() * 100
    )

    for label in counts.index:
        class_name = (
            "Non-Survivors" if label == 0
            else "Survivors" if label == 1
            else f"Class {label}"
        )

        print(
            f"{class_name}: {counts[label]} "
            f"({percentages[label]:.2f}%)"
        )

    print("\nTotal Records:", len(df))

    majority = counts.max()
    minority = counts.min()

    print(
        "\nMajority-to-Minority Ratio:",
        f"{majority / minority:.2f}:1"
    )


def main():
    """Execute data loading and analysis."""

    df = load_data()

    inspect_data(df)

    analyze_class_distribution(df)


if __name__ == "__main__":
    main()
