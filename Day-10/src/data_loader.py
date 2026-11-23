import pandas as pd

from config import DATASET_PATH, TARGET_COLUMN


def load_data():
    """Load the Titanic dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    return df


def display_dataset_info(df):
    """Display basic dataset information."""

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTarget Distribution:")
    print(df[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    data = load_data()
    display_dataset_info(data)
