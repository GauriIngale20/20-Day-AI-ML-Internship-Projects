
import pandas as pd
from pathlib import Path


def load_data():
    """Load the Titanic dataset."""

    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "raw" / "titanic.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    load_data()
