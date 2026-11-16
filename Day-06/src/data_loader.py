
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "titanic.csv"


def load_data():
    """Load the Titanic dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df


def prepare_features_target(df):
    """Separate input features and target variable."""

    X = df[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ].copy()

    y = df["Survived"].copy()

    return X, y
