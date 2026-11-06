import pandas as pd
from pathlib import Path


def load_employee_data(file_path):
    """
    Load employee data from a CSV file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)