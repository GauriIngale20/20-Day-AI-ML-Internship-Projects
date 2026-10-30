
import csv
from pathlib import Path


def count_csv_rows(file_path: str) -> int:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError("The given path is not a file.")

    if path.suffix.lower() != ".csv":
        raise ValueError("Please provide a CSV file.")

    with path.open(
        mode="r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.reader(file)

        next(reader, None)

        total_rows = sum(1 for row in reader if row)

    return total_rows
