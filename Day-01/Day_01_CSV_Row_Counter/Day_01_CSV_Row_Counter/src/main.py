
import argparse
from pathlib import Path

from src.csv_counter import count_csv_rows


def main():
    parser = argparse.ArgumentParser(
        description="CSV Row Counter - Count data rows in a CSV file."
    )

    parser.add_argument(
        "file",
        help="Path of the CSV file"
    )

    args = parser.parse_args()

    try:
        total_rows = count_csv_rows(args.file)

        print("\n========== CSV ROW COUNTER ==========")
        print(f"File Name       : {Path(args.file).name}")
        print(f"Total Data Rows : {total_rows}")
        print("=====================================\n")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
