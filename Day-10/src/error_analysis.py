import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix

from src.data_loader import load_data
from src.split_data import split_dataset

from config import MODEL_DIR, REPORT_DIR


def perform_error_analysis():

    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    model = joblib.load(MODEL_DIR / "best_model.joblib")

    predictions = model.predict(X_test)

    # Create prediction analysis dataframe
    results = X_test.copy()

    results["Actual"] = y_test.values
    results["Predicted"] = predictions

    results["Error_Type"] = results.apply(
        lambda row: (
            "False Positive"
            if row["Actual"] == 0 and row["Predicted"] == 1
            else "False Negative"
            if row["Actual"] == 1 and row["Predicted"] == 0
            else "Correct"
        ),
        axis=1
    )

    errors = results[results["Error_Type"] != "Correct"].copy()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save prediction errors
    error_path = REPORT_DIR / "error_analysis.csv"
    errors.to_csv(error_path, index=False)

    print("\n========== ERROR ANALYSIS ==========")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nPrediction Summary:")
    print(results["Error_Type"].value_counts())

    print("\nError Count:", len(errors))
    print("Correct Predictions:", (results["Error_Type"] == "Correct").sum())

    print("\nErrors by Gender:")
    print(errors.groupby("sex").size())

    print("\nErrors by Passenger Class:")
    print(errors.groupby("pclass").size())

    print(f"\nError report saved: {error_path}")


if __name__ == "__main__":

    perform_error_analysis()
