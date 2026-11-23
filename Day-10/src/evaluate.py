import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data_loader import load_data
from src.preprocessing import prepare_features
from src.split_data import split_dataset

from config import REPORT_DIR, MODEL_DIR, RANDOM_STATE

import joblib


def evaluate_best_model():

    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    model_path = MODEL_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found.")

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0))
    }

    print("\n========== FINAL TEST METRICS ==========")

    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\n========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y_test, predictions))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nMetrics saved successfully: {metrics_path}")


if __name__ == "__main__":
    evaluate_best_model()
