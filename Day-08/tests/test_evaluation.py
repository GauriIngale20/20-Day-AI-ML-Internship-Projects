from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required_files = [
        "data/raw/titanic.csv",
        "models/logistic_regression.pkl",
        "models/balanced_logistic_regression.pkl",
        "models/decision_tree.pkl",
        "models/test_data.pkl",
        "reports/model_metrics.csv",
        "reports/threshold_results.csv",
        "reports/metrics_report.md",
        "reports/threshold_analysis.md",
        "reports/error_cost_analysis.md",
    ]

    for file_path in required_files:
        assert (BASE_DIR / file_path).exists(), f"Missing file: {file_path}"


def test_dataset_is_not_empty():
    data = pd.read_csv(BASE_DIR / "data/raw/titanic.csv")

    assert not data.empty
    assert "Survived" in data.columns


def test_model_files_load():
    model_files = [
        "models/logistic_regression.pkl",
        "models/balanced_logistic_regression.pkl",
        "models/decision_tree.pkl",
    ]

    for file_path in model_files:
        model = joblib.load(BASE_DIR / file_path)

        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


def test_metrics_report_has_required_columns():
    metrics = pd.read_csv(BASE_DIR / "reports/model_metrics.csv")

    required_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
    ]

    for column in required_columns:
        assert column in metrics.columns

    assert len(metrics) >= 3


def test_threshold_results_are_valid():
    results = pd.read_csv(BASE_DIR / "reports/threshold_results.csv")

    required_columns = [
        "Model",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    for column in required_columns:
        assert column in results.columns

    assert len(results) == 15
    assert results["Threshold"].between(0, 1).all()
    assert results["F1-Score"].between(0, 1).all()
