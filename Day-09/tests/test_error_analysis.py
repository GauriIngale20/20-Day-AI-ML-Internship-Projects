import pandas as pd
import pytest

from src.error_analysis import (
    load_data,
    load_models,
    calculate_metrics,
    analyze_errors,
    analyze_groups,
)


def test_load_data():
    original_data, X_test, y_test, original_ids = load_data()

    assert not original_data.empty
    assert X_test.shape[0] == 179
    assert len(y_test) == 179
    assert len(original_ids) == 179


def test_load_models():
    models = load_models()

    assert len(models) == 3
    assert "Logistic Regression" in models
    assert "Balanced Logistic Regression" in models
    assert "Decision Tree" in models


def test_calculate_metrics():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 0, 1]

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["Accuracy"] == pytest.approx(0.8)
    assert metrics["Precision"] == pytest.approx(1.0)
    assert metrics["Recall"] == pytest.approx(2 / 3)
    assert metrics["F1 Score"] == pytest.approx(0.8)


def test_analyze_errors():
    _, X_test, y_test, original_ids = load_data()
    models = load_models()

    errors = analyze_errors(models, X_test, y_test, original_ids)

    assert isinstance(errors, pd.DataFrame)
    assert "Error_Type" in errors.columns
    assert "PassengerId" in errors.columns


def test_analyze_groups():
    _, X_test, y_test, _ = load_data()
    models = load_models()

    result = analyze_groups(models, X_test, y_test)

    assert not result.empty
    assert "Group" in result.columns
    assert "F1 Score" in result.columns
    assert set(result["Group"].unique()) == {
        "Sex", "Pclass", "AgeGroup"
    }
