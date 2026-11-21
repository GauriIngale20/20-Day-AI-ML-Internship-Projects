from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.inspection import permutation_importance


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "titanic.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    original_data = pd.read_csv(DATA_PATH)
    test_data = joblib.load(MODEL_DIR / "test_data.pkl")

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    original_ids = original_data.loc[X_test.index, "PassengerId"]

    return original_data, X_test, y_test, original_ids


def load_models():
    model_names = {
        "Logistic Regression": "logistic_regression.pkl",
        "Balanced Logistic Regression": "balanced_logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
    }

    models = {}

    for name, filename in model_names.items():
        models[name] = joblib.load(MODEL_DIR / filename)

    return models


def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }


def analyze_errors(models, X_test, y_test, original_ids):
    error_records = []
    metric_records = []

    for model_name, model in models.items():
        predictions = model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)
        metrics["Model"] = model_name
        metric_records.append(metrics)

        for position, (actual, predicted) in enumerate(
            zip(y_test.to_numpy(), predictions)
        ):
            if actual != predicted:
                error_type = (
                    "False Positive" if predicted == 1
                    else "False Negative"
                )

                error_records.append({
                    "PassengerId": original_ids.iloc[position],
                    "Original_Index": X_test.index[position],
                    "Model": model_name,
                    "Actual": actual,
                    "Predicted": predicted,
                    "Error_Type": error_type,
                    "Sex": X_test.iloc[position]["Sex"],
                    "Pclass": X_test.iloc[position]["Pclass"],
                    "Age": X_test.iloc[position]["Age"],
                    "Fare": X_test.iloc[position]["Fare"],
                })

    errors_df = pd.DataFrame(error_records)
    metrics_df = pd.DataFrame(metric_records)

    errors_df.to_csv(
        REPORT_DIR / "error_analysis_examples.csv", index=False
    )
    metrics_df.to_csv(
        REPORT_DIR / "overall_metrics.csv", index=False
    )

    print("\nOverall Model Metrics:")
    print(metrics_df.round(3).to_string(index=False))

    print("\nError examples saved successfully.")

    return errors_df


def analyze_groups(models, X_test, y_test):
    group_data = X_test.copy()
    group_data["Actual"] = y_test

    group_data["AgeGroup"] = pd.cut(
        group_data["Age"],
        bins=[-np.inf, 18, 35, 60, np.inf],
        labels=["0-18", "19-35", "36-60", "60+"],
    ).astype("object").fillna("Unknown")

    records = []

    for model_name, model in models.items():
        predictions = model.predict(X_test)

        for group_name in ["Sex", "Pclass", "AgeGroup"]:
            for category in group_data[group_name].unique():
                mask = group_data[group_name] == category

                if mask.sum() == 0:
                    continue

                actual = y_test.loc[mask]
                predicted = predictions[mask.to_numpy()]

                metrics = calculate_metrics(actual, predicted)

                records.append({
                    "Model": model_name,
                    "Group": group_name,
                    "Category": str(category),
                    "Samples": int(mask.sum()),
                    **metrics,
                })

    result = pd.DataFrame(records)
    result.to_csv(REPORT_DIR / "group_metrics.csv", index=False)

    print("\nGroup-wise metrics saved successfully.")

    return result


def calculate_feature_importance(model, X_test, y_test):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    importance_df.to_csv(
        REPORT_DIR / "permutation_importance.csv", index=False
    )

    print("\nPermutation feature importance:")
    print(importance_df.round(4).to_string(index=False))

    return importance_df


def create_report(metrics_df, errors_df, group_df, importance_df):
    report_path = REPORT_DIR / "error_analysis.md"

    total_errors = len(errors_df)

    false_positives = (
        errors_df["Error_Type"] == "False Positive"
    ).sum()

    false_negatives = (
        errors_df["Error_Type"] == "False Negative"
    ).sum()

    best_model = metrics_df.sort_values(
        "F1 Score", ascending=False
    ).iloc[0]

    top_features = importance_df.head(5)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# Day 09: Error Analysis and Explainability\n\n")

        file.write("## 1. Overall Model Performance\n\n")
        file.write(metrics_df.round(3).to_string(index=False))
        file.write("\n\n")

        file.write("## 2. Error Analysis\n\n")
        file.write(f"- Total errors across all models: {total_errors}\n")
        file.write(f"- False Positives: {false_positives}\n")
        file.write(f"- False Negatives: {false_negatives}\n\n")

        file.write("## 3. Group-wise Analysis\n\n")
        file.write(
            "Performance was evaluated across Sex, Passenger Class, "
            "and Age Group.\n\n"
        )
        file.write(group_df.round(3).to_string(index=False))
        file.write("\n\n")

        file.write("## 4. Permutation Feature Importance\n\n")
        file.write(top_features.round(4).to_string(index=False))
        file.write("\n\n")
        file.write(
            "Permutation importance measures how much model performance "
            "changes when a feature is shuffled. It indicates model "
            "dependence, not causation.\n\n"
        )

        file.write("## 5. Two Proposed Improvements\n\n")
        file.write(
            "1. Investigate groups with lower F1 scores and collect or "
            "review more representative training examples for those groups.\n"
        )
        file.write(
            "2. Experiment with decision thresholds and compare precision "
            "and recall before selecting a suitable threshold.\n\n"
        )

        file.write("## 6. Important Caveat\n\n")
        file.write(
            "Feature importance shows association with model predictions, "
            "not proof of causation. Group metrics may also be less reliable "
            "for groups with very few samples.\n"
        )

    print(f"\nReport created: {report_path}")


def main():
    print("Loading Titanic test data and models...")

    _, X_test, y_test, original_ids = load_data()
    models = load_models()

    errors_df = analyze_errors(
        models, X_test, y_test, original_ids
    )

    metrics_df = pd.DataFrame([
        {
            "Model": name,
            **calculate_metrics(y_test, model.predict(X_test)),
        }
        for name, model in models.items()
    ])

    group_df = analyze_groups(models, X_test, y_test)

    importance_df = calculate_feature_importance(
        models["Logistic Regression"], X_test, y_test
    )

    create_report(
        metrics_df, errors_df, group_df, importance_df
    )

    print("\nDay 09 error analysis completed successfully!")


if __name__ == "__main__":
    main()
