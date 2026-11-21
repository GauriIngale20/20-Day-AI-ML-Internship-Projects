from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a classification model using multiple metrics."""

    # Predictions
    y_pred = model.predict(X_test)

    # Prediction probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_prob)

    pr_auc = average_precision_score(y_test, y_prob)

    # Confusion matrix
    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    # Display results
    print(f"\n========== {model_name.upper()} ==========")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Return results
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1])
    }


def main():

    print("\nLoading test dataset...")

    test_data = joblib.load(
        MODEL_DIR / "test_data.pkl"
    )

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    model_names = [
        "logistic_regression",
        "balanced_logistic_regression",
        "decision_tree"
    ]

    results = []

    print("\n========== MODEL EVALUATION ==========")

    for name in model_names:

        model_path = MODEL_DIR / f"{name}.pkl"

        model = joblib.load(model_path)

        result = evaluate_model(
            model,
            X_test,
            y_test,
            name
        )

        results.append(result)

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========")

    print(results_df.round(4).to_string(index=False))

    # Save metrics as CSV
    results_df.to_csv(
        REPORT_DIR / "model_metrics.csv",
        index=False
    )

    # Generate Markdown report
    report = "# Day 08 - Model Evaluation Report\n\n"

    report += "## Model Performance Comparison\n\n"

    report += results_df.round(4).to_markdown(index=False)

    report += """

## Evaluation Metrics

- Accuracy: Overall correct predictions.
- Precision: Correct positive predictions among predicted positives.
- Recall: Correct positive predictions among actual positives.
- F1-Score: Harmonic mean of precision and recall.
- ROC-AUC: Ability to distinguish between classes.
- PR-AUC: Area under the Precision-Recall curve.

## Confusion Matrix

- TN: True Negative
- FP: False Positive
- FN: False Negative
- TP: True Positive

## Conclusion

The three classification models were evaluated on the same
test dataset. Model performance should be compared using
multiple metrics rather than accuracy alone.
"""

    with open(
        REPORT_DIR / "metrics_report.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nReports saved successfully!")

    print("1. reports/model_metrics.csv")
    print("2. reports/metrics_report.md")


if __name__ == "__main__":
    main()
