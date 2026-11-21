from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Threshold values
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def analyze_threshold(model, X_test, y_test, model_name):
    """Evaluate a model at different classification thresholds."""

    # Get probability of positive class
    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    print(f"\n========== {model_name.upper()} ==========")

    for threshold in THRESHOLDS:

        # Apply custom threshold
        predictions = (
            probabilities >= threshold
        ).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        result = {
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        }

        results.append(result)

        print(f"\nThreshold: {threshold}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")

    return results


def plot_threshold_comparison(results_df):

    models = results_df["Model"].unique()

    for model_name in models:

        model_data = results_df[
            results_df["Model"] == model_name
        ]

        plt.figure(figsize=(9, 5))

        for metric in ["Precision", "Recall", "F1-Score"]:

            plt.plot(
                model_data["Threshold"],
                model_data[metric],
                marker="o",
                linewidth=2,
                label=metric
            )

        plt.title(
            f"Threshold Analysis - {model_name.replace('_', ' ').title()}"
        )

        plt.xlabel("Classification Threshold")
        plt.ylabel("Score")

        plt.xticks(THRESHOLDS)
        plt.ylim(0, 1.05)

        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()

        filename = f"{model_name}_threshold_analysis.png"

        plt.savefig(
            FIGURE_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved plot: {filename}")


def generate_threshold_report(results_df):

    report = "# Day 08 - Threshold Analysis Report\n\n"

    report += "## Threshold Performance Comparison\n\n"

    report += results_df.round(4).to_markdown(index=False)

    report += "\n\n## Threshold Analysis Observations\n\n"

    for model_name in results_df["Model"].unique():

        model_data = results_df[
            results_df["Model"] == model_name
        ]

        best_f1 = model_data.loc[
            model_data["F1-Score"].idxmax()
        ]

        report += f"### {model_name.replace('_', ' ').title()}\n\n"

        report += (
            f"- Best F1-score threshold: "
            f"{best_f1['Threshold']}\n"
        )

        report += (
            f"- Precision: {best_f1['Precision']:.4f}\n"
        )

        report += (
            f"- Recall: {best_f1['Recall']:.4f}\n"
        )

        report += (
            f"- F1-score: {best_f1['F1-Score']:.4f}\n\n"
        )

    report += """
## Threshold Selection Explanation

A lower classification threshold generally increases
the number of positive predictions and can improve recall.

A higher threshold generally reduces positive predictions
and can improve precision.

The best threshold depends on the problem and the cost
of false positive and false negative predictions.

The threshold with the highest F1-score provides one
possible balance between precision and recall. It is
not automatically the best threshold for every use case.

## Conclusion

Threshold selection changes the classification results.
Therefore, model evaluation should consider multiple
thresholds instead of relying only on the default 0.5.
"""

    with open(
        REPORT_DIR / "threshold_analysis.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nThreshold report saved successfully!")


def main():

    print("\nLoading test data...")

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

    all_results = []

    print("\n========== THRESHOLD ANALYSIS ==========")

    for name in model_names:

        model = joblib.load(
            MODEL_DIR / f"{name}.pkl"
        )

        results = analyze_threshold(
            model,
            X_test,
            y_test,
            name
        )

        all_results.extend(results)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    print("\n========== COMPLETE THRESHOLD COMPARISON ==========")

    print(results_df.round(4).to_string(index=False))

    # Save CSV
    results_df.to_csv(
        REPORT_DIR / "threshold_results.csv",
        index=False
    )

    print("\nThreshold results CSV saved!")

    # Generate graphs
    plot_threshold_comparison(results_df)

    # Generate Markdown report
    generate_threshold_report(results_df)

    print("\n========== THRESHOLD ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
