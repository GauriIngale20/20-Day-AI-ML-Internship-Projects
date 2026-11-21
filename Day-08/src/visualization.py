from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():

    test_data = joblib.load(
        MODEL_DIR / "test_data.pkl"
    )

    return test_data["X_test"], test_data["y_test"]


def plot_confusion_matrices(X_test, y_test):

    model_names = [
        "logistic_regression",
        "balanced_logistic_regression",
        "decision_tree"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, name in zip(axes, model_names):

        model = joblib.load(MODEL_DIR / f"{name}.pkl")

        predictions = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"],
            ax=ax
        )

        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

    plt.suptitle("Confusion Matrix Comparison", fontsize=15)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "confusion_matrix_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix comparison saved!")


def plot_roc_curves(X_test, y_test):

    model_names = [
        "logistic_regression",
        "balanced_logistic_regression",
        "decision_tree"
    ]

    plt.figure(figsize=(9, 6))

    for name in model_names:

        model = joblib.load(MODEL_DIR / f"{name}.pkl")

        probabilities = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(
            y_test,
            probabilities
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name.replace('_', ' ').title()} (AUC={roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("ROC curve saved!")


def plot_precision_recall_curves(X_test, y_test):

    model_names = [
        "logistic_regression",
        "balanced_logistic_regression",
        "decision_tree"
    ]

    plt.figure(figsize=(9, 6))

    for name in model_names:

        model = joblib.load(MODEL_DIR / f"{name}.pkl")

        probabilities = model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(
            y_test,
            probabilities
        )

        pr_auc = auc(recall, precision)

        plt.plot(
            recall,
            precision,
            label=f"{name.replace('_', ' ').title()} (AUC={pr_auc:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Precision-Recall curve saved!")


def plot_model_comparison():

    metrics_path = REPORT_DIR / "model_metrics.csv"

    results = pd.read_csv(metrics_path)

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC"
    ]

    comparison = results.set_index("Model")[metrics]

    comparison.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Classification Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Models")

    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)

    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "model_performance_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Model comparison chart saved!")


def main():

    print("\nLoading test data...")

    X_test, y_test = load_test_data()

    print("\n========== GENERATING VISUALIZATIONS ==========")

    plot_confusion_matrices(X_test, y_test)

    plot_roc_curves(X_test, y_test)

    plot_precision_recall_curves(X_test, y_test)

    plot_model_comparison()

    print("\nAll visualizations generated successfully!")

    print(f"Saved inside: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
