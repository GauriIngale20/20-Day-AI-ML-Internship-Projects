from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

sns.set_theme(style="whitegrid")


def plot_group_metrics():
    file_path = REPORT_DIR / "group_metrics.csv"
    df = pd.read_csv(file_path)

    for group_name in ["Sex", "Pclass", "AgeGroup"]:
        subset = df[df["Group"] == group_name]

        if subset.empty:
            continue

        plt.figure(figsize=(10, 5))

        sns.barplot(
            data=subset,
            x="Category",
            y="F1 Score",
            hue="Model",
        )

        plt.title(f"Model F1 Score by {group_name}")
        plt.xlabel(group_name)
        plt.ylabel("F1 Score")
        plt.ylim(0, 1)
        plt.legend(title="Model")
        plt.tight_layout()

        output_path = FIGURE_DIR / f"f1_score_by_{group_name.lower()}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path.name}")


def plot_error_counts():
    file_path = REPORT_DIR / "error_analysis_examples.csv"
    df = pd.read_csv(file_path)

    if df.empty:
        print("No prediction errors found.")
        return

    summary = (
        df.groupby(["Model", "Error_Type"])
        .size()
        .reset_index(name="Count")
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=summary,
        x="Model",
        y="Count",
        hue="Error_Type",
    )

    plt.title("False Positive and False Negative Counts")
    plt.xlabel("Model")
    plt.ylabel("Number of Errors")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = FIGURE_DIR / "error_type_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path.name}")


def plot_feature_importance():
    file_path = REPORT_DIR / "permutation_importance.csv"
    df = pd.read_csv(file_path)

    df = df.sort_values("Importance_Mean", ascending=True)

    plt.figure(figsize=(9, 6))

    plt.barh(
        df["Feature"],
        df["Importance_Mean"],
        xerr=df["Importance_Std"],
    )

    plt.title("Permutation Feature Importance")
    plt.xlabel("Mean Decrease in F1 Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_path = FIGURE_DIR / "permutation_feature_importance.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path.name}")


def main():
    plot_group_metrics()
    plot_error_counts()
    plot_feature_importance()

    print("\nAll visualizations created successfully!")


if __name__ == "__main__":
    main()
