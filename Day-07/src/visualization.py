
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.evaluate import evaluate_models


def create_visualizations():

    results = evaluate_models()

    project_root = Path(__file__).resolve().parent.parent
    output_folder = project_root / "reports" / "figures"
    output_folder.mkdir(parents=True, exist_ok=True)

    models = results["Model"].tolist()

    # 1. Accuracy comparison
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        results["Training Accuracy"],
        width,
        label="Training Accuracy"
    )

    plt.bar(
        x + width / 2,
        results["Testing Accuracy"],
        width,
        label="Testing Accuracy"
    )

    plt.xticks(x, models)
    plt.ylabel("Accuracy")
    plt.title("Training vs Testing Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_folder / "accuracy_comparison.png")
    plt.close()

    # 2. Evaluation metrics comparison
    metrics = ["Precision", "Recall", "F1 Score"]

    plt.figure(figsize=(9, 5))

    for metric in metrics:
        plt.plot(
            models,
            results[metric],
            marker="o",
            label=metric
        )

    plt.ylabel("Score")
    plt.title("Model Evaluation Metrics Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_folder / "metrics_comparison.png")
    plt.close()

    # 3. Cross-validation comparison
    plt.figure(figsize=(8, 5))

    plt.bar(
        models,
        results["Mean CV Accuracy"]
    )

    plt.ylabel("Mean CV Accuracy")
    plt.title("5-Fold Cross-Validation Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(output_folder / "cross_validation_comparison.png")
    plt.close()

    print("\nVisualizations saved successfully!")
    print("Location:", output_folder)


if __name__ == "__main__":
    create_visualizations()
