import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data
from config import FIGURES_DIR, FIGURE_DPI

sns.set_theme(style="whitegrid")


def create_visualizations(df):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Survival distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="survived")
    plt.title("Titanic Survival Distribution")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "survival_distribution.png", dpi=FIGURE_DPI)
    plt.close()

    # 2. Survival by gender
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="sex", y="survived", errorbar=None)
    plt.title("Survival Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "survival_by_gender.png", dpi=FIGURE_DPI)
    plt.close()

    # 3. Survival by passenger class
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pclass", y="survived", errorbar=None)
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "survival_by_class.png", dpi=FIGURE_DPI)
    plt.close()

    # 4. Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="age", hue="survived", bins=30, kde=True)
    plt.title("Age Distribution by Survival")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "age_distribution.png", dpi=FIGURE_DPI)
    plt.close()

    # 5. Correlation heatmap
    plt.figure(figsize=(10, 7))
    correlation = df.select_dtypes(include="number").corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Titanic Dataset Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=FIGURE_DPI)
    plt.close()

    print("\nAll 5 visualizations created successfully!")

    for file in FIGURES_DIR.glob("*.png"):
        print("Saved:", file.name)


if __name__ == "__main__":

    data = load_data()
    create_visualizations(data)
