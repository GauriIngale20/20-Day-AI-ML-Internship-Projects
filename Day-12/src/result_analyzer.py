import pandas as pd
import matplotlib.pyplot as plt

from src.config import BASE_DIR, OUTPUT_DIR


def analyze_results():

    detection_file = OUTPUT_DIR / "detection_results.csv"
    summary_file = OUTPUT_DIR / "batch_summary.csv"

    if not detection_file.exists():
        raise FileNotFoundError("Detection results CSV not found!")

    if not summary_file.exists():
        raise FileNotFoundError("Batch summary CSV not found!")

    detection_df = pd.read_csv(detection_file)
    summary_df = pd.read_csv(summary_file)

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class distribution
    class_counts = detection_df["Class_Name"].value_counts()

    plt.figure(figsize=(9, 5))
    class_counts.plot(kind="bar")
    plt.title("Object Class Distribution")
    plt.xlabel("Object Class")
    plt.ylabel("Detection Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / "class_distribution.png")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # 2. Image-wise object count
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["Image"], summary_df["Total_Objects"])
    plt.title("Image-wise Object Count")
    plt.xlabel("Image")
    plt.ylabel("Number of Objects")
    plt.tight_layout()
    plt.savefig(plot_dir / "image_object_count.png")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # 3. Confidence distribution
    plt.figure(figsize=(8, 5))
    plt.hist(detection_df["Confidence"], bins=5)
    plt.title("Detection Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(plot_dir / "confidence_distribution.png")
    plt.show(block=False)
    plt.pause(2)
    plt.close()


    print("\n========== DETECTION ANALYSIS ==========")
    print("\nClass Distribution:")
    print(class_counts)

    print("\nImage Summary:")
    print(summary_df)

    print("\nAverage Confidence:",
          round(detection_df["Confidence"].mean(), 4))

    print("\nAll visualization plots saved successfully!")


if __name__ == "__main__":
    analyze_results()
