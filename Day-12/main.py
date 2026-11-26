from src.inference import run_inference
from src.batch_inference import run_batch_inference
from src.result_analyzer import analyze_results
from src.config import INPUT_DIR


def main():
    print("\n========== DAY 12: YOLO INFERENCE ==========")

    # Single image inference
    print("\n1. SINGLE IMAGE INFERENCE")
    image_path = INPUT_DIR / "bus.jpg"
    run_inference(image_path)

    # Batch inference
    print("\n2. BATCH INFERENCE")
    run_batch_inference()

    # Result analysis
    print("\n3. RESULT ANALYSIS")
    analyze_results()

    print("\nDay-12 execution completed successfully!")


if __name__ == "__main__":
    main()
