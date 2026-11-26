import pandas as pd
from ultralytics import YOLO

from src.config import (
    MODEL_PATH,
    INPUT_DIR,
    ANNOTATED_DIR,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    IMAGE_EXTENSIONS
)


def run_batch_inference():

    # Load model once
    model = YOLO(str(MODEL_PATH))

    # Get all input images
    images = [
        path for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError("No input images found!")

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    detection_records = []
    summary_records = []

    # Process each image
    for image_path in images:

        results = model.predict(
            source=str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        result = results[0]

        output_path = (
            ANNOTATED_DIR /
            f"{image_path.stem}_detected.jpg"
        )

        result.save(filename=str(output_path))

        object_count = len(result.boxes)

        summary_records.append({
            "Image": image_path.name,
            "Total_Objects": object_count
        })

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection_records.append({
                "Image": image_path.name,
                "Class_ID": class_id,
                "Class_Name": class_name,
                "Confidence": round(confidence, 4),
                "X1": round(x1, 2),
                "Y1": round(y1, 2),
                "X2": round(x2, 2),
                "Y2": round(y2, 2)
            })

        print(
            f"Processed: {image_path.name} | "
            f"Objects: {object_count}"
        )

    # Save CSV reports
    detection_df = pd.DataFrame(detection_records)
    summary_df = pd.DataFrame(summary_records)

    detection_df.to_csv(
        "outputs/detection_results.csv",
        index=False
    )

    summary_df.to_csv(
        "outputs/batch_summary.csv",
        index=False
    )

    print("\n========== BATCH INFERENCE SUMMARY ==========")
    print(f"Total Images: {len(images)}")
    print(f"Total Objects: {len(detection_records)}")
    print("Detection results saved successfully!")


if __name__ == "__main__":
    run_batch_inference()
