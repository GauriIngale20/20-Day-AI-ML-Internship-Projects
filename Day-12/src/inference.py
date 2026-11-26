from ultralytics import YOLO
from src.config import (
    MODEL_PATH,
    INPUT_DIR,
    ANNOTATED_DIR,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE
)


def run_inference(image_path):
    """
    Run YOLO object detection on a single image.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        save=False,
        verbose=False
    )

    result = results[0]

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ANNOTATED_DIR / f"{image_path.stem}_detected.jpg"

    result.save(filename=str(output_path))

    print("\n========== YOLO DETECTION RESULTS ==========")
    print(f"Image: {image_path.name}")
    print(f"Detected objects: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(
            f"Object: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

    print(f"\nAnnotated image saved: {output_path}")

    return result


if __name__ == "__main__":
    image_path = INPUT_DIR / "bus.jpg"
    run_inference(image_path)
