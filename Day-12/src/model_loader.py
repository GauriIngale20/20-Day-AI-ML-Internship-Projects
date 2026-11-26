from ultralytics import YOLO
from src.config import MODEL_PATH


def load_model():
    """
    Load the pretrained YOLOv8 model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found at: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    print("YOLO model loaded successfully!")
    print(f"Model location: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    model = load_model()
