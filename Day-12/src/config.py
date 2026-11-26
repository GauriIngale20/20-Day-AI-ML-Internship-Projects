from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Important directories
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input_images"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"

# YOLO model configuration
MODEL_PATH = MODEL_DIR / "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# Supported image formats
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

