import cv2
from PIL import Image
from pathlib import Path


def inspect_image(image_path: str):
    """
    Extract image dimensions, channels and format.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Unable to inspect image: {path}")

    height, width = image.shape[:2]

    if len(image.shape) == 2:
        channels = 1
        color_mode = "Grayscale"
    else:
        channels = image.shape[2]
        color_mode = "BGR" if channels == 3 else "BGRA"

    with Image.open(path) as pil_image:
        image_format = pil_image.format
        pil_mode = pil_image.mode

    return {
        "filename": path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "color_mode": color_mode,
        "pil_mode": pil_mode,
        "format": image_format,
        "file_size_kb": round(path.stat().st_size / 1024, 2)
    }
