from pathlib import Path
import cv2


def load_image(image_path: str):
    """
    Load an image using OpenCV.

    Args:
        image_path: Path of the image.

    Returns:
        Loaded image in BGR format.

    Raises:
        FileNotFoundError: If image does not exist.
        ValueError: If image cannot be decoded.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Unable to load image: {path}")

    return image


def get_image_files(folder_path: str):
    """
    Return all supported image files from a folder.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    return sorted(
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    )
